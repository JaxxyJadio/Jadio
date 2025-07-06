"""
main.py - Main entry point for Jadio-quick-llm

This module starts the FastAPI server, serves the UI, exposes all API endpoints, and integrates all core modules (model loading, chat management, config, etc.).

USAGE:
- Run this script directly (python main.py) to start the server.
- The server is intended for LAN-only, personal use. It exposes open endpoints and permissive CORS for trusted networks.
- All configuration is loaded from config.yaml via config.py.
- The server provides endpoints for chat, file browsing, tool execution, and status.

SECURITY:
- Minimal by default. Do NOT expose to the public internet without strong authentication and rate limiting.
- Password protection and rate limiting are configurable in config.yaml.

COMPONENTS:
- FastAPI: Web framework for API and UI.
- Uvicorn: ASGI server for running FastAPI.
- Jinja2: Templating for frontend UI.
- ModelLoader: Handles model and tokenizer loading/unloading.
- ChatManager: Manages chat sessions, message formatting, and agent tool execution.
- Config: Loads and validates all configuration values.
- Logging: All major actions and errors are logged for debugging and audit.

FUTURE MAINTAINERS:
- Read all docstrings and comments for detailed explanations of each function and endpoint.
- See README.md and flowguide.md for full architecture and rationale.
"""

# --- Imports ---
from fastapi import FastAPI, Request, Response, HTTPException, Body, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, constr, ValidationError
from .error_utils import error_service_unavailable, error_unauthorized, json_error
import uvicorn
import os
import socket
from .model_loader import ModelLoader
from .chat_manager import ChatManager
from .config import (
    PORT, PROJECT_PATH, GIT_REPO_URL, MAX_NEW_TOKENS, DEBUG, AUTH_PASSWORD, CORS_ORIGINS, logger, RATE_LIMIT, ALLOWED_TOOLS, MAX_MESSAGE_LENGTH
)
import git
from collections import defaultdict, deque
import time
import importlib
import sys
from typing import List, Optional
import json

# --- Structured Logging Helper ---
def log_event(event: str, **kwargs):
    logger.info(json.dumps({"event": event, **kwargs}))

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Startup Security Warnings ---
if AUTH_PASSWORD is None:
    logger.warning("AUTH_PASSWORD is not set. Server is UNPROTECTED. Anyone on the LAN can access all endpoints.")
if RATE_LIMIT is None:
    logger.warning("Rate limiting DISABLED. Server may be vulnerable to abuse from LAN clients.")
if CORS_ORIGINS == ["*"]:
    logger.warning("CORS_ORIGINS is set to '*'. All origins are allowed. This is unsafe for public deployments.")

# --- Ban List for Rate Limiting ---
BANNED_IPS = set()

# --- Templates ---
templates_dir = "templates"
if not os.path.isdir(templates_dir):
    os.makedirs(templates_dir, exist_ok=True)
    with open(os.path.join(templates_dir, "frontend.html"), "w", encoding="utf-8") as f:
        f.write("<html><body><h1>Jadio-quick-llm UI</h1></body></html>")
templates = Jinja2Templates(directory=templates_dir)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model and Chat Manager Initialization ---
model_loader = ModelLoader()
chat_manager = ChatManager()

# --- Auth Dependency ---
def require_auth(request: Request):
    if AUTH_PASSWORD:
        auth = request.headers.get("Authorization")
        if not auth or auth != f"Bearer {AUTH_PASSWORD}":
            log_event("unauthorized", path=str(request.url), ip=request.client.host if request.client else "unknown")
            raise HTTPException(401, "Missing or invalid password.")
    return True

# --- In-memory Rate Limiting ---
RATE_LIMIT_WINDOW = 60  # seconds
rate_limit_data = defaultdict(lambda: deque())  # IP -> deque of timestamps

def check_rate_limit(ip: str) -> bool:
    if ip in BANNED_IPS:
        return False
    if RATE_LIMIT is None:
        return True
    now = time.time()
    dq = rate_limit_data[ip]
    while dq and dq[0] < now - RATE_LIMIT_WINDOW:
        dq.popleft()
    if len(dq) >= RATE_LIMIT:
        return False
    dq.append(now)
    return True

def prune_rate_limit():
    now = time.time()
    for ip, dq in list(rate_limit_data.items()):
        while dq and dq[0] < now - RATE_LIMIT_WINDOW:
            dq.popleft()
        if not dq:
            del rate_limit_data[ip]

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)

class ToolRequest(BaseModel):
    tool: str = Field(..., min_length=1)
    args: Optional[List[str]] = []

class ReloadToolRequest(BaseModel):
    tool: str = Field(..., min_length=1)

# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
def serve_ui(request: Request):
    """
    Serves the frontend HTML/JS chat UI at the root endpoint ('/').
    INPUTS:
    - request: FastAPI Request object (provides context for template rendering).
    OUTPUTS:
    - Returns an HTMLResponse with the rendered frontend.html template.
    SIDE EFFECTS:
    - None.
    EDGE CASES:
    - If frontend.html is missing, a default is created at startup.
    """
    return templates.TemplateResponse("frontend.html", {"request": request})

@app.post("/api/chat")
async def chat_endpoint(request: Request, payload: ChatRequest = Body(...), auth=Depends(require_auth)):
    """
    Handles chat requests from the frontend UI.
    INPUTS:
    - request: FastAPI Request object containing JSON body with 'message'.
    OUTPUTS:
    - Returns a StreamingResponse with the assistant's reply.
    SIDE EFFECTS:
    - Updates chat session state, logs messages, and streams response.
    EDGE CASES:
    - Handles missing/invalid model, tokenizer, or input.
    - Handles model/tokenizer compatibility issues.
    - Handles rate limiting and authentication.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        log_event("rate_limit_exceeded", ip=client_ip, path="/api/chat")
        raise HTTPException(429, "Too many requests. Try again later.")
    try:
        user_msg = payload.message
        session_id = request.cookies.get("session_id")
        session_id = chat_manager.get_or_create_session(session_id)
        chat_manager.add_message(session_id, "user", user_msg)
        model, tokenizer = model_loader.get_model_and_tokenizer()
        if model is None or tokenizer is None:
            log_event("model_unavailable", ip=client_ip)
            return error_service_unavailable("Model or tokenizer not loaded. Check model path and files.")
        messages = chat_manager.format_messages_for_qwen(session_id, user_msg)
        # --- Compatibility and fallback logic ---
        try:
            if hasattr(tokenizer, "apply_chat_template") and callable(tokenizer.apply_chat_template):  # type: ignore[attr-defined]
                text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            if callable(tokenizer):
                model_inputs = tokenizer([text], return_tensors="pt")
                # Move tensors to model.device if available
                if hasattr(model, "device"):
                    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}  # type: ignore[attr-defined]
            else:
                raise RuntimeError("Tokenizer is not callable.")
        except Exception as e:
            log_event("tokenizer_error", error=str(e), ip=client_ip)
            return json_error("Model/tokenizer compatibility error", code=500, detail=str(e))
        try:
            if hasattr(model, "generate") and callable(model.generate):  # type: ignore[attr-defined]
                generated_ids = model.generate(  # type: ignore[attr-defined]
                    **model_inputs,
                    max_new_tokens=MAX_NEW_TOKENS
                )
            else:
                raise RuntimeError("Model missing 'generate' method.")
            input_ids = model_inputs["input_ids"] if "input_ids" in model_inputs else list(model_inputs.values())[0]
            output_ids = generated_ids[0][len(input_ids[0]):].tolist()  # type: ignore[index]
            if hasattr(tokenizer, "decode") and callable(tokenizer.decode):  # type: ignore[attr-defined]
                response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()  # type: ignore[attr-defined]
            else:
                raise RuntimeError("Tokenizer missing 'decode' method.")
        except Exception as e:
            log_event("model_inference_error", error=str(e), ip=client_ip)
            return json_error("Model inference error", code=500, detail=str(e))
        chat_manager.add_message(session_id, "assistant", response)
        def stream():
            """
            Generator function to stream the assistant's response in chunks.
            Yields encoded bytes for StreamingResponse.
            """
            for chunk in chat_manager.stream_response(response):
                yield chunk.encode()
        log_event("chat_success", ip=client_ip)
        return StreamingResponse(stream(), media_type="text/plain")
    except ValidationError as e:
        log_event("chat_validation_error", error=str(e), ip=client_ip)
        return json_error("Invalid input", code=400, detail=str(e))
    except Exception as e:
        log_event("chat_error", error=str(e), ip=client_ip)
        return json_error("Chat error", code=500, detail=str(e))

@app.get("/api/filetree")
def filetree_endpoint(request: Request, auth=Depends(require_auth)):
    """
    Returns a text-based file tree of the project directory.
    INPUTS:
    - request: FastAPI Request object (for auth and logging).
    OUTPUTS:
    - Returns a dict with 'root' (absolute path) and 'tree' (string representation).
    SIDE EFFECTS:
    - Logs access and enforces authentication.
    EDGE CASES:
    - Handles missing or unreadable directories.
    """
    log_event("filetree_access", ip=request.client.host if request.client else "unknown")
    def list_tree(root):
        """
        Recursively builds a string representation of the directory tree.
        INPUTS:
        - root: Root directory path.
        OUTPUTS:
        - Returns a string with indented directory/file structure.
        SIDE EFFECTS:
        - None.
        EDGE CASES:
        - Handles empty directories and deeply nested trees.
        """
        tree = []
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            indent = '  ' * rel.count(os.sep) if rel != '.' else ''
            tree.append(f"{indent}{os.path.basename(dirpath)}/")
            for f in filenames:
                tree.append(f"{indent}  {f}")
        return '\n'.join(tree)
    return {"root": os.path.abspath(PROJECT_PATH), "tree": list_tree(PROJECT_PATH)}

@app.get("/api/file")
def file_endpoint(request: Request, path: str, auth=Depends(require_auth)):
    """
    Returns the contents of a file in the project directory.
    INPUTS:
    - request: FastAPI Request object (for auth and logging).
    - path: Relative path from the project root.
    OUTPUTS:
    - Returns a dict with 'path' and 'content'.
    SIDE EFFECTS:
    - Logs access and enforces authentication.
    EDGE CASES:
    - Handles missing files, large files, and directory traversal attempts.
    SECURITY: No authentication by default; enable authentication for production or untrusted networks.
    """
    log_event("file_access", path=path, ip=request.client.host if request.client else "unknown")
    abs_path = os.path.abspath(os.path.join(PROJECT_PATH, path))
    if not abs_path.startswith(os.path.abspath(PROJECT_PATH)):
        raise HTTPException(403, "Access denied.")
    if not os.path.isfile(abs_path):
        raise HTTPException(404, "File not found.")
    # Limit file size to 128KB
    if os.path.getsize(abs_path) > 128 * 1024:
        raise HTTPException(413, "File too large.")
    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return {"path": path, "content": content}

@app.get("/api/status")
def status_endpoint(request: Request, auth=Depends(require_auth)):
    """
    Returns server status info for debugging and UI display.
    INPUTS:
    - request: FastAPI Request object (for auth and logging).
    OUTPUTS:
    - Returns a dict with model_loaded, project_path, and git_repo_url.
    SIDE EFFECTS:
    - Logs access and enforces authentication.
    EDGE CASES:
    - None.
    """
    log_event("status_access", ip=request.client.host if request.client else "unknown")
    return {
        "model_loaded": model_loader.is_loaded(),
        "project_path": os.path.abspath(PROJECT_PATH),
        "git_repo_url": GIT_REPO_URL,
    }

@app.post("/api/gitpull")
def gitpull_endpoint(request: Request, auth=Depends(require_auth)):
    """
    Pulls the latest changes from the configured git repository.
    INPUTS:
    - request: FastAPI Request object (for auth).
    OUTPUTS:
    - Returns a dict with status and message.
    SIDE EFFECTS:
    - Updates the local git repo.
    EDGE CASES:
    - Handles missing repo, pull errors, and authentication.
    """
    if not GIT_REPO_URL:
        return json_error("No git repo configured.", code=400)
    try:
        repo = git.Repo(PROJECT_PATH)
        repo.remotes.origin.pull()
        log_event("git_pull", status="ok")
        return {"status": "ok", "message": "Repo pulled."}
    except Exception as e:
        log_event("git_pull_failed", error=str(e))
        return json_error("Git pull failed", code=500, detail=str(e))

@app.post("/api/tool")
def tool_endpoint(request: Request, payload: ToolRequest = Body(...), auth=Depends(require_auth)):
    """
    Executes an agent tool (Python script) with optional arguments.
    INPUTS:
    - request: FastAPI Request object (for auth and rate limiting).
    - data: Dict with 'tool' (tool name) and 'args' (list of arguments).
    OUTPUTS:
    - Returns a dict with 'output' (tool output).
    SIDE EFFECTS:
    - Runs a subprocess, logs execution, and enforces authentication/rate limiting.
    EDGE CASES:
    - Handles missing tool, execution errors, and rate limiting.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        log_event("rate_limit_exceeded", ip=client_ip, path="/api/tool")
        raise HTTPException(429, "Too many requests. Try again later.")
    tool_name = payload.tool
    args = payload.args or []
    if tool_name not in ALLOWED_TOOLS:
        log_event("tool_not_allowed", tool=tool_name, ip=client_ip)
        return json_error("Tool not allowed.", code=403)
    # Only allow tools in quick-llm/quick-llm-tools/
    tool_path = os.path.abspath(os.path.join(PROJECT_PATH, "quick-llm", "quick-llm-tools", f"{tool_name}.py"))
    allowed_dir = os.path.abspath(os.path.join(PROJECT_PATH, "quick-llm", "quick-llm-tools"))
    if not tool_path.startswith(allowed_dir) or not os.path.isfile(tool_path):
        log_event("tool_script_not_found", tool=tool_name, path=tool_path, ip=client_ip)
        return json_error("Tool script not found.", code=404)
    # Validate args: only allow list of str, no shell metacharacters
    for arg in args:
        if not isinstance(arg, str) or any(c in arg for c in [';', '|', '&', '`', '$', '>', '<']):
            log_event("tool_arg_invalid", tool=tool_name, arg=arg, ip=client_ip)
            return json_error("Invalid tool argument.", code=400)
    log_event("tool_execute", tool=tool_name, args=args, ip=client_ip)
    output = chat_manager.run_tool(tool_path, args=args, cwd=PROJECT_PATH)
    return {"output": output}

@app.get("/api/tools")
def list_tools_endpoint():
    """
    Lists all available agent tools by name.
    INPUTS:
    - None.
    OUTPUTS:
    - Returns a dict with 'tools' (list of tool names).
    SIDE EFFECTS:
    - None.
    EDGE CASES:
    - Handles empty or missing tools directory.
    """
    return {"tools": chat_manager.list_tools()}

@app.post("/api/reload-tools")
def reload_tools_endpoint(request: Request, payload: ReloadToolRequest = Body(...), auth=Depends(require_auth)):
    """
    Hot-reloads a Python tool module by name.
    INPUTS:
    - request: FastAPI Request object (for auth).
    - data: Dict with 'tool' (tool name).
    OUTPUTS:
    - Returns a dict with status and message.
    SIDE EFFECTS:
    - Reloads the tool module in sys.modules.
    EDGE CASES:
    - Handles missing tool, reload errors, and submodule cleanup.
    """
    tool_name = payload.tool
    try:
        module_name = f"tools.{tool_name}"
        # Remove submodules before reload
        for k in list(sys.modules):
            if k.startswith(module_name + "."):
                del sys.modules[k]
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)
        log_event("tool_reloaded", tool=tool_name)
        return {"status": "ok", "message": f"Reloaded {tool_name}"}
    except Exception as e:
        log_event("tool_reload_failed", tool=tool_name, error=str(e))
        return json_error(f"Failed to reload tool: {tool_name}", code=500, detail=str(e))

# --- LAN IP Detection on Startup ---
@app.on_event("startup")
def show_lan_ip():
    """
    Prints and logs the LAN IP and port to the console on server startup.
    INPUTS:
    - None.
    OUTPUTS:
    - None (side effect: print and log).
    SIDE EFFECTS:
    - Prints and logs the server URL for LAN access.
    EDGE CASES:
    - Handles network errors and falls back to localhost.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to a public IP to determine the local LAN IP (does not send data)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "localhost"
    finally:
        s.close()
    logger.info(f"Jadio-quick-llm server running at: http://{ip}:{PORT}/")
    print(f"\nJadio-quick-llm server running at: http://{ip}:{PORT}/\n")

# --- Error Handling ---
@app.exception_handler(HTTPException)
def http_exception_handler(request, exc):
    client_ip = request.client.host if hasattr(request, 'client') and request.client else 'unknown'
    log_event("http_exception", error=exc.detail, path=getattr(request, 'url', 'unknown'), ip=client_ip)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
def handle_exception(request, exc):
    client_ip = request.client.host if hasattr(request, 'client') and request.client else 'unknown'
    log_event("unhandled_exception", error=str(exc), path=getattr(request, 'url', 'unknown'), ip=client_ip)
    return json_error("Internal server error", code=500, detail=str(exc))

# --- Main Entrypoint ---
if __name__ == "__main__":
    """
    Runs the FastAPI app with uvicorn when executed as a script.
    INPUTS:
    - None (uses config values).
    OUTPUTS:
    - None (side effect: starts server).
    SIDE EFFECTS:
    - Binds to 0.0.0.0 for LAN access.
    - Enables auto-reload if DEBUG is True.
    EDGE CASES:
    - Handles invalid config or port in use.
    """
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=DEBUG)

"""
Note: Dynamic tool reloading uses importlib.reload, but stateful tools may not be fully reloaded. Stateless tool design is recommended. See README.md for limitations. TODO: Consider a plugin interface for agent tools in the future.
"""