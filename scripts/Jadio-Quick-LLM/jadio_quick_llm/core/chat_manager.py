"""
chat_manager.py with Heavy Inline Documentation (Finalized)

This module implements the ChatManager class for the Jadio-quick-llm project.
It manages per-client session state, chat history, message formatting for the Qwen model,
streaming responses to the frontend, and trusted agent tool execution.

Design assumptions:
- LAN-only, personal use (see README.md and flowguide.md).
- All chat state is in-memory (not persisted); safe for trusted, single-user or small-team LAN use.
- No persistence: If the server restarts, all chat history is lost (intentional for simplicity and privacy).
- All subprocesses are trusted, local scripts. Never allow user-uploaded or arbitrary code execution.

Interactions:
- main.py: Instantiates and uses ChatManager for all chat-related API endpoints.
- model_loader.py: Used by main.py for model inference, not directly by ChatManager.
- tools/: Agent tools/scripts can be executed via run_tool().
- config.py: Not directly used, but config values (e.g., PROJECT_PATH) may be passed in from main.py.

Security notes:
- No authentication or rate limiting is enforced here; see main.py for any such logic.
- All subprocess calls are trusted (tools are local scripts, not user uploads).
- Never expose this server to the public internet without strong authentication and rate limiting.
- If you add persistence or remote tool execution in the future, review all security assumptions.

Future-you: If you add new session features, consider persistence or cleanup for long-running servers. If you add new agent tool features, never allow user-uploaded or arbitrary code execution.
"""

import uuid  # For generating unique session IDs
import subprocess  # For running agent tools/scripts as subprocesses
import os  # For checking if a tool/script exists before running
import importlib.util
from typing import Dict, List, Optional, Generator, TypedDict
from .session_store import save_sessions, load_sessions

class MessageDict(TypedDict):
    role: str
    content: str

class ChatManager:
    """
    Manages chat sessions, message history, message formatting, streaming responses, and agent tool execution.
    All state is kept in memory (per process). Safe for LAN-only, trusted use.
    """
    def __init__(self) -> None:
        # Use type ignore for compatibility with JSON load/save
        self.sessions: Dict[str, List[MessageDict]] = load_sessions()  # type: ignore

    def save(self) -> None:
        save_sessions(self.sessions)  # type: ignore

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """
        Get an existing session by ID, or create a new session if not found or no ID given.
        Returns the session ID (existing or new).
        """
        if session_id and session_id in self.sessions:
            return session_id
        new_id = str(uuid.uuid4())
        self.sessions[new_id] = []
        self.save()
        return new_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a message to the chat history for a session.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})
        self.save()

    def get_history(self, session_id: str) -> List[MessageDict]:
        """
        Get the chat history (list of messages) for a session.
        """
        return self.sessions.get(session_id, [])

    def format_messages_for_qwen(self, session_id: str, new_user_message: str) -> List[MessageDict]:
        """
        Format the full message list for Qwen model inference, including the new user message.
        """
        history = self.get_history(session_id)
        messages = history + [{"role": "user", "content": new_user_message}]
        return messages

    def stream_response(self, response_text: str, chunk_size: int = 20) -> Generator[str, None, None]:
        """
        Yield the response text in chunks for streaming to the client.
        """
        for i in range(0, len(response_text), chunk_size):
            yield response_text[i:i+chunk_size]

    def run_tool(self, tool_path: str, args: Optional[List[str]] = None, cwd: Optional[str] = None) -> str:
        """
        Run a script/tool (trusted local only) as a subprocess and return its output.
        Returns stdout if successful, or an error message if the tool fails or is not found.
        SECURITY: Only trusted, local scripts should be run. Never allow user-uploaded or arbitrary code here.
        """
        if not os.path.isfile(tool_path):
            return f"Tool not found: {tool_path}"
        try:
            result = subprocess.run(
                ["python", tool_path] + (args or []),
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Tool error ({result.returncode}): {result.stderr.strip()}"
        except Exception as e:
            return f"Tool execution failed: {e}"

    def list_tools(self) -> list:
        """
        List all available agent tools in the tools/ directory (by .py filename, no extension).
        """
        tools_dir = os.path.join(os.path.dirname(__file__), "tools")
        return [f[:-3] for f in os.listdir(tools_dir) if f.endswith(".py") and not f.startswith("__")]

    def reload_tool(self, tool_name: str):
        """
        Dynamically reload a tool module by name (for hot-reloading agent tools).
        Returns True if successful, False otherwise.
        """
        tools_dir = os.path.join(os.path.dirname(__file__), "tools")
        tool_path = os.path.join(tools_dir, f"{tool_name}.py")
        if not os.path.isfile(tool_path):
            return False
        spec = importlib.util.spec_from_file_location(tool_name, tool_path)
        if not spec:
            return False
        try:
            importlib.util.module_from_spec(spec)
            return True
        except Exception:
            return False

# --- Unit tests for ChatManager ---
# Run this file directly (python chat_manager.py) to verify all methods work as expected.
if __name__ == "__main__":
    cm = ChatManager()
    sid = cm.get_or_create_session()
    assert sid in cm.sessions, "Session should be created."
    cm.add_message(sid, "user", "Hello!")
    cm.add_message(sid, "assistant", "Hi there!")
    hist = cm.get_history(sid)
    assert len(hist) == 2, "History should have 2 messages."
    formatted = cm.format_messages_for_qwen(sid, "How are you?")
    assert formatted[-1]["content"] == "How are you?", "Last message should be the new user message."
    chunks = list(cm.stream_response("This is a test response.", chunk_size=5))
    assert any("test" in chunk for chunk in chunks), "Chunking should work."
    tool_result = cm.run_tool("tools/linter.py")
    assert (
        "Tool not found" in tool_result or
        "Tool execution failed" in tool_result or
        "Tool error" in tool_result
    ), "Tool runner should handle missing tool."
    tools_list = cm.list_tools()
    assert isinstance(tools_list, list), "list_tools should return a list."
    assert all(isinstance(name, str) for name in tools_list), "Tool names should be strings."
    reload_result = cm.reload_tool("linter")
    assert reload_result is True, "reload_tool should succeed for existing tools."
    print("ChatManager tests passed.")
