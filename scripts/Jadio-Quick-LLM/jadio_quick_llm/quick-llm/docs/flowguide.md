# Jadio-quick-llm Technical Flow Guide & Dependency Map

## 1. High-Level Architecture

```
[frontend.html] <--> [main.py (FastAPI)] <--> [chat_manager.py] <--> [model_loader.py]
                                         |                    |
                                         |                    +--> [tools/*]
                                         |
                                         +--> [config.py]
                                         +--> [Project Files / Git Repo]
                                         +--> [test.py] (for testing)
```

- **frontend.html**: Browser UI for chat, file tree, file viewer. Talks to FastAPI endpoints.
- **main.py**: FastAPI server. Serves API, static UI, manages sessions, routes requests to chat_manager, model_loader, tools, and project files.
- **chat_manager.py**: Handles chat sessions, message formatting, chat history, and agent tool execution. Calls model_loader for inference.
- **model_loader.py**: Loads/unloads models on demand. Provides inference API to chat_manager.
- **config.py**: Central config for model path, project path, repo URL, authentication, and rate limiting.
- **tools/**: Folder for agent tool scripts. Executed by chat_manager via /api/tool.
- **test.py**: Integration tests for all endpoints and features.

## 2. File-by-File Breakdown

### main.py
- **Purpose**: Entry point. FastAPI app. Serves API endpoints and web UI.
- **Key Functions**:
  - `/api/chat`: Handles chat requests, calls chat_manager.
  - `/api/filetree`, `/api/file`: Serve project file tree/content.
  - `/api/tool`: Runs agent tools via chat_manager.
  - `/api/status`, `/api/gitpull`: Health and repo sync.
  - Serves `frontend.html` at root.
- **Dependencies**: chat_manager.py, model_loader.py, config.py, tools/, project files, FastAPI, Uvicorn.

### chat_manager.py
- **Purpose**: Manages chat sessions, history, message formatting, and agent tool execution.
- **Key Functions**:
  - `handle_chat(session_id, message)`: Processes chat, manages history.
  - `run_tool(tool_name, args)`: Executes agent tool scripts from tools/.
  - Calls model_loader for inference.
- **Dependencies**: model_loader.py, tools/, config.py.

### model_loader.py
- **Purpose**: Loads/unloads language models and tokenizers. Handles inference.
- **Key Functions**:
  - `load_model(path)`, `unload_model()`, `infer(input)`.
- **Dependencies**: config.py, transformers, torch.

### config.py
- **Purpose**: Central configuration (model path, project path, repo URL, authentication, rate limiting).
- **Key Functions**:
  - Loads/validates settings.
- **Dependencies**: None (standalone, imported by all other modules).

### frontend.html
- **Purpose**: Standalone web UI for chat, file tree, and file viewer.
- **Key Functions**:
  - Calls API endpoints via fetch/AJAX.
- **Dependencies**: main.py (served by backend).

### tools/ (folder)
- **Purpose**: Agent tool scripts (e.g., `createtestdoc.py`).
- **Key Functions**:
  - Each script exposes a callable interface (usually a main function or CLI entry).
- **Dependencies**: Called by chat_manager.py via subprocess or import.

### test.py
- **Purpose**: Integration tests for all endpoints and features.
- **Key Functions**:
  - Tests chat, file, tool, and status endpoints.
- **Dependencies**: main.py, requests, pytest, etc.

### Project Files / Git Repo
- **Purpose**: The codebase or repo being introspected.
- **Dependencies**: Accessed by main.py and chat_manager.py for file tree/content.

## 3. Dependency Map (Text)

- `main.py` imports/uses: `chat_manager.py`, `model_loader.py`, `config.py`, `tools/`, `frontend.html`, project files, FastAPI, Uvicorn
- `chat_manager.py` imports/uses: `model_loader.py`, `tools/`, `config.py`
- `model_loader.py` imports/uses: `config.py`, `transformers`, `torch`
- `config.py`: Standalone, imported by all
- `frontend.html`: Calls API endpoints in `main.py`
- `tools/`: Called by `chat_manager.py`
- `test.py`: Calls `main.py` endpoints

## 4. Main Data/Control Flow

1. **User opens frontend.html in browser**
2. **User sends chat or file request**
3. **main.py receives API call**
   - For chat: calls `chat_manager.handle_chat()`
     - May call `model_loader.infer()` for LLM response
     - May call `run_tool()` for agent tool execution
   - For file: reads from project folder/repo
   - For tool: calls tool script via chat_manager
4. **Response sent back to frontend**

## 5. Example: Chat Request Flow

1. User types message in frontend.html
2. JS sends POST to `/api/chat` (main.py)
3. main.py calls `chat_manager.handle_chat()`
4. chat_manager updates session, formats message
5. chat_manager calls `model_loader.infer()` for LLM output
6. chat_manager may call `run_tool()` if tool requested
7. main.py returns response to frontend

## 6. Example: Agent Tool Flow

1. User sends `/tool createtestdoc` in chat or via UI
2. main.py routes to chat_manager.run_tool('createtestdoc')
3. chat_manager executes `tools/createtestdoc.py`
4. Tool runs, modifies project files or returns output
5. Output sent back to user via chat

---

This guide provides a full map of how all files interact and the main data/control flows in Jadio-quick-llm. For further details, see inline code comments and the README.
