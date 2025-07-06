# Jadio-quick-llm

A LAN-only, private AI chat server and agent tool platform powered by Qwen3-0.6B. Features a modern web UI, agent tool integration, and local-first privacy. Designed for trusted home/office networks.

---

## Features

- **Local LLM Chat**: Run Qwen3-0.6B on your own PC, no cloud required.
- **Modern Web UI**: Responsive chat, file browser, and file viewer.
- **Agent Tools**: Run trusted scripts (e.g., code/document generators) from chat.
- **No Data Leaves Your Network**: All inference and data stay local.
- **Configurable**: YAML-based config for model path, port, authentication, and more.
- **Extensible**: Add your own agent tools in the `tools/` directory.
- **Testing**: Unit and integration tests for all endpoints and features.

---

## Safety Warning

**Do NOT expose this server to the public internet!**  
By default, there is no authentication and all LAN users can access the server. Only run on a trusted home or office network.

---

## Quickstart

### 1. Install Python

- Python 3.9+ (recommended: 3.10+)

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Configure

Edit `quick-llm/config.yaml`:

- `model_path`: Path to your Qwen3-0.6B model files.
- `project_path`: Path to your project root (for file browsing).
- `port`: TCP port for the server (default: 8000).
- `auth_password`: (Optional) Set a password for API access.
- `rate_limit`: (Optional) Requests per minute per IP.
- `max_new_tokens`: Max tokens per model response.

### 4. Start the Server

```sh
python -m core.main
```

### 5. Access the Web UI

Open your browser to:  
`http://localhost:8000/`

---

## Project Structure

- `core/`  
  - `main.py`: FastAPI server, API endpoints, serves UI.
  - `chat_manager.py`: Manages chat sessions, agent tool execution.
  - `model_loader.py`: Loads/unloads Qwen3-0.6B model and tokenizer.
  - `config.py`: Loads and validates YAML config.
  - `error_utils.py`: Error handling utilities.
  - `session_store.py`: (If present) Session persistence.
  - `Qwen3-0.6B/`: Model files and config.
  - `testing/`: Unit and integration tests.
- `quick-llm/`
  - `config.yaml`: Main configuration file.
  - `quickllm.html`: Standalone web frontend.
  - `docs/`: User guide, troubleshooting, flow guide.
  - `quick-llm-tools/`: Example agent tools (e.g., `createtestdoc.py`, `tex2pdf.py`).
- `requirements.txt`: Python dependencies.
- `LICENSE`: MIT License.

---

## Agent Tools

- Place trusted scripts in `quick-llm/quick-llm-tools/`.
- Use `/api/tools` to list available tools.
- Use `/api/reload-tools` to reload after adding new scripts.
- Example: The `createtestdoc` tool creates a test README file.

---

## Testing

- **Unit tests**: `core/testing/test_unit.py`
- **Integration tests**: `core/testing/test_integration.py`, `core/testing/test.py`
- Run with `pytest` or directly as scripts.

---

## Troubleshooting

See `quick-llm/docs/TROUBLESHOOTING.md` for common issues, debugging tips, and help.

---

## License

MIT License (see `LICENSE` for details).

---

## Credits

- Qwen3-0.6B by Alibaba/QwenLM ([GitHub](https://github.com/QwenLM/Qwen3))
- Project by Jadio, 2025
