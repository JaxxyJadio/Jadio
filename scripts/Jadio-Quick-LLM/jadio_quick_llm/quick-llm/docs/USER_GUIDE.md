# Jadio-quick-llm User Guide

Welcome to Jadio-quick-llm! This guide will help you set up and use your own local AI chat server on your LAN.

## What is Jadio-quick-llm?
- **LAN-only AI chat server**: Run a private, local LLM (Qwen3-0.6B) on your own PC.
- **Modern web UI**: Chat with the model, browse project files, and view file contents in your browser.
- **Agent tools**: Run trusted scripts (like code generators or file creators) from the chat interface.
- **No cloud, no data leaves your network**: All inference and data stay on your machine.

---

## Safety Warning: LAN-Only Use
- **Do NOT expose this server to the public internet!**
- By default, there is **no authentication** and all LAN users can access the server.
- Only run on a trusted home or office network.

---

## Quick Setup
1. **Install Python 3.9+** (recommended: 3.10+).
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Edit your configuration:**
   - Open `config.yaml` in a text editor.
   - Adjust the following fields as needed:
     - `model_path`: Folder with your Qwen3-0.6B model files.
     - `project_path`: Path to your project root (for file browsing).
     - `git_repo_url`: (Optional) URL of a git repo for agent tools.
     - `port`: TCP port for the server (default: 8000).
     - `auth_password`: (Optional) Set a password for API access (leave blank for no auth).
     - `rate_limit`: (Optional) Requests per minute per IP (leave blank for no limit).
     - `max_new_tokens`: Max tokens per model response (prevents runaway outputs).
     - `debug`: Set to `true` for development (auto-reload), `false` for production.

---

## Starting the Server
- Run the server with:
  ```sh
  python main.py
  ```
- The server will print your LAN IP and port (e.g., `http://127.0.0.1:8000` or `http://192.168.1.42:8000`).

---

## Accessing the Web UI
- Open your browser and go to the address shown in the server output.
- On the same PC: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- From another device on your LAN: Use your PC's LAN IP and port (e.g., `http://192.168.1.42:8000`).

---

## Using Agent Tools
- Some chat commands can trigger agent tools (trusted scripts in the `tools/` folder).
- Example: The `createtestdoc` tool creates a test README file.
- Only scripts you trust and place in `tools/` can be run.

---

## Known Limitations
- **No authentication by default**: Anyone on your LAN can access the server unless you set `auth_password`.
- **No persistence**: Chat history and sessions are lost when the server restarts.
- **LAN-only**: Not designed for public or cloud use.

---

## Need Help?
- See the `README.md` and inline code documentation for more details.
- For troubleshooting, check the server logs or ask your local admin.
