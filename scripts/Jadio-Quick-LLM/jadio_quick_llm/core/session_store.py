"""
session_store.py with Heavy Inline Documentation

This module provides persistent session storage for the Jadio-quick-llm project.
It is responsible for saving and loading chat sessions to/from disk, allowing chat history to survive server restarts.

Design assumptions:
- This project is for LAN-only, personal use. Security is not a concern for this persistence layer.
- All session data is stored in a single JSON file (sessions.json) in the project directory.
- The file is small and only accessed by the server process, so no locking or concurrency control is implemented.

Interactions:
- Used by chat_manager.py to persist and restore chat sessions.
- Not used directly by main.py, model_loader.py, or tools/.

Security notes:
- The session file is not encrypted or protected. Do not use on untrusted systems.
- If the file is corrupted, all session data may be lost. Consider adding backup/restore logic if needed.

Future-you: If you add multi-user support or scale to multiple servers, replace this with a database or distributed store.
"""

import json  # Used for serializing/deserializing session data to/from JSON
import os    # Used for file path manipulations and existence checks
from typing import Dict, List, Any  # Used for type annotations of session data

# --- Path to the session file ---
SESSION_FILE = os.path.join(os.path.dirname(__file__), "sessions.json")
# This file will be created in the same directory as this script.
# It stores all chat sessions as a dict: {session_id: [messages]}

# --- Save all sessions to disk ---
def save_sessions(sessions: Dict[str, List[dict]]) -> None:
    """
    Save the current sessions to disk as JSON.
    Overwrites the entire sessions.json file.
    Args:
        sessions: Dict mapping session IDs to lists of message dicts.
    Returns:
        None
    """
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

# --- Load all sessions from disk ---
def load_sessions() -> Dict[str, List[dict]]:
    """
    Load all sessions from disk (sessions.json).
    Returns an empty dict if the file does not exist.
    Returns:
        Dict mapping session IDs to lists of message dicts.
    """
    if not os.path.isfile(SESSION_FILE):
        return {}
    with open(SESSION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
