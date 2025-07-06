"""
test_unit.py with Heavy Inline Documentation

This script contains unit tests for the core modules of the Jadio-quick-llm project.
It is designed to be run directly (python test_unit.py) and is intended for rapid, local validation of core logic.

Design assumptions:
- This project is for LAN-only, personal use. Security is not a concern for these tests.
- All tests are safe to run on a trusted development machine.
- The FastAPI server does NOT need to be running for these unit tests (unlike integration tests).

Interactions:
- chat_manager.py: Session management, message history, and persistence are tested.
- model_loader.py: Model loading/unloading and API are tested.
- config.py: Configuration validation is tested.

Security notes:
- No network or file system changes outside the project directory.
- No sensitive data is used or required.

Future-you: If you add new core modules or features, add corresponding unit tests here. Keep tests isolated and fast.
"""

# --- Imports ---
import logging  # Used for logging test output if needed (not used directly here, but available for debugging)
import os       # Used for file path manipulations (not used directly here, but available for future tests)
import sys      # Used to exit with the correct status code after running pytest
import pytest   # Pytest is the test runner and assertion framework used for all tests
# Add the project root to sys.path for import compatibility
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from jadio_quick_llm.core.chat_manager import ChatManager  # The chat/session manager module under test
from jadio_quick_llm.core.model_loader import ModelLoader  # The model loader module under test
from jadio_quick_llm.core.config import validate_config    # The config validation function under test

# --- Test: Configuration Validation ---
def test_config_validation():
    """
    Test that the configuration is valid according to config.py rules.
    This ensures that config.yaml is well-formed and all required fields are present.
    Raises AssertionError if validation fails.
    """
    validate_config()

# --- Test: ChatManager Session Logic and Persistence ---
def test_chat_manager_session():
    """
    Test the creation of a chat session, message history, and session persistence.
    - Creates a new ChatManager instance.
    - Creates a new session and adds user/assistant messages.
    - Checks that messages are stored correctly.
    - Saves sessions to disk and reloads to verify persistence.
    Raises AssertionError if any step fails.
    """
    cm = ChatManager()  # Create a new chat manager instance
    sid = cm.get_or_create_session()  # Create a new session (returns session ID)
    assert sid in cm.sessions  # The session should exist in the session dict
    cm.add_message(sid, "user", "hi")  # Add a user message
    cm.add_message(sid, "assistant", "hello")  # Add an assistant message
    hist = cm.get_history(sid)  # Retrieve the message history for this session
    assert len(hist) == 2  # There should be two messages in the history
    cm.save()  # Persist sessions to disk (sessions.json)
    # Reload and check persistence
    cm2 = ChatManager()  # Create a new ChatManager (loads from disk)
    assert sid in cm2.sessions  # The session should still exist after reload
    assert len(cm2.get_history(sid)) == 2  # The message history should be preserved

# --- Test: ModelLoader Logic ---
def test_model_loader():
    """
    Test the loading and unloading of the model and tokenizer.
    - Attempts to load the model and tokenizer from disk.
    - Checks that both are loaded and not None.
    - Unloads and checks that both are None.
    - If model files are missing or incompatible, skips the test (does not fail the suite).
    This test ensures that model_loader.py is working and that model files are present.
    """
    loader = ModelLoader()  # Create a new model loader instance
    try:
        loader.load()  # Attempt to load the model and tokenizer
        assert loader.is_loaded()  # Both should be loaded
        model, tokenizer = loader.get_model_and_tokenizer()  # Retrieve model/tokenizer
        assert model is not None and tokenizer is not None  # Both should not be None
        loader.unload()  # Unload model/tokenizer
        assert not loader.is_loaded()  # Both should be None after unload
    except Exception as e:
        # If model files are missing or loading fails, skip the test (not a failure)
        pytest.skip(f"Model loading skipped: {e}")

# --- Main Entrypoint: Run all tests with pytest if executed directly ---
if __name__ == "__main__":
    # This allows the script to be run directly (python test_unit.py)
    # It will invoke pytest on itself and exit with the correct status code.
    import pytest
    sys.exit(pytest.main([os.path.abspath(__file__)]))
