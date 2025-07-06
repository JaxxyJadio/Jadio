"""
test_integration.py with Heavy Inline Documentation

This script contains integration tests for the Jadio-quick-llm API endpoints.
It is designed to be run directly (python test_integration.py) and is intended for validating the end-to-end functionality of the running FastAPI server.

Design assumptions:
- The FastAPI server must be running and accessible on the configured port before running these tests.
- Intended for LAN-only, personal use. Security is not a concern for these tests, but do not run against a public server.
- All endpoints are assumed to be open or protected by password as configured in config.yaml.

Interactions:
- main.py: Provides all API endpoints tested here (status, filetree, chat, tools).
- config.py: Supplies the PORT value for the server base URL and config validation.
- tools/: Agent tools are invoked via the /api/tool endpoint.

Security notes:
- No sensitive data is used or required.
- Tests are safe for trusted LAN environments only.

Future-you: If you add new endpoints or features, add corresponding tests here. Keep tests isolated and fast. Clean up any artifacts created by tests.
"""

# --- Imports ---
import requests  # Used for making HTTP requests to the FastAPI server
import os        # Used for environment variable and path manipulations
import time      # Used for waiting/retrying until the server is ready
import logging   # Used for logging test output
from jadio_quick_llm.core.config import PORT, validate_config  # Import the server port and config validation

# --- Helper: Construct the BASE URL for API requests ---
def get_base_url():
    """
    Construct the BASE URL for API requests.
    Uses 127.0.0.1 by default to avoid IPv6/localhost surprises.
    Can be overridden with the JADIO_SERVER environment variable (e.g., http://192.168.1.10:8000).
    Returns the base URL as a string.
    """
    override = os.environ.get("JADIO_SERVER")
    if override:
        return override.rstrip("/")
    return f"http://127.0.0.1:{PORT}"

# --- Global: BASE URL for all requests ---
BASE = get_base_url()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jadio-integration")

# --- Helper: Wait for the server to be ready before running tests ---
def wait_for_server_ready(timeout=10, delay=1.0):
    """
    Wait for the server to be ready by polling /api/status.
    Retries up to 'timeout' times with 'delay' seconds between attempts.
    Raises RuntimeError if the server does not become ready in time.
    """
    for _ in range(timeout):
        try:
            r = requests.get(f"{BASE}/api/status")
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(delay)
    raise RuntimeError("Server did not become ready in time.")

# --- Test: /api/status endpoint ---
def test_status():
    """
    Test the /api/status endpoint to ensure the server is running.
    Raises AssertionError if the status code is not 200.
    """
    r = requests.get(f"{BASE}/api/status")
    assert r.status_code == 200

# --- Test: /api/filetree endpoint ---
def test_filetree():
    """
    Test the /api/filetree endpoint to ensure the file tree is accessible.
    Raises AssertionError if the status code is not 200 or 'tree' is missing.
    """
    r = requests.get(f"{BASE}/api/filetree")
    assert r.status_code == 200 and "tree" in r.json()

# --- Test: /api/chat endpoint ---
def test_chat():
    """
    Test the /api/chat endpoint by sending a simple message.
    Raises AssertionError if the status code is not 200.
    """
    r = requests.post(f"{BASE}/api/chat", json={"message": "Hello!"})
    assert r.status_code == 200

# --- Test: /api/tools and /api/tool endpoints ---
def test_tools():
    """
    Test the /api/tools endpoint to list available tools, and /api/tool to run one.
    Raises AssertionError if the endpoints do not return expected results.
    """
    r = requests.get(f"{BASE}/api/tools")
    assert r.status_code == 200 and "tools" in r.json()
    if r.json()["tools"]:
        tool = r.json()["tools"][0]
        r2 = requests.post(f"{BASE}/api/tool", json={"tool": tool})
        assert r2.status_code == 200

# --- Main Entrypoint: Run all integration tests in order ---
def run_all():
    """
    Run all integration tests in order, with readiness check.
    Logs a summary at the end.
    Raises AssertionError if any test fails.
    """
    wait_for_server_ready()
    test_status()
    test_filetree()
    test_chat()
    test_tools()
    logger.info("All integration tests passed.")

if __name__ == "__main__":
    run_all()
