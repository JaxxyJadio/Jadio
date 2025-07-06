"""
test.py with Heavy Inline Documentation (Quality Enforced)

This script contains unit and integration tests for the Jadio-quick-llm project.
It covers model loading, chat, API endpoints, file/repo access, and agent tools.

Design assumptions:
- This project is for LAN-only, personal use (see README.md and flowguide.md).
- The FastAPI server must be running locally on the configured port before running these tests.
- All endpoints are open to trusted LAN users; no authentication is required unless set in config.yaml.
- These tests are intended for development and maintenance, not for public CI/CD or untrusted networks.
- File path logic assumes this script is in the project root or a subdirectory (see test_file and test_tool).
- Logging is used for all output; logging level is configurable via LOGLEVEL env var.
- BASE URL can be overridden with the JADIO_SERVER environment variable for remote/alternate host testing.

Interactions:
- main.py: Provides all API endpoints tested here.
- config.py: Supplies the PORT value for the server base URL and validate_config().
- tools/: Agent tools (e.g., createtestdoc) are invoked via the /api/tool endpoint.

Security notes:
- These tests assume a trusted, local environment. Never run them against a public server.
- If authentication or rate limiting is enabled, update tests accordingly.

Future-you: If you add new endpoints or features, add corresponding tests here. Keep tests fast, isolated, and easy to debug. Clean up any artifacts created by tests.
"""

import requests  # For making HTTP requests to the FastAPI server
import os  # For file path manipulations and existence checks
import time  # For waiting/retrying until the server is ready
import logging  # For configurable, robust output
from jadio_quick_llm.core.config import PORT, validate_config  # Import the server port and config validation
from typing import Optional

# --- Logging setup ---
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("jadio-test")

# --- Validate configuration before running tests ---
try:
    validate_config()
    logger.info("Config validation passed.")
except Exception as e:
    logger.error(f"Config validation failed: {e}")
    raise SystemExit(1)

# --- BASE URL construction ---
def get_base_url() -> str:
    """
    Construct the BASE URL for API requests.
    Uses 127.0.0.1 by default to avoid IPv6/localhost surprises.
    Can be overridden with the JADIO_SERVER environment variable (e.g., http://192.168.1.10:8000).
    """
    override = os.environ.get("JADIO_SERVER")
    if override:
        logger.info(f"Using JADIO_SERVER override: {override}")
        return override.rstrip("/")
    return f"http://127.0.0.1:{PORT}"

BASE: str = get_base_url()

# --- Test functions ---
def test_status() -> None:
    """
    Test the /api/status endpoint to ensure the server is running.
    Raises AssertionError if the status code is not 200.
    """
    r = requests.get(f"{BASE}/api/status")
    assert r.status_code == 200, f"/api/status failed: {r.status_code} {r.text}"
    logger.info("Status endpoint OK.")

def test_filetree() -> None:
    """
    Test the /api/filetree endpoint to ensure the file tree is accessible.
    Raises AssertionError if the status code is not 200 or 'tree' is missing.
    """
    r = requests.get(f"{BASE}/api/filetree")
    assert r.status_code == 200 and "tree" in r.json(), f"/api/filetree failed: {r.status_code} {r.text}"
    logger.info("Filetree endpoint OK.")

def test_file() -> None:
    """
    Test the /api/file endpoint by fetching this test file itself.
    Raises AssertionError if the status code is not 200 or 'content' is missing.
    Assumes this script is in the project root or a subdirectory.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.relpath(__file__, project_root)
    r = requests.get(f"{BASE}/api/file", params={"path": path})
    assert r.status_code == 200 and "content" in r.json(), f"/api/file failed: {r.status_code} {r.text}"
    logger.info("File endpoint OK.")

def test_chat() -> None:
    """
    Test the /api/chat endpoint by sending a simple message.
    Raises AssertionError if the status code is not 200.
    """
    r = requests.post(f"{BASE}/api/chat", json={"message": "Hello!"})
    assert r.status_code == 200, f"/api/chat failed: {r.status_code} {r.text}"
    logger.info("Chat endpoint OK (streaming).")

def test_tool(cleanup: bool = True) -> None:
    """
    Test the /api/tool endpoint by invoking the 'createtestdoc' agent tool.
    Raises AssertionError if the status code is not 200, 'output' is missing, or README_TEST.md is not created.
    Optionally cleans up README_TEST.md after the test.
    Assumes the createtestdoc tool exists and is registered in the server.
    """
    r = requests.post(f"{BASE}/api/tool", json={"tool": "createtestdoc"})
    assert r.status_code == 200 and "output" in r.json(), f"/api/tool failed: {r.status_code} {r.text}"
    logger.info("Tool endpoint OK.")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme_test_path = os.path.join(project_root, "README_TEST.md")
    assert os.path.exists(readme_test_path), "README_TEST.md was not created by the tool."
    logger.info("Tool created README_TEST.md.")
    if cleanup:
        try:
            os.remove(readme_test_path)
            logger.info("Cleaned up README_TEST.md artifact.")
        except Exception as e:
            logger.warning(f"Failed to clean up README_TEST.md: {e}")

def wait_for_server_ready(timeout: int = 10, delay: float = 1.0) -> None:
    """
    Wait for the server to be ready by polling /api/status.
    Retries up to 'timeout' times with 'delay' seconds between attempts.
    Raises RuntimeError if the server does not become ready in time.
    """
    logger.info("Waiting for server to be ready...")
    for attempt in range(timeout):
        try:
            test_status()
            logger.info(f"Server ready after {attempt+1} attempt(s).")
            return
        except Exception as e:
            logger.debug(f"Server not ready yet: {e}")
            time.sleep(delay)
    raise RuntimeError("Server did not become ready in time.")

def run_all() -> None:
    """
    Run all integration tests in order, with readiness check and artifact cleanup.
    Logs a summary at the end.
    """
    try:
        wait_for_server_ready()
        test_filetree()
        test_file()
        test_chat()
        test_tool(cleanup=True)
        logger.info("All integration tests passed.")
    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    run_all()
