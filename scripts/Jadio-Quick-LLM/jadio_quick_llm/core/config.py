"""
config.py with Heavy Inline Documentation (YAML-based, Hardened)

This module loads and validates all configuration values for the Jadio-quick-llm project from config.yaml.
It is the single source of truth for model location, server port, authentication, rate limiting, logging, and other key settings.
All other modules (main.py, model_loader.py, etc.) import from here to ensure consistency and maintainability.

Design assumptions:
- This project is intended for LAN-only, personal use (see README.md and flowguide.md).
- Security is minimal by default, but optional password and rate limiting are provided.
- All config values are meant to be easily editable for trusted users via config.yaml.

Interactions:
- main.py: Reads all config values for server setup, CORS, logging, and API security.
- model_loader.py: Reads MODEL_PATH and TORCH_DTYPE to locate and load the model files.
- chat_manager.py and tools/: May use PROJECT_PATH or other config values as needed.

Security notes:
- If AUTH_PASSWORD is None, all API endpoints are open to the LAN (trusted network only!).
- If RATE_LIMIT is None, there is no protection against accidental DoS from LAN clients.
- Never expose this server to the public internet without strong authentication and rate limiting.

Future-you: If you add new config values, document them here, update config.yaml and config_schema.yaml, and update validate_config().
"""

import os  # For checking config.yaml existence
import sys  # For clean exit in CLI test
import yaml  # For loading YAML config files
from typing import Optional, List
import logging  # For logging setup

# --- YAML schema validation ---
try:
    import yaml
    from yaml import safe_load
    try:
        import cerberus  # For schema validation (install with pip if missing)
    except ImportError:
        cerberus = None  # Will error if schema validation is attempted
except ImportError as e:
    raise ImportError("PyYAML is required for config loading. Please install it with 'pip install pyyaml'.")

CONFIG_YAML_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
CONFIG_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "config_schema.yaml")

# Try to find config.yaml in the expected location, else look in ../quick-llm/ and quick-llm/quick-llm/
if not os.path.isfile(CONFIG_YAML_PATH):
    alt_path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "quick-llm", "config.yaml"))
    alt_path2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "quick-llm", "config.yaml"))
    alt_path3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "quick-llm", "quick-llm", "config.yaml"))
    for path in [alt_path1, alt_path2, alt_path3]:
        if os.path.isfile(path):
            CONFIG_YAML_PATH = path
            break
    else:
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_YAML_PATH}, {alt_path1}, {alt_path2}, or {alt_path3}. Please create it (see project README).")

with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
    _raw_config = yaml.safe_load(f)

# --- YAML schema validation (optional, if cerberus is available) ---
def _validate_yaml_schema(config: dict) -> None:
    if not os.path.isfile(CONFIG_SCHEMA_PATH):
        return  # No schema file, skip
    if cerberus is None:
        print("WARNING: cerberus not installed, skipping YAML schema validation.")
        return
    import yaml
    with open(CONFIG_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_yaml = yaml.safe_load(f)
    # Convert schema_yaml to cerberus schema
    schema = {}
    for k, v in schema_yaml.get('mapping', {}).items():
        field = {'type': v['type']}
        if v.get('required'): field['required'] = True
        if v.get('enum'): field['allowed'] = v['enum']
        if v.get('range'):
            if 'min' in v['range']: field['min'] = v['range']['min']
            if 'max' in v['range']: field['max'] = v['range']['max']
        if v.get('allow_null') is not None:
            field['nullable'] = v['allow_null']
        schema[k] = field
    v = cerberus.Validator(schema)  # type: ignore
    if not v.validate(config):  # type: ignore
        raise ValueError(f"config.yaml schema validation failed: {v.errors}")  # type: ignore

_validate_yaml_schema(_raw_config)

# --- Assign config values from YAML ---
MODEL_PATH: str = _raw_config.get("model_path")
PROJECT_PATH: str = _raw_config.get("project_path")
GIT_REPO_URL: Optional[str] = _raw_config.get("git_repo_url")
PORT: int = _raw_config.get("port")
AUTH_PASSWORD: Optional[str] = _raw_config.get("auth_password")
RATE_LIMIT: Optional[int] = _raw_config.get("rate_limit")
MAX_NEW_TOKENS: int = _raw_config.get("max_new_tokens")
DEBUG: bool = _raw_config.get("debug")
TORCH_DTYPE: Optional[str] = _raw_config.get("torch_dtype")
CORS_ORIGINS: List[str] = _raw_config.get("cors_origins", ["*"])
LOGLEVEL: str = _raw_config.get("loglevel", "INFO")
MAX_MESSAGE_LENGTH: int = _raw_config.get("max_message_length", 1000)
ALLOWED_TOOLS: list = _raw_config.get("allowed_tools", ["createtestdoc"])

# --- Logging setup ---
logging.basicConfig(level=LOGLEVEL, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("jadio-quick-llm")

# --- Configuration validation function ---
def validate_config():
    """
    Validates all config values loaded from config.yaml to catch common mistakes early.
    - Ensures MODEL_PATH and PROJECT_PATH are non-empty strings.
    - Ensures PORT is a valid TCP port (1-65535).
    - If AUTH_PASSWORD is set, ensures it is a string of at least 4 characters.
    - If RATE_LIMIT is set, ensures it is a positive integer.
    - Ensures MAX_NEW_TOKENS is a positive integer.
    - Ensures DEBUG is a boolean.
    - Ensures TORCH_DTYPE is valid if set.
    - Ensures CORS_ORIGINS is a list of strings.
    - Ensures LOGLEVEL is valid.
    Raises AssertionError if any check fails.
    Used by main.py at startup and in the unit test below.
    """
    assert isinstance(MODEL_PATH, str) and MODEL_PATH, "MODEL_PATH must be a non-empty string."
    assert isinstance(PROJECT_PATH, str) and PROJECT_PATH, "PROJECT_PATH must be a non-empty string."
    assert isinstance(PORT, int) and 1 <= PORT <= 65535, "PORT must be a valid TCP port (1-65535)."
    if AUTH_PASSWORD is not None:
        assert isinstance(AUTH_PASSWORD, str) and len(AUTH_PASSWORD) >= 4, "AUTH_PASSWORD must be at least 4 chars."
    if RATE_LIMIT is not None:
        assert isinstance(RATE_LIMIT, int) and RATE_LIMIT > 0, "RATE_LIMIT must be a positive integer."
    assert isinstance(MAX_NEW_TOKENS, int) and MAX_NEW_TOKENS > 0, "MAX_NEW_TOKENS must be a positive integer."
    assert isinstance(DEBUG, bool), "DEBUG must be a boolean."
    if TORCH_DTYPE is not None:
        assert TORCH_DTYPE in ("float16", "float32", "bfloat16"), "TORCH_DTYPE must be float16, float32, or bfloat16."
    assert isinstance(CORS_ORIGINS, list) and len(CORS_ORIGINS) > 0 and all(isinstance(x, str) and x for x in CORS_ORIGINS), "CORS_ORIGINS must be a non-empty list of valid origins."
    assert LOGLEVEL in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), "LOGLEVEL must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL."
    assert isinstance(MAX_MESSAGE_LENGTH, int) and MAX_MESSAGE_LENGTH > 0, "MAX_MESSAGE_LENGTH must be a positive integer."
    assert isinstance(ALLOWED_TOOLS, list) and all(isinstance(t, str) and t for t in ALLOWED_TOOLS), "ALLOWED_TOOLS must be a list of non-empty strings."

# --- Unit test for config validation ---
if __name__ == "__main__":
    try:
        validate_config()
        logger.info("Config validation passed.")
    except AssertionError as e:
        logger.error(f"Config validation failed: {e}")
        sys.exit(1)
