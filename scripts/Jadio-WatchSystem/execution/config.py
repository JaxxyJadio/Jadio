import os
import yaml
import threading

# --- Registry-driven config path resolution ---
def get_bootstrap_path():
    folder = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(folder, '..', 'registry', 'bootstrap.txt')

def parse_bootstrap(path):
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, val = line.split(':', 1)
            mapping[key.strip()] = val.strip()
    return mapping

_bootstrap_map = parse_bootstrap(get_bootstrap_path())
_CONFIG_PATH = _bootstrap_map.get('CONFIG_YAML') or ''
if not _CONFIG_PATH or not os.path.exists(_CONFIG_PATH):
    raise RuntimeError('Config YAML path not found in bootstrap.txt or file does not exist.')

_config_lock = threading.Lock()
_config_data = None

def _load_config():
    global _config_data
    with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
        _config_data = yaml.safe_load(f)

_load_config()

def get_config():
    with _config_lock:
        return _config_data.copy() if _config_data else {}

def get_section(section):
    cfg = get_config()
    return cfg.get(section, {})

def get_value(section, key, default=None):
    sec = get_section(section)
    return sec.get(key, default)

def reload_config():
    with _config_lock:
        _load_config()

# Optionally, add a watcher for live reloads if needed
# Example usage in other scripts:
# from config import get_config, get_section, get_value, reload_config
