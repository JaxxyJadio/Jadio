import yaml
import os

class ConfigLoader:
    def __init__(self, config_path=None, shortcuts_path=None):
        # No hardcoded shortcut file path: must be provided by env or argument
        if shortcuts_path is None:
            shortcuts_path = os.environ.get('SHORTCUTS_PATH')
            if not shortcuts_path:
                raise RuntimeError("Shortcut file path must be provided via environment variable.")
        with open(shortcuts_path, 'r', encoding='utf-8') as f:
            shortcuts = yaml.safe_load(f)
        if config_path is None:
            config_path = shortcuts.get('config')
            if not config_path:
                raise RuntimeError("Required shortcut key missing.")
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), config_path))
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            print("[ConfigLoader] Config file not found. Using defaults.")
            return {}
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            print("[ConfigLoader] Loaded config.")
            return config
        except Exception as e:
            print(f"[ConfigLoader] Failed to load config: {e}")
            return {}

    def get(self, key, default=None):
        return self.config.get(key, default)
