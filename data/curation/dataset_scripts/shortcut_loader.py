import os
import yaml
import re
import sys

class ShortcutLoader:
    def __init__(self, shortcut_path=None):
        # No hardcoded shortcut file path: must be provided by env or argument
        if shortcut_path is None:
            shortcut_path = os.environ.get('SHORTCUTS_PATH')
            if not shortcut_path:
                raise RuntimeError("Shortcut file path must be provided via environment variable.")
        self.shortcut_path = shortcut_path
        self.shortcuts = self.load_shortcuts()

    def load_shortcuts(self):
        with open(self.shortcut_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get(self, key):
        return self.shortcuts.get(key)

    def substitute(self, text):
        for key, value in self.shortcuts.items():
            placeholder = str(value)
            if placeholder.isupper():
                text = re.sub(rf'\b{placeholder}\b', str(self.shortcuts.get(key)), text)
        return text

if __name__ == '__main__':
    loader = ShortcutLoader()
    if len(sys.argv) == 1:
        for k, v in loader.shortcuts.items():
            print(f'{k}: {v}')
    elif len(sys.argv) == 2:
        print(loader.substitute(sys.argv[1]))
    else:
        print('Usage: [string_with_shortcuts]')
