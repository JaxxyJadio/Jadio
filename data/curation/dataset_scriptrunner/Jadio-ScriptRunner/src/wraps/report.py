import os
import re

PANEL_DIR = os.path.join(os.path.dirname(__file__), '..')
WRAPS_INIT = os.path.join(os.path.dirname(__file__), '__init__.py')

def get_all_wrappers():
    with open(WRAPS_INIT, 'r', encoding='utf-8') as f:
        content = f.read()
    return set(re.findall(r'(codeaigent_\w+)', content))

def main():
    wrappers = get_all_wrappers()
    used = set()
    for fname in os.listdir(PANEL_DIR):
        if fname.endswith('_panel.py'):
            with open(os.path.join(PANEL_DIR, fname), 'r', encoding='utf-8') as f:
                code = f.read()
            used.update(re.findall(r'(codeaigent_\w+)', code))
    missing = used - wrappers
    if missing:
        print("Missing wrappers:", missing)
    else:
        print("No missing wrappers found.")

if __name__ == '__main__':
    main()