import os
import re

PANEL_DIR = os.path.join(os.path.dirname(__file__), '..')
WRAPS_IMPORT = 'from wraps import'
WRAP_PREFIX = 'codeaigent_'

def get_all_wrappers():
    # Parse __init__.py to get all exported wrappers
    wraps_init = os.path.join(os.path.dirname(__file__), '__init__.py')
    with open(wraps_init, 'r', encoding='utf-8') as f:
        content = f.read()
    return set(re.findall(r'(codeaigent_\w+)', content))

def rewrap_file(filepath, wrappers):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    # Replace direct PyQt6 usage with wrappers
    for wrap in wrappers:
        # e.g., replace QPushButton( with codeaigent_QPushButton(
        base = wrap.replace(WRAP_PREFIX, '')
        code = re.sub(rf'(?<!\w){base}\(', f'{wrap}(', code)
    # Remove direct PyQt6 imports
    code = re.sub(r'from PyQt6[^\n]+\n', '', code)
    code = re.sub(r'import PyQt6[^\n]+\n', '', code)
    # Ensure only wraps are imported
    code = re.sub(r'from wraps import .*\n', '', code)
    # Add correct wraps import at the top
    import_line = f"{WRAPS_IMPORT} (\n    " + ',\n    '.join(sorted(wrappers)) + "\n)\n"
    code = import_line + code
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)

def main():
    wrappers = get_all_wrappers()
    for fname in os.listdir(PANEL_DIR):
        if fname.endswith('_panel.py'):
            fpath = os.path.join(PANEL_DIR, fname)
            rewrap_file(fpath, wrappers)
            print(f"Rewrapped: {fname}")

if __name__ == '__main__':
    main()