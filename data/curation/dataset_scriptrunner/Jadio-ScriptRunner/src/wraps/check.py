import os
import re

PANEL_DIR = os.path.join(os.path.dirname(__file__), '..')

def check_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    if re.search(r'from PyQt6|import PyQt6', code):
        print(f"Direct PyQt6 usage found in {filepath}")

def main():
    for fname in os.listdir(PANEL_DIR):
        if fname.endswith('_panel.py'):
            fpath = os.path.join(PANEL_DIR, fname)
            check_file(fpath)

if __name__ == '__main__':
    main()