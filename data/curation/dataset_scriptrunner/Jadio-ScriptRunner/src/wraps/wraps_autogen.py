import os
import ast
from pathlib import Path
import re

WRAPS_DIR = Path(__file__).parent
CLASS_LIST_FILE = WRAPS_DIR / 'pyqt6_class_list.txt'
DRY_RUN = False  # Set to False to actually write files

# Utility to convert PyQt6 class name to a wrapper function name

def class_to_func(class_path):
    cls = class_path.split('.')[-1]
    return f'codeaigent_create_{cls.lower()}'

def get_existing_wraps():
    wraps = set()
    for pyfile in WRAPS_DIR.glob('*.py'):
        if pyfile.name.startswith('__') or pyfile.name.endswith('autogen.py'):
            continue
        with open(pyfile, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(pyfile))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith('codeaigent_'):
                    wraps.add(node.name)
    return wraps

def get_class_list():
    with open(CLASS_LIST_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    classes = [l.strip() for l in lines if l.startswith('PyQt6.')]
    return classes

def get_module_for_class(class_path):
    # Use the last part of the module path as the wraps file
    mod = class_path.split('.')[-2].lower()
    return WRAPS_DIR / f'{mod}utils.py'

def generate_wrapper_code(class_path):
    cls = class_path.split('.')[-1]
    func = class_to_func(class_path)
    code = (
        f"\ndef {func}(parent=None, *args, **kwargs):\n"
        f"    \"\"\"Auto-generated wrapper for {class_path}\"\"\"\n"
        f"    from {'.'.join(class_path.split('.')[:-1])} import {cls}\n"
        f"    return {cls}(parent, *args, **kwargs)\n"
    )
    return code

def main():
    existing = get_existing_wraps()
    classes = get_class_list()
    to_generate = []
    for class_path in classes:
        func = class_to_func(class_path)
        if func not in existing:
            to_generate.append((class_path, func))
    print(f"Found {len(to_generate)} missing wraps.")
    for class_path, func in to_generate:
        wraps_file = get_module_for_class(class_path)
        code = generate_wrapper_code(class_path)
        if DRY_RUN:
            print(f"Would add to {wraps_file.name}: {func}")
        else:
            with open(wraps_file, 'a', encoding='utf-8') as f:
                f.write(code)
    if DRY_RUN:
        print("Dry run complete. Set DRY_RUN = False to actually write wrappers.")
    else:
        print("Wrapper generation complete. All missing wraps have been written.")

if __name__ == '__main__':
    main()
