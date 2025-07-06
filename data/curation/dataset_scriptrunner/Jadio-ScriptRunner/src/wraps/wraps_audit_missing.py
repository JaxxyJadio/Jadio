import ast
from pathlib import Path

WRAPS_DIR = Path(__file__).parent
CLASS_LIST_FILE = WRAPS_DIR / 'pyqt6_class_list.txt'

def get_class_list():
    with open(CLASS_LIST_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    classes = [l.strip() for l in lines if l.startswith('PyQt6.')]
    return classes

def class_to_func(class_path):
    cls = class_path.split('.')[-1]
    return f'codeaigent_create_{cls.lower()}'

def get_existing_wraps():
    wraps = set()
    for pyfile in WRAPS_DIR.glob('*.py'):
        if pyfile.name.startswith('__') or 'audit' in pyfile.name:
            continue
        with open(pyfile, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(pyfile))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith('codeaigent_create_'):
                    wraps.add(node.name)
    return wraps

def main():
    classes = get_class_list()
    existing = get_existing_wraps()
    missing = []
    for class_path in classes:
        func = class_to_func(class_path)
        if func not in existing:
            missing.append((class_path, func))
    print(f"Missing wraps: {len(missing)}")
    for class_path, func in missing:
        print(f"  {func} for {class_path}")
    if not missing:
        print("All PyQt6 classes are wrapped!")

if __name__ == '__main__':
    main()
