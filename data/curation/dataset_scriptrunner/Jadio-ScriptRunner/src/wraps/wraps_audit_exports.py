import ast
from pathlib import Path

WRAPS_DIR = Path(__file__).parent
INIT_FILE = WRAPS_DIR / '__init__.py'

def get_all_wraps():
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

def get_init_exports():
    with open(INIT_FILE, 'r', encoding='utf-8') as f:
        code = f.read()
    exports = set()
    for line in code.splitlines():
        if 'codeaigent_create_' in line:
            for part in line.split('"'):
                if part.startswith('codeaigent_create_'):
                    exports.add(part)
    return exports

def main():
    wraps = get_all_wraps()
    exports = get_init_exports()
    missing = wraps - exports
    if missing:
        print(f"Wraps missing from __init__.py: {len(missing)}")
        for name in sorted(missing):
            print(f"  {name}")
    else:
        print("All wraps are exported in __init__.py!")

if __name__ == '__main__':
    main()
