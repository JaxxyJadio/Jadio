import ast
from pathlib import Path
from collections import defaultdict

WRAPS_DIR = Path(__file__).parent

def get_all_wraps():
    wraps = defaultdict(list)
    for pyfile in WRAPS_DIR.glob('*.py'):
        if pyfile.name.startswith('__') or 'audit' in pyfile.name:
            continue
        with open(pyfile, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(pyfile))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith('codeaigent_create_'):
                    wraps[pyfile.name].append(node.name)
    return wraps

def main():
    wraps = get_all_wraps()
    total = sum(len(fns) for fns in wraps.values())
    print(f"Total wraps: {total}")
    for fname, fns in sorted(wraps.items()):
        print(f"  {fname}: {len(fns)} wraps")
    if not wraps:
        print("No wraps found.")

if __name__ == '__main__':
    main()
