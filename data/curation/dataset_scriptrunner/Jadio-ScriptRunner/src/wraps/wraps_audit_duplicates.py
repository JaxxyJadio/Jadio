import ast
from pathlib import Path
from collections import Counter

WRAPS_DIR = Path(__file__).parent

def get_all_wraps():
    wraps = []
    for pyfile in WRAPS_DIR.glob('*.py'):
        if pyfile.name.startswith('__') or 'audit' in pyfile.name:
            continue
        with open(pyfile, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(pyfile))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith('codeaigent_create_'):
                    wraps.append((node.name, pyfile.name))
    return wraps

def main():
    wraps = get_all_wraps()
    names = [w[0] for w in wraps]
    counter = Counter(names)
    dups = [name for name, count in counter.items() if count > 1]
    if dups:
        print(f"Duplicate wraps found: {len(dups)}")
        for name in dups:
            files = [f for n, f in wraps if n == name]
            print(f"  {name} in {files}")
    else:
        print("No duplicate wraps found.")

if __name__ == '__main__':
    main()
