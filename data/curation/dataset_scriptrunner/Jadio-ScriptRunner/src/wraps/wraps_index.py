import os
import ast
from pathlib import Path

WRAPS_DIR = Path(__file__).parent
INDEX_FILE = WRAPS_DIR / 'wraps_index.md'

def get_docstring(node):
    return ast.get_docstring(node) or ''

def build_index():
    lines = ['# CodeAigent Wraps Index\n']
    for pyfile in sorted(WRAPS_DIR.glob('*.py')):
        if pyfile.name.startswith('__') or pyfile.name.endswith('_inspect.py'):
            continue
        with open(pyfile, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(pyfile))
            funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('codeaigent_')]
            if funcs:
                lines.append(f'## {pyfile.name}\n')
                for fn in funcs:
                    doc = get_docstring(fn)
                    lines.append(f'- **{fn.name}**: {doc}')
                lines.append('')
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Wraps index written to {INDEX_FILE}')

if __name__ == '__main__':
    build_index()
