import ast
import os

def find_python_paths(script_path):
    """
    Extracts all assignments to variables with 'path' in their name from a Python file.
    Handles string literals, f-strings, and os.path.join calls at any depth.
    Returns a list of (field, value) tuples.
    """
    results = []
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=script_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and 'path' in target.id.lower():
                        # String literal
                        if isinstance(node.value, ast.Str):
                            results.append((target.id, node.value.s))
                        # f-string
                        elif isinstance(node.value, ast.JoinedStr):
                            value = ''.join([
                                part.s if isinstance(part, ast.Str) else ''
                                for part in node.value.values
                            ])
                            results.append((target.id, value))
                        # os.path.join('a', 'b', ...)
                        elif (isinstance(node.value, ast.Call)
                              and isinstance(node.value.func, ast.Attribute)
                              and node.value.func.attr == 'join'):
                            parts = []
                            for arg in node.value.args:
                                if isinstance(arg, ast.Str):
                                    parts.append(arg.s)
                            if parts:
                                value = os.path.join(*parts)
                                results.append((target.id, value))
    except Exception:
        pass
    return results

def find_python_fields(script_path):
    """
    Extract all top-level variable assignments (field stubs) from a Python file.
    Returns a list of (field, value) tuples.
    """
    results = []
    try:
        import ast
        with open(script_path, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source, filename=script_path)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        field = target.id
                        try:
                            value = ast.literal_eval(node.value)
                        except Exception:
                            value = None
                        results.append((field, value))
    except Exception:
        pass
    return results
