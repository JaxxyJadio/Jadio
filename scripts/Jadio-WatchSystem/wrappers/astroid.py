import astroid
import os
import json

WRAPPERS = {}

def register_wrapper(name):
    def decorator(func):
        WRAPPERS[name] = func
        return func
    return decorator

@register_wrapper('find_path_assignments')
def find_path_assignments(script_path):
    """
    Uses astroid to find all assignments to variables with 'path' in their name
    that are string literals. Returns a list of (field, value) tuples.
    Walks the full AST to find assignments at any depth.
    """
    results = []
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            tree = astroid.parse(f.read(), script_path)
        for node in tree.nodes_of_class(astroid.Assign):
            for target in node.targets:
                if isinstance(target, astroid.AssignName) and 'path' in target.name.lower():
                    # String literal
                    if isinstance(node.value, astroid.Const) and isinstance(node.value.value, str):
                        results.append((target.name, node.value.value))
                    # f-string
                    elif isinstance(node.value, astroid.JoinedStr):
                        try:
                            value = node.value.as_string()
                            results.append((target.name, value))
                        except Exception:
                            pass
                    # os.path.join('a', 'b', ...)
                    elif (isinstance(node.value, astroid.Call)
                          and hasattr(node.value.func, 'attrname')
                          and getattr(node.value.func, 'attrname', None) == 'join'):
                        try:
                            parts = []
                            for arg in node.value.args:
                                if isinstance(arg, astroid.Const) and isinstance(arg.value, str):
                                    parts.append(arg.value)
                            if parts:
                                value = os.path.join(*parts)
                                results.append((target.name, value))
                        except Exception:
                            pass
    except Exception:
        pass
    return results

@register_wrapper('path_assist')
def path_assist(index_payload_path, script_dirs):
    """
    Scans all scripts in script_dirs for path assignments, resolves them against the index payload,
    and returns a list of dicts with script, field, path, and file (if found in index).
    """
    with open(index_payload_path, 'r', encoding='utf-8') as f:
        all_files = set(json.load(f))
    script_files = []
    for subdir in script_dirs:
        for fname in os.listdir(subdir):
            if fname.endswith('.py'):
                script_files.append(os.path.join(subdir, fname))
    results = []
    for script in script_files:
        for field, value in WRAPPERS['find_path_assignments'](script):
            abs_path = os.path.normpath(os.path.join(os.path.dirname(script), value))
            if abs_path in all_files:
                results.append({'script': script, 'field': field, 'path': value, 'file': abs_path})
            else:
                results.append({'script': script, 'field': field, 'path': value, 'file': None})
    return results

# Example: you can add more wrappers and register them here
