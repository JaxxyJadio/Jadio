def find_json_paths(json_path):
    """
    Extracts all fields with 'path' in their name from a JSON file.
    Returns a list of (field, value) tuples.
    """
    import json
    results = []
    def recurse(obj, parent_key=''):
        if isinstance(obj, dict):
            for k, v in obj.items():
                full_key = f'{parent_key}.{k}' if parent_key else k
                if 'path' in k.lower() and isinstance(v, str):
                    results.append((full_key, v))
                recurse(v, full_key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                recurse(item, f'{parent_key}[{i}]')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        recurse(data)
    except Exception:
        pass
    return results

def json_load(f):
    import json
    return json.load(f)

def json_dump(obj, f, indent=2):
    import json
    return json.dump(obj, f, indent=indent)
