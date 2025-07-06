import yaml

def find_yaml_paths(yaml_path):
    """
    Extracts all fields with 'path' in their name from a YAML file.
    Returns a list of (field, value) tuples.
    """
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
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        recurse(data)
    except Exception:
        pass
    return results
