import yaml

def yaml_load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def yaml_dump(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
