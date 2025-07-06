import os
import sys
import yaml
import json
from config import get_section
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Bootstrap: find the registry file path from bootstrap.txt ---
def parse_bootstrap(path):
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, val = line.split(':', 1)
            mapping[key.strip()] = val.strip()
    return mapping

# Find bootstrap.txt (assume it's in the registry folder relative to this script)
BOOTSTRAP_TXT_PATH = os.path.join(os.path.dirname(__file__), '..', 'registry', 'bootstrap.txt')
if not os.path.exists(BOOTSTRAP_TXT_PATH):
    raise RuntimeError('bootstrap.txt not found')
bootstrap_map = parse_bootstrap(BOOTSTRAP_TXT_PATH)

# Get the output registry YAML path
INDEX_REGISTRY_PATH = bootstrap_map.get('INDEX_REGISTRY_PATH')
if not INDEX_REGISTRY_PATH:
    raise RuntimeError('INDEX_REGISTRY_PATH must be set in bootstrap.txt')

# Load payloads
INDEX_PAYLOAD_PATH = bootstrap_map.get('INDEX_PAYLOAD_PATH')
PATH_PAYLOAD_PATH = bootstrap_map.get('PATH_PAYLOAD_PATH')
if not INDEX_PAYLOAD_PATH or not PATH_PAYLOAD_PATH:
    raise RuntimeError('INDEX_PAYLOAD_PATH and PATH_PAYLOAD_PATH must be set in bootstrap.txt')
with open(INDEX_PAYLOAD_PATH, 'r', encoding='utf-8') as f:
    file_tree = json.load(f)
with open(PATH_PAYLOAD_PATH, 'r', encoding='utf-8') as f:
    field_stubs = json.load(f)

# Helper to preprocess field names
SUFFIXES = get_section('index_manager').get('suffixes', ['_PATH', '_FILE', '_DIR', '_ROOT'])
ignore_stubs = set(get_section('index_manager').get('ignore_stubs', ['SCRIPT_STUBS', 'FILE_PATTERN']))
def preprocess(field):
    base = field
    for suf in SUFFIXES:
        if base.endswith(suf):
            base = base[:-len(suf)]
    return base.lower().replace('_', '')

# Map stubs to paths (simple heuristic: fuzzy/endswith match, or None)
field_to_path = {}
failures = []
for field in field_stubs:
    if field in ignore_stubs:
        continue  # Ignore these stubs
    search = preprocess(field)
    match = None
    # Try endswith match
    for path in file_tree:
        if os.path.basename(path).lower().replace('_', '').replace('-', '').replace('.', '').endswith(search):
            match = path
            break
    if not match:
        for path in file_tree:
            if search in os.path.basename(path).lower().replace('_', '').replace('-', '').replace('.', ''):
                match = path
                break
    # If still not matched, check bootstrap.txt
    if not match and field in bootstrap_map:
        match = bootstrap_map[field]
    if match:
        field_to_path[field] = match
    else:
        field_to_path[field] = None
        failures.append(field)

# Add bootstrap.txt and registry path to the registry
field_to_path['BOOTSTRAP_TXT_PATH'] = BOOTSTRAP_TXT_PATH
field_to_path['INDEX_REGISTRY_PATH'] = INDEX_REGISTRY_PATH

# Add the full file index
field_to_path['FILE_INDEX'] = file_tree

# Write the new registry YAML
with open(INDEX_REGISTRY_PATH, 'w', encoding='utf-8') as f:
    yaml.dump(field_to_path, f, sort_keys=False, allow_unicode=True)

# Log and print failures
failures = [f for f in failures if f not in ignore_stubs]  # Exclude non-errors
if failures:
    print("[INDEX_MANAGER] Could not locate path for the following stubs:")
    for field in failures:
        print(f"  - {field}")
else:
    print("""
[INDEX_MANAGER] 🎉 All fields mapped and registry created successfully! 🎉
""")

print(f"[INDEX_MANAGER] Registry includes {len(file_tree)} files in FILE_INDEX")