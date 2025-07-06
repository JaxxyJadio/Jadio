import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wrappers.datetime import log_path_assistant
from wrappers.ast import find_python_fields
from wrappers.json import json_dump, json_load
from wrappers.glob import find_files_by_pattern
from config import get_section
import yaml

# --- Registry-driven path resolution ---
# 1. Read registry location from bootstrap.txt
BOOTSTRAP_PATH = os.path.join(os.path.dirname(__file__), '..', 'registry', 'bootstrap.txt')
with open(BOOTSTRAP_PATH, 'r', encoding='utf-8') as f:
   lines = f.readlines()
registry_map = {}
for line in lines:
   if ':' in line:
       k, v = line.strip().split(':', 1)
       registry_map[k.strip()] = v.strip()
# 2. Load registry YAML (use INDEX_REGISTRY_PATH for all stubs)
REGISTRY_YAML = registry_map.get('INDEX_REGISTRY_PATH')
if not REGISTRY_YAML or not os.path.exists(REGISTRY_YAML):
   raise RuntimeError('Registry YAML path not found in bootstrap.txt or file does not exist.')
with open(REGISTRY_YAML, 'r', encoding='utf-8') as f:
   registry = yaml.safe_load(f)
# 3. Resolve all required paths from registry
PROJECT_ROOT = registry.get('PROJECT_ROOT')
INDEX_PAYLOAD_PATH = registry.get('INDEX_PAYLOAD_PATH')
PATH_PAYLOAD_PATH = registry.get('PATH_PAYLOAD_PATH')
PATH_LOG_PATH = registry.get('PATH_LOG_PATH')
if not (PROJECT_ROOT and INDEX_PAYLOAD_PATH and PATH_PAYLOAD_PATH and PATH_LOG_PATH):
   raise RuntimeError('Required path stubs missing from registry.')
# 4. Load ignore fields/dirs from config.py backend
ignore_fields = set(get_section('path_assistant').get('ignore_fields', []))
ignore_dirs = set(get_section('path_assistant').get('ignore_dirs', []))

results = []
# Recursively find all Python scripts in the project
for script_path in find_files_by_pattern(PROJECT_ROOT, '**/*.py'):
    # Skip common directories that shouldn't be scanned
    path_parts = script_path.lower().split(os.sep)
    if any(skip_dir in path_parts for skip_dir in ignore_dirs):
        continue
    for field, value in find_python_fields(script_path):
        if isinstance(value, set):
            value = list(value)
        results.append({'script': script_path, 'field': field, 'path': value, 'file': None})

# Write ALL results (including duplicates) first
with open(PATH_PAYLOAD_PATH, 'w', encoding='utf-8') as f:
    json_dump(results, f, indent=2)
num_fields = len(results)
log_path_assistant(PATH_LOG_PATH, num_fields)
print(f'[path_assistant] Payload dumped with {num_fields} path fields (including duplicates).')

# Remove all lowercase fields, and for all others, keep only the first occurrence
with open(PATH_PAYLOAD_PATH, 'r', encoding='utf-8') as f:
    payload = json_load(f)
seen_fields = set()
removed = []
final_fields = []
for entry in payload:
    # Support both dict and str entries for backward compatibility
    if isinstance(entry, dict):
        field = entry.get('field', '')
    else:
        field = entry
    if isinstance(field, str) and (field.islower() or field in ignore_fields):
        removed.append(field)
    elif field not in seen_fields:
        final_fields.append(field)
        seen_fields.add(field)
    else:
        removed.append(field)
if removed:
    with open(PATH_LOG_PATH, 'a', encoding='utf-8') as logf:
        logf.write(f"[REMOVED] {len(removed)} fields removed\n")
if final_fields:
    with open(PATH_PAYLOAD_PATH, 'w', encoding='utf-8') as f:
        json_dump(final_fields, f, indent=2)
    num_fields = len(final_fields)
    log_path_assistant(PATH_LOG_PATH, num_fields)
    print(f'[path_assistant] Payload purged to {num_fields} unique fields.')