import os
import sys
import json
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

folder = os.path.dirname(os.path.abspath(__file__))
bootstrap_path = os.path.join(folder, '..', 'registry', 'bootstrap.txt')
bootstrap_map = parse_bootstrap(bootstrap_path)

PROJECT_ROOT = bootstrap_map['PROJECT_ROOT']
CONFIG_PATH = bootstrap_map['CONFIG_YAML']

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
   config = yaml.safe_load(f)
ignore_dirs = set(config.get('reference_dumper', {}).get('ignore_dirs', []))

def create_reference_skeleton():
   return {
       "title": "",
       "tl;dr": "",
       "dependencies": [],
       "examples": {},
       "DO": [],
       "DONT": [],
       "NEVER": [],
       "instructions": ""
   }

created_count = 0
for root, dirs, files in os.walk(PROJECT_ROOT):
   # Remove ignored directories in-place
   dirs[:] = [d for d in dirs if d not in ignore_dirs]
   reference_file = os.path.join(root, 'ai_reference.json')
   if not os.path.exists(reference_file):
       skeleton = create_reference_skeleton()
       with open(reference_file, 'w', encoding='utf-8') as f:
           json.dump(skeleton, f, indent=2)
       created_count += 1

print(f'[reference_dumper] Created {created_count} reference skeletons.')