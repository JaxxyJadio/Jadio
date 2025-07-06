import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_section

# Bootstrap: find all index registry paths from bootstrap.txt
BOOTSTRAP_PATH = os.path.join(os.path.dirname(__file__), '..', 'registry', 'bootstrap.txt')
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

bootstrap_map = parse_bootstrap(BOOTSTRAP_PATH)
PROJECT_ROOT = bootstrap_map['PROJECT_ROOT']
INDEX_REGISTRY_PATH = bootstrap_map['INDEX_REGISTRY_PATH']
INDEX_PAYLOAD_PATH = bootstrap_map['INDEX_PAYLOAD_PATH']
ignore_dirs = set(get_section('index_assistant').get('ignore_dirs', []))


def scan_project_tree(root_dir):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Ignore directories from config
        dirnames[:] = [d for d in dirnames if d.lower() not in ignore_dirs]
        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root_dir)  # Make relative to PROJECT_ROOT
            file_paths.append(os.path.normpath(rel_path))
    return file_paths


def dump_index_payload(paths):
    with open(INDEX_PAYLOAD_PATH, 'w', encoding='utf-8') as f:
        json.dump(paths, f, indent=2)


def main():
    files = scan_project_tree(PROJECT_ROOT)
    dump_index_payload(files)
    print(f'Payload file overwritten with {len(files)} file paths.')

    with open(INDEX_PAYLOAD_PATH, 'r', encoding='utf-8') as f:
        file_list = json.load(f)
    # Remove any entries containing '__pycache__' (case-insensitive)
    filtered = [p for p in file_list if '__pycache__' not in p.lower()]
    removed = [p for p in file_list if '__pycache__' in p.lower()]
    if removed:
        print('[index_assistant] The following entries were removed because they contain pycache:')
        for p in removed:
            print('  ', p)
        print(f'[index_assistant] {len(removed)} pycache entries removed. {len(filtered)} entries remain.')
    else:
        print("""
🎉🎉🎉 [index_assistant] Hooray! No __pycache__ entries found! 🎉🎉🎉
Your project index is sparkling clean and fully compliant!
Keep up the great work! 🚀✨
        """)
        print(f'[index_assistant] {len(filtered)} entries remain.')
    with open(INDEX_PAYLOAD_PATH, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2)
    print(f'[index_assistant] {len(removed)} pycache entries removed. {len(filtered)} entries remain.')

    if not removed:
        print('\n🎉🎉🎉 [index_assistant] No __pycache__ entries found! Your project is sparkling clean! 🎉🎉🎉\n')


if __name__ == "__main__":
    main()
