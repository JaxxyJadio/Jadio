import os
import sys
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wrappers.glob import find_files_by_pattern

# Set up paths
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
FLAGS_LOG_PATH = bootstrap_map['FLAGS_LOG_PATH']

# Only match file extensions that exist in the project
FILE_PATTERN = re.compile(r"['\"]([^'\"]+\.(py|json|jsonl|yaml|txt|dm))['\"]", re.IGNORECASE)

found_flag = False
# Recursively find all Python scripts in the project (excluding wrappers)
for script_path in find_files_by_pattern(PROJECT_ROOT, '**/*.py'):
    if 'wrappers' in script_path.lower().split(os.sep):
        continue
    with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if (stripped.startswith('print') or stripped.startswith('raise')) and '(' in stripped and ')' in stripped:
                continue
            for match in FILE_PATTERN.finditer(line):
                file_name = match.group(1)
                if file_name == '**/*.py' or 'bootstrap.txt' in file_name.lower():
                    continue
                found_flag = True
                with open(FLAGS_LOG_PATH, 'a', encoding='utf-8') as logf:
                    logf.write(f"[FLAG] {script_path}:{lineno}: {file_name}\n")
                print(f"[FLAG] {script_path}:{lineno}: {file_name}")
if not found_flag:
    with open(FLAGS_LOG_PATH, 'a', encoding='utf-8') as logf:
        logf.write('[FLAGGER] 🎉🎉🎉 No hardcoded file names found! Your codebase is registry-pure! 🎉🎉🎉\n')
        logf.write('  \n  (•̀ᴗ•́)و ̑̑  All scripts are compliant and sparkling clean!  \n  Keep up the registry-driven magic! ✨🚀\n')
    print("""
[FLAGGER] 🎉🎉🎉 No hardcoded file names found! Your codebase is registry-pure! 🎉🎉🎉
  
  (•̀ᴗ•́)و ̑̑  All scripts are compliant and sparkling clean!  
  Keep up the registry-driven magic! ✨🚀
    """)
