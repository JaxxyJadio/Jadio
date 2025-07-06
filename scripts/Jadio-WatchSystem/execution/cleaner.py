import os
import sys
import time
import yaml
from datetime import datetime

# --- Registry-driven config and log file resolution ---
folder = os.path.dirname(os.path.abspath(__file__))
bt_path = os.path.join(folder, '..', 'registry', 'bootstrap.txt')
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
bootstrap_map = parse_bootstrap(bt_path)
# Find all stubs ending with _LOG_PATH
log_paths = [v for k, v in bootstrap_map.items() if k.endswith('_LOG_PATH')]
log_files = [p for p in log_paths if os.path.isfile(p)]
CLEANER_LOG_PATH = bootstrap_map.get('CLEANER_LOG_PATH')
# Get config.yaml path from registry
CONFIG_PATH = bootstrap_map.get('CONFIG_YAML') or bootstrap_map.get('CONFIG')
if CONFIG_PATH is not None and not os.path.isabs(CONFIG_PATH):
    # Resolve relative to project root
    PROJECT_ROOT = bootstrap_map.get('PROJECT_ROOT', '.')
    CONFIG_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, CONFIG_PATH))

# Read max_lines from config.yaml
max_lines = 50
if CONFIG_PATH:
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        max_lines = int(config.get('cleaner', {}).get('max_lines', 50))
    except Exception:
        max_lines = 50

report_lines = []
trimmed_any = False
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
for log_file in log_files:
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        trimmed = False
        if len(lines) > max_lines:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.writelines(lines[-max_lines:])
            msg = f"[cleaner] Trimmed {log_file} to last {max_lines} lines."
            print(msg)
            report_lines.append(msg + '\n')
            trimmed_any = True
            trimmed = True
        # Always append the cleaned timestamp
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"LOG CLEANED AT {now}\n")
        if not trimmed:
            report_lines.append(f"[cleaner] Appended clean timestamp to {log_file}\n")
    except Exception as e:
        err = f"[cleaner] Failed to trim {log_file}: {e}"
        print(err)
        report_lines.append(err + '\n')
if not trimmed_any:
    happy = """
[cleaner] 🎉🎉🎉 All log files are already sparkling clean! Nothing to trim! 🎉🎉🎉
  (•̀ᴗ•́)و ̑̑  Your logs are tidy and beautiful.  
  Keep up the registry-driven magic! ✨🚀
"""
    print(happy)
    report_lines.append(happy)
if CLEANER_LOG_PATH:
    with open(CLEANER_LOG_PATH, 'a', encoding='utf-8') as f:
        f.writelines(report_lines)
