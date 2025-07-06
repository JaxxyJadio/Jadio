"""
A robust, debuggable, and user-friendly batch converter for raw data to JSONL.
- Uses the shortcut system for all path resolution (shortcut key only).
- Reads all settings from config (shortcut key only).
- Supports .json, .csv, .txt, .parquet, and fallback to text.
- Parallel processing, progress bar, colorized output, and detailed error/debug logging.
- Fails if required directories are missing.
"""

import os
import glob
import json
import csv
import yaml
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# Minimal bootstrap for shortcut system
SHORTCUTS_PATH = os.environ.get('SHORTCUTS_PATH')
if not SHORTCUTS_PATH:
    raise RuntimeError('SHORTCUTS_PATH environment variable must be set.')
with open(SHORTCUTS_PATH, 'r', encoding='utf-8') as f:
    bootstrap_shortcuts = yaml.safe_load(f)

def shortcut(key):
    path = bootstrap_shortcuts.get(key)
    if not path:
        print(Fore.RED + "[ERROR] Required shortcut key missing.")
        sys.exit(1)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

CONFIG_PATH = shortcut('config')
SHORTCUTS_PATH = shortcut('shortcuts')
if not os.path.exists(CONFIG_PATH):
    print(Fore.RED + "[ERROR] Required shortcut file missing.")
    sys.exit(1)
if not os.path.exists(SHORTCUTS_PATH):
    print(Fore.RED + "[ERROR] Required shortcut file missing.")
    sys.exit(1)
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
with open(SHORTCUTS_PATH, 'r', encoding='utf-8') as f:
    shortcuts = yaml.safe_load(f)

def resolve(key):
    path = shortcuts.get(key)
    if not path:
        print(Fore.RED + "[ERROR] Required shortcut key missing.")
        sys.exit(1)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

RAW_DUMP_DIR = resolve(config['input_dir'])
FACTORY_DIR = resolve(config['output_dir'])
LOGS_DIR = resolve(config['log_dir'])
MANIFEST_LOG = resolve(config['manifest_log'])
SUPPORTED_TYPES = set(config.get('supported_types', ['json', 'csv', 'txt', 'parquet']))
PARALLELISM = config.get('parallelism', os.cpu_count() or 2)
MAX_INFLIGHT = config.get('max_inflight_tasks', 0)
MAX_FILES = config.get('max_files', 0)
INCLUDE_PATTERNS = config.get('include_patterns', ['*.json', '*.csv', '*.txt', '*.parquet'])
EXCLUDE_PATTERNS = config.get('exclude_patterns', [])
LOG_LEVEL = config.get('log_level', 'INFO').upper()

# Fail if required directories are missing
for d in [RAW_DUMP_DIR, FACTORY_DIR, LOGS_DIR]:
    if not os.path.exists(d):
        print(Fore.RED + f"[ERROR] Required directory does not exist: {d}")
        sys.exit(1)


def debug(msg):
    if LOG_LEVEL == 'DEBUG':
        print(Fore.CYAN + f"[DEBUG] {msg}")

def error(msg):
    print(Fore.RED + f"[ERROR] {msg}")

def info(msg):
    print(Fore.GREEN + f"[INFO] {msg}")

def warn(msg):
    print(Fore.YELLOW + f"[WARN] {msg}")


def convert_file_to_jsonl(input_path, output_path):
    ext = os.path.splitext(input_path)[1].lower().lstrip('.')
    try:
        if ext == 'json':
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            payloads = data if isinstance(data, list) else [data]
        elif ext == 'csv':
            with open(input_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                payloads = list(reader)
        elif ext == 'txt':
            with open(input_path, 'r', encoding='utf-8') as f:
                payloads = [{"text": line.rstrip()} for line in f]
        elif ext == 'parquet':
            import pandas as pd
            df = pd.read_parquet(input_path, engine='fastparquet')
            payloads = df.to_dict(orient='records')
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                payloads = [{"text": f.read()}]
        with open(output_path, 'w', encoding='utf-8') as out:
            for item in payloads:
                out.write(json.dumps(item, ensure_ascii=False) + '\n')
        return True, None
    except Exception as e:
        tb = traceback.format_exc()
        error(f"Failed to convert {input_path}: {e}\n{tb}")
        return False, str(e)


def main():
    # File selection
    files = []
    for pattern in INCLUDE_PATTERNS:
        if pattern == '*' or pattern == ['*']:
            files.extend([os.path.join(RAW_DUMP_DIR, f) for f in os.listdir(RAW_DUMP_DIR)])
        else:
            files.extend(glob.glob(os.path.join(RAW_DUMP_DIR, pattern)))
    files = [f for f in files if os.path.isfile(f)]
    files = list(dict.fromkeys(files))
    if EXCLUDE_PATTERNS:
        import fnmatch
        for pat in EXCLUDE_PATTERNS:
            files = [f for f in files if not fnmatch.fnmatch(os.path.basename(f), pat)]
    if MAX_FILES and MAX_FILES > 0:
        files = files[:MAX_FILES]
    debug(f"RAW_DUMP_DIR: {RAW_DUMP_DIR}")
    debug(f"Files found: {files}")
    if not files:
        warn(f"No files found in {RAW_DUMP_DIR} matching {INCLUDE_PATTERNS}")
        return
    results = []
    errors = []
    info(f"Found {len(files)} files to process.")
    inflight = set()
    with tqdm(total=len(files), desc=Fore.BLUE + "Converting", unit="file") as pbar:
        with ProcessPoolExecutor(max_workers=PARALLELISM) as executor:
            future_to_file = {}
            for f in files:
                ext = os.path.splitext(f)[1].lower().lstrip('.')
                if ext not in SUPPORTED_TYPES:
                    warn(f"Skipping unsupported file type: {f}")
                    pbar.update(1)
                    continue
                # Enforce shortcut-driven output file path
                base = os.path.splitext(os.path.basename(f))[0]
                shortcut_key = f"output_file__{base}"
                try:
                    output_path = shortcut(shortcut_key)
                except SystemExit:
                    error(f"Missing output file shortcut for: {shortcut_key}")
                    pbar.update(1)
                    continue
                # Throttle inflight tasks if needed
                while MAX_INFLIGHT and len(inflight) >= MAX_INFLIGHT:
                    done, inflight = wait_any(inflight)
                    for fut in done:
                        pbar.update(1)
                fut = executor.submit(convert_file_to_jsonl, f, output_path)
                inflight.add(fut)
                future_to_file[fut] = f
            while inflight:
                done, inflight = wait_any(inflight)
                for fut in done:
                    pbar.update(1)
            for fut in future_to_file:
                f = future_to_file[fut]
                try:
                    success, err = fut.result()
                    if success:
                        results.append(f)
                    else:
                        errors.append((f, err))
                except Exception as exc:
                    errors.append((f, str(exc)))
    summary = f"convert_to_jsonl: found {len(files)} files, dumped payload successfully. {len(errors)} errors."
    info(summary)
    with open(MANIFEST_LOG, 'a', encoding='utf-8') as logf:
        logf.write(summary + '\n')
        if errors:
            for f, err in errors:
                logf.write(f"ERROR: {f}: {err}\n")

def wait_any(futures):
    done, not_done = wait(futures, return_when=FIRST_COMPLETED)
    return done, not_done

if __name__ == '__main__':
    main()
