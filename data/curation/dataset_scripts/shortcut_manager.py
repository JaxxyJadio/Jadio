import os
import yaml
import re
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
import click

# No hardcoded shortcut file path: must be provided by env
SHORTCUTS_FILE = os.environ.get('SHORTCUTS_PATH')
if not SHORTCUTS_FILE:
    raise RuntimeError("Shortcut file path must be provided via environment variable.")
SHORTCUTS_FILE = str(SHORTCUTS_FILE)
SCRIPTS_DIR = Path(__file__).parent
console = Console()

# Load shortcuts
def load_shortcuts():
    path = str(SHORTCUTS_FILE)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def invert_shortcuts(shortcuts):
    # Map actual value to placeholder (e.g., '../dataset_rawdump' -> 'INPUTDIR')
    mapping = {}
    for key, value in shortcuts.items():
        if isinstance(value, str) and value and not value.isupper():
            mapping[value] = key.upper()
    return mapping

def apply_shortcuts_to_script(script_path, value_to_placeholder):
    with open(script_path, 'r', encoding='utf-8') as f:
        code = f.read()
    replaced = False
    replacements = []
    # Replace all actual values with their ALLCAPS shortcut
    for value, placeholder in value_to_placeholder.items():
        # Use regex to match only full path/word
        new_code, n = re.subn(rf'(?<![\w/]){re.escape(value)}(?![\w/])', placeholder, code)
        if n > 0:
            code = new_code
            replaced = True
            replacements.append((value, placeholder, n))
    if replaced:
        # Backup original
        backup_path = script_path.with_suffix(script_path.suffix + '.bak')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(code)
        # Overwrite script with replaced code
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(code)
    return replaced, replacements

@click.command()
@click.option('--dry-run', is_flag=True, help='Show what would be replaced, but do not modify files.')
def main(dry_run):
    shortcuts = load_shortcuts()
    value_to_placeholder = invert_shortcuts(shortcuts)
    PY_FILE_PATTERN = os.environ.get('PY_FILE_PATTERN')
    if not PY_FILE_PATTERN:
        raise RuntimeError('PY_FILE_PATTERN environment variable must be set.')
    py_files = [p for p in SCRIPTS_DIR.rglob(PY_FILE_PATTERN)]
    table = Table(title="Shortcut Replacements", show_lines=True)
    table.add_column("File")
    table.add_column("Replaced Value")
    table.add_column("With Shortcut")
    table.add_column("Count", justify="right")
    changed_files = 0
    with tqdm(total=len(py_files), desc="Processing scripts", unit="file") as pbar:
        for script_path in py_files:
            replaced, replacements = apply_shortcuts_to_script(script_path, value_to_placeholder) if not dry_run else (False, [])
            if replaced or dry_run:
                changed_files += 1
                for value, placeholder, n in replacements:
                    table.add_row(str(script_path), value, placeholder, str(n))
            pbar.update(1)
    if changed_files:
        console.print(table)
        console.print(f"[bold green]Processed {changed_files} files with replacements.[/bold green]")
    else:
        console.print("[bold yellow]No replacements made.[/bold yellow]")

if __name__ == '__main__':
    main()
