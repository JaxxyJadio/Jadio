import glob as _glob
import os

def find_files_by_pattern(root_dir, pattern):
    """
    Recursively finds all files matching the given pattern under root_dir.
    Returns a list of absolute file paths.
    """
    matches = []
    for dirpath, _, _ in os.walk(root_dir):
        matches.extend(_glob.glob(os.path.join(dirpath, pattern), recursive=True))
    return matches
