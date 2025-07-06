from slugify import slugify


def path_slugify(script, field, path):
    """
    Returns a slugified string for deduplication based on script, field, and path.
    This helps normalize different representations of the same path.
    """
    # Normalize script and path using os.path.normpath, then slugify
    import os
    script_norm = os.path.normpath(script)
    path_norm = os.path.normpath(path)
    slug = slugify(f"{script_norm}-{field}-{path_norm}")
    return slug
