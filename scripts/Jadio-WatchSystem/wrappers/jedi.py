def find_path_assignments_jedi(script_path):
    """
    Uses jedi to find all assignments to variables with 'path' in their name.
    Returns a list of (field, value) tuples for string assignments, using Jedi's inference engine.
    """
    results = []
    try:
        import jedi
        with open(script_path, 'r', encoding='utf-8') as f:
            source = f.read()
        script = jedi.Script(source, path=script_path)
        for name in script.get_names(all_scopes=True, definitions=True, references=False):
            if 'path' in name.name.lower() and name.type == 'statement':
                try:
                    inferences = name.infer()
                    for inf in inferences:
                        if hasattr(inf, 'type') and inf.type in ('str', 'string'):
                            value = getattr(inf, 'string_value', None)
                            if value is None:
                                value = str(inf)
                            results.append((name.name, value))
                except Exception:
                    pass
    except Exception:
        pass
    return results
