def best_fuzzy_match(query, choices, score_cutoff=70):
    try:
        from fuzzywuzzy import process
        result = process.extractOne(query, choices)
        if result is None:
            return None
        if len(result) == 2:
            match, score = result
        else:
            match, score, *_ = result
        if score >= score_cutoff:
            return match
        return None
    except ImportError:
        raise RuntimeError('fuzzywuzzy is not installed')
    except Exception:
        return None
