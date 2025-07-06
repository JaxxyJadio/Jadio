def best_fuzzy_match(query, choices, score_cutoff=70):
    try:
        from rapidfuzz import process, fuzz
        match, score, _ = process.extractOne(query, choices, scorer=fuzz.ratio, score_cutoff=score_cutoff)
        return match if match else None
    except ImportError:
        raise RuntimeError('rapidfuzz is not installed')
    except Exception:
        return None
