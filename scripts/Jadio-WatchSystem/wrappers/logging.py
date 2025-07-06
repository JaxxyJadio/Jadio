def get_logger(name=None, level='INFO'):
    import logging
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger

def log_message(log_path, message, level='INFO', name=None):
    import logging
    logger = get_logger(name, level)
    getattr(logger, level.lower(), logger.info)(message)
    # Also write to the log file specified by log_path
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{level}] {message}\n")
