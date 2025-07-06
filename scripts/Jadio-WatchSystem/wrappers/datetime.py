from datetime import datetime


def log_message(log_path, assistant_name, message):
    """
    Appends a timestamped log message to the specified log file.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f'[{timestamp}] [{assistant_name}] {message}\n')


def log_path_assistant(log_path, num_fields):
    """
    Appends a log entry for path_assistant with the correct message format.
    """
    log_message(log_path, 'path_assistant', f'"Payload dumped with {num_fields} path fields."')

# Add more assistant-specific loggers here as needed.
