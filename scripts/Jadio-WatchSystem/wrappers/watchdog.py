from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

class SimpleWatchHandler(FileSystemEventHandler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
    def on_any_event(self, event):
        self.callback(event)

def start_watch(path, callback, recursive=True):
    """
    Starts watching the given path. Calls callback(event) on any file event.
    Returns the observer so you can stop it later.
    """
    event_handler = SimpleWatchHandler(callback)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=recursive)
    observer_thread = threading.Thread(target=observer.start, daemon=True)
    observer_thread.start()
    return observer

def stop_watch(observer):
    observer.stop()
    observer.join()
