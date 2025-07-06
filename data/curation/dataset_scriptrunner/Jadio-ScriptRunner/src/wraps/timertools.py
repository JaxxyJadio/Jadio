from PyQt6.QtCore import QTimer

def codeaigent_make_timer(interval, callback, start=True):
    timer = QTimer()
    timer.timeout.connect(callback)
    if start:
        timer.start(interval)
    return timer
