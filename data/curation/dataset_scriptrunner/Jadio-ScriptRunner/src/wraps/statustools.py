from PyQt6.QtWidgets import QStatusBar

def codeaigent_make_status_bar(window, initial_message=None):
    status_bar = QStatusBar()
    window.setStatusBar(status_bar)
    if initial_message:
        status_bar.showMessage(initial_message)
    return status_bar
