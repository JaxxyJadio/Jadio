from PyQt6.QtWidgets import QProgressBar

def codeaigent_create_progressbar(parent=None, minimum=0, maximum=100, value=0, text_visible=True):
    progressbar = QProgressBar(parent)
    progressbar.setMinimum(minimum)
    progressbar.setMaximum(maximum)
    progressbar.setValue(value)
    progressbar.setTextVisible(text_visible)
    return progressbar
