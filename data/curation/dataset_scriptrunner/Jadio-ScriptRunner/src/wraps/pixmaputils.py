from PyQt6.QtGui import QPixmap, QImage, QClipboard
from PyQt6.QtWidgets import QApplication

def codeaigent_load_pixmap(path):
    return QPixmap(path)

def codeaigent_load_image(path):
    return QImage(path)

def codeaigent_copy_to_clipboard(text):
    app = QApplication.instance()
    if app is not None and hasattr(app, 'clipboard'):
        clipboard = app.clipboard()
        clipboard.setText(text)

def codeaigent_get_clipboard_text():
    app = QApplication.instance()
    if app is not None and hasattr(app, 'clipboard'):
        clipboard = app.clipboard()
        return clipboard.text()
    return ''
