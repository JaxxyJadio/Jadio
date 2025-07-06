from PyQt6.QtWidgets import QMdiArea, QMdiSubWindow

def codeaigent_create_mdiarea(parent=None):
    return QMdiArea(parent)

def codeaigent_create_mdisubwindow(parent=None):
    return QMdiSubWindow(parent)
