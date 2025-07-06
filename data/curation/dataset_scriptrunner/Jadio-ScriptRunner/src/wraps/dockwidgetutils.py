from PyQt6.QtWidgets import QDockWidget

def codeaigent_create_dockwidget(title='', parent=None, widget=None):
    dock = QDockWidget(title, parent)
    if widget:
        dock.setWidget(widget)
    return dock
