from PyQt6.QtWidgets import QScrollArea

def codeaigent_create_scrollarea(parent=None, widget=None):
    area = QScrollArea(parent)
    if widget:
        area.setWidget(widget)
    return area
