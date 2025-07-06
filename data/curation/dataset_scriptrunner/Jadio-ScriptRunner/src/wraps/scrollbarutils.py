from PyQt6.QtWidgets import QScrollBar
from PyQt6.QtCore import Qt

def codeaigent_create_scrollbar(orientation=Qt.Orientation.Vertical, parent=None, minimum=0, maximum=100, value=0, slot=None):
    bar = QScrollBar(orientation, parent)
    bar.setMinimum(minimum)
    bar.setMaximum(maximum)
    bar.setValue(value)
    if slot:
        bar.valueChanged.connect(slot)
    return bar
