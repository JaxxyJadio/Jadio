from PyQt6.QtWidgets import QSlider
from PyQt6.QtCore import Qt

def codeaigent_create_slider(orientation=Qt.Orientation.Horizontal, parent=None, minimum=0, maximum=100, value=0, slot=None):
    slider = QSlider(orientation, parent)
    slider.setMinimum(minimum)
    slider.setMaximum(maximum)
    slider.setValue(value)
    if slot:
        slider.valueChanged.connect(slot)
    return slider
