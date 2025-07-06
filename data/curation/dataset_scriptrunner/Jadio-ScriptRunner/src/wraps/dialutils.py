from PyQt6.QtWidgets import QDial

def codeaigent_create_dial(parent=None, minimum=0, maximum=100, value=0, slot=None):
    dial = QDial(parent)
    dial.setMinimum(minimum)
    dial.setMaximum(maximum)
    dial.setValue(value)
    if slot:
        dial.valueChanged.connect(slot)
    return dial
