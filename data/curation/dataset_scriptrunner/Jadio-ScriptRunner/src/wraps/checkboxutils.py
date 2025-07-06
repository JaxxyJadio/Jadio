from PyQt6.QtWidgets import QCheckBox

def codeaigent_create_checkbox(text='', parent=None, checked=False, slot=None):
    checkbox = QCheckBox(text, parent)
    checkbox.setChecked(checked)
    if slot:
        checkbox.stateChanged.connect(slot)
    return checkbox
