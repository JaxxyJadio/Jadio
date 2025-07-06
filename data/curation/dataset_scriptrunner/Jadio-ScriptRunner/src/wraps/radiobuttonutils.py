from PyQt6.QtWidgets import QRadioButton

def codeaigent_create_radiobutton(text='', parent=None, checked=False, slot=None):
    radiobutton = QRadioButton(text, parent)
    radiobutton.setChecked(checked)
    if slot:
        radiobutton.toggled.connect(slot)
    return radiobutton
