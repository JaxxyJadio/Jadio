from PyQt6.QtWidgets import QSpinBox, QDoubleSpinBox

def codeaigent_create_spinbox(parent=None, minimum=0, maximum=100, value=0, step=1, slot=None):
    spinbox = QSpinBox(parent)
    spinbox.setMinimum(minimum)
    spinbox.setMaximum(maximum)
    spinbox.setValue(value)
    spinbox.setSingleStep(step)
    if slot:
        spinbox.valueChanged.connect(slot)
    return spinbox

def codeaigent_create_doublespinbox(parent=None, minimum=0.0, maximum=100.0, value=0.0, step=0.1, decimals=2, slot=None):
    doublespinbox = QDoubleSpinBox(parent)
    doublespinbox.setMinimum(minimum)
    doublespinbox.setMaximum(maximum)
    doublespinbox.setValue(value)
    doublespinbox.setSingleStep(step)
    doublespinbox.setDecimals(decimals)
    if slot:
        doublespinbox.valueChanged.connect(slot)
    return doublespinbox
