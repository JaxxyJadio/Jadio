from PyQt6.QtWidgets import QComboBox

def codeaigent_make_combo_box(items=None, object_name=None, **kwargs):
    combo = QComboBox()
    if object_name:
        combo.setObjectName(object_name)
    if items:
        combo.addItems(items)
    for k, v in kwargs.items():
        setattr(combo, k, v)
    return combo
