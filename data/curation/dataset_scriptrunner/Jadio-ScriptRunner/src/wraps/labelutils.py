from PyQt6.QtWidgets import QLabel

def codeaigent_make_label(text='', parent=None, object_name=None, **kwargs):
    label = QLabel(text, parent)
    if object_name:
        label.setObjectName(object_name)
    for k, v in kwargs.items():
        setattr(label, k, v)
    return label
