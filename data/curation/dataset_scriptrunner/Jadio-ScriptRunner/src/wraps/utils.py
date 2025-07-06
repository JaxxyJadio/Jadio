from PyQt6.QtWidgets import QPushButton, QLabel

def codeaigent_make_button(text, object_name=None, **kwargs):
    btn = QPushButton(text)
    if object_name:
        btn.setObjectName(object_name)
    for k, v in kwargs.items():
        setattr(btn, k, v)
    return btn

def codeaigent_make_label(text, object_name=None, **kwargs):
    lbl = QLabel(text)
    if object_name:
        lbl.setObjectName(object_name)
    for k, v in kwargs.items():
        setattr(lbl, k, v)
    return lbl
