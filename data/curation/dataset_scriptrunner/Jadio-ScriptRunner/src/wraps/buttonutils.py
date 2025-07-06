from PyQt6.QtWidgets import QPushButton

def codeaigent_make_button(text='', parent=None, object_name=None, **kwargs):
    btn = QPushButton(text, parent)
    if object_name:
        btn.setObjectName(object_name)
    for k, v in kwargs.items():
        setattr(btn, k, v)
    return btn
