from PyQt6.QtWidgets import QWidget

def codeaigent_make_widget(parent=None, object_name=None, **kwargs):
    w = QWidget(parent)
    if object_name:
        w.setObjectName(object_name)
    for k, v in kwargs.items():
        setattr(w, k, v)
    return w
