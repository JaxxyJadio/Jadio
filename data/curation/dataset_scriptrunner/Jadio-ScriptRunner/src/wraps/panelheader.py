from PyQt6.QtWidgets import QLabel

def codeaigent_make_panel_header(text, object_name=None):
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    return label
