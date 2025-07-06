from PyQt6.QtWidgets import QLineEdit

def codeaigent_make_line_edit(text='', parent=None, object_name=None, **kwargs):
    edit = QLineEdit(text, parent)
    if object_name:
        edit.setObjectName(object_name)
    for k, v in kwargs.items():
        setattr(edit, k, v)
    return edit
