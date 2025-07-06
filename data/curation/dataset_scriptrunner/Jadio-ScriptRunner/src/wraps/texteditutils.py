from PyQt6.QtWidgets import QTextEdit, QPlainTextEdit

def codeaigent_make_text_edit(object_name=None, read_only=False, placeholder=None, **kwargs):
    edit = QTextEdit()
    if object_name:
        edit.setObjectName(object_name)
    edit.setReadOnly(read_only)
    if placeholder:
        edit.setPlaceholderText(placeholder)
    for k, v in kwargs.items():
        setattr(edit, k, v)
    return edit

def codeaigent_make_plain_text_edit(object_name=None, read_only=False, placeholder=None, **kwargs):
    edit = QPlainTextEdit()
    if object_name:
        edit.setObjectName(object_name)
    edit.setReadOnly(read_only)
    if placeholder:
        edit.setPlaceholderText(placeholder)
    for k, v in kwargs.items():
        setattr(edit, k, v)
    return edit
