from PyQt6.QtWidgets import QInputDialog, QErrorMessage

def codeaigent_get_text_input(parent=None, title='', label='', mode=None):
    return QInputDialog.getText(parent, title, label, mode) if mode else QInputDialog.getText(parent, title, label)

def codeaigent_show_error_message(parent=None, message=''):
    error = QErrorMessage(parent)
    error.showMessage(message)
    return error
