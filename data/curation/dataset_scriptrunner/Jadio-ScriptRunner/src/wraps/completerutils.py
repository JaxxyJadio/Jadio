from PyQt6.QtWidgets import QCompleter

def codeaigent_create_completer(items=None, parent=None):
    completer = QCompleter(items or [], parent)
    return completer
