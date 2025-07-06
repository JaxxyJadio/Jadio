from PyQt6.QtGui import QShortcut

def codeaigent_create_shortcut(keysequence, parent, slot=None):
    shortcut = QShortcut(keysequence, parent)
    if slot:
        shortcut.activated.connect(slot)
    return shortcut
