from PyQt6.QtWidgets import QCommandLinkButton

def codeaigent_create_commandlinkbutton(text='', parent=None, description='', slot=None):
    btn = QCommandLinkButton(text, description, parent)
    if slot:
        btn.clicked.connect(slot)
    return btn
