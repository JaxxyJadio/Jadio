from PyQt6.QtWidgets import QToolBar, QToolButton

def codeaigent_create_toolbar(parent=None):
    return QToolBar(parent)

def codeaigent_create_toolbutton(text='', parent=None, icon=None, slot=None):
    button = QToolButton(parent)
    button.setText(text)
    if icon:
        button.setIcon(icon)
    if slot:
        button.clicked.connect(slot)
    return button
