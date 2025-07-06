from PyQt6.QtWidgets import QGroupBox

def codeaigent_create_groupbox(title='', parent=None, layout=None):
    groupbox = QGroupBox(title, parent)
    if layout:
        groupbox.setLayout(layout)
    return groupbox
