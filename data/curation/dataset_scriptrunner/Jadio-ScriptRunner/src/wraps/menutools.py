from PyQt6.QtWidgets import QAction, QMenu

def codeaigent_add_menu(menubar, title):
    menu = menubar.addMenu(title)
    return menu

def codeaigent_add_action(menu, text, shortcut=None, callback=None):
    action = QAction(text, menu)
    if shortcut:
        action.setShortcut(shortcut)
    if callback:
        action.triggered.connect(callback)
    menu.addAction(action)
    return action

def codeaigent_add_separator(menu):
    menu.addSeparator()
