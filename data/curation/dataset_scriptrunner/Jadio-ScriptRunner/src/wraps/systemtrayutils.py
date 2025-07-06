from PyQt6.QtWidgets import QSystemTrayIcon

def codeaigent_create_systemtrayicon(icon, parent=None, menu=None):
    tray = QSystemTrayIcon(icon, parent)
    if menu:
        tray.setContextMenu(menu)
    return tray
