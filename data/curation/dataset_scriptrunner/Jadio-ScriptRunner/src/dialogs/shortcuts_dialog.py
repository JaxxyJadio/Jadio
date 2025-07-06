from PyQt6.QtWidgets import QApplication, QMessageBox
from ..styles.styles import MESSAGEBOX_STYLE

def show_shortcuts(parent=None):
    text = (
        "<b>Keyboard Shortcuts</b><br><br>"
        "<ul>"
        "<li><b>Ctrl+S</b>: Save Configuration</li>"
        "<li><b>Ctrl+L</b>: Clear Terminal</li>"
        "<li><b>Ctrl+Shift+C</b>: Stop All Scripts</li>"
        "<li><b>Ctrl+Q</b>: Quit</li>"
        "<li><b>Left-click</b>: Execute Button Command</li>"
        "<li><b>Right-click</b>: Configure Button</li>"
        "</ul>"
    )
    box = QMessageBox(parent)
    box.setWindowTitle("Keyboard Shortcuts")
    box.setText(text)
    box.setStyleSheet(MESSAGEBOX_STYLE)
    box.setIcon(QMessageBox.Icon.Information)
    box.exec()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    show_shortcuts()
