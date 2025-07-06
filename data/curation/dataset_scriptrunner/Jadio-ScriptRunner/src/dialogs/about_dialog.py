from PyQt6.QtWidgets import QApplication, QMessageBox
from ..styles.styles import MESSAGEBOX_STYLE

def show_about(parent=None):
    text = (
        "<b>Jadio-ScriptRunner v1.1</b><br><br>"
        "Features:<ul>"
        "<li>50 customizable buttons (10x5 grid)</li>"
        "<li>Real-time shell output (cross-platform)</li>"
        "<li>Thread-safe script execution</li>"
        "<li>Auto-save configuration</li>"
        "<li>Modern dark theme</li>"
        "<li>Buffered and batched output for high-volume scripts</li>"
        "<li>Configurable default shell and concurrency</li>"
        "<li>HTML-escaped output for security</li>"
        "<li>Process group kill for robust script stopping</li>"
        "</ul>"
    )
    box = QMessageBox(parent)
    box.setWindowTitle("About Jadio-ScriptRunner")
    box.setText(text)
    box.setStyleSheet(MESSAGEBOX_STYLE)
    box.setIcon(QMessageBox.Icon.Information)
    box.exec()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    show_about()
