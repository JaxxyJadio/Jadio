from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit

class EditorPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.editor = QTextEdit()
        self.editor.setPlaceholderText("Write or edit your script here...")
        layout.addWidget(self.editor)

    def get_text(self):
        return self.editor.toPlainText()

    def set_text(self, text):
        self.editor.setPlainText(text)
