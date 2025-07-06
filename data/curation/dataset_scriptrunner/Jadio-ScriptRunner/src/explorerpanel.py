import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QListWidget, QLabel

class ExplorerPanel(QWidget):
    def __init__(self, root_path=None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label = QLabel("Explorer")
        layout.addWidget(self.label)
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        self.root_path = root_path or os.getcwd()
        self.refresh()

    def refresh(self):
        self.list_widget.clear()
        for item in os.listdir(self.root_path):
            self.list_widget.addItem(item)
