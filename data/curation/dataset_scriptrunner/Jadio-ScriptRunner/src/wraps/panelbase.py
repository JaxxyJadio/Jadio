from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class CodeAigentPanelBase(QWidget):
    def __init__(self, object_name, style_sheet=None, header=None, parent=None):
        super().__init__(parent)
        self.setObjectName(object_name)
        if style_sheet:
            self.setStyleSheet(style_sheet)
        self.codeaigent_layout = QVBoxLayout(self)
        if header:
            self.codeaigent_header_label = QLabel(header)
            self.codeaigent_layout.addWidget(self.codeaigent_header_label)
