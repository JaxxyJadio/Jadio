from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QLabel, QDialogButtonBox, QApplication
from ..styles.styles import DIALOG_STYLE

class ButtonConfigDialog(QDialog):
    def __init__(self, name='', command='', parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Button")
        self.setStyleSheet(DIALOG_STYLE)
        layout = QVBoxLayout(self)
        self.name_edit = QLineEdit(name)
        self.command_edit = QLineEdit(command)
        layout.addWidget(QLabel("Button Name:"))
        layout.addWidget(self.name_edit)
        layout.addWidget(QLabel("Shell Command:"))
        layout.addWidget(self.command_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    def get_values(self):
        return self.name_edit.text(), self.command_edit.text()

def show_button_config_dialog(name='', command='', parent=None):
    dialog = ButtonConfigDialog(name, command, parent)
    result = dialog.exec()
    if result == QDialog.DialogCode.Accepted:
        return dialog.get_values()
    return None, None

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    name, command = show_button_config_dialog()
    print(f"Name: {name}, Command: {command}")
