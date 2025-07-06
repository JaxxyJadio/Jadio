from PyQt6.QtWidgets import QMessageBox

def codeaigent_show_question(parent, title, text):
    return QMessageBox.question(parent, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
