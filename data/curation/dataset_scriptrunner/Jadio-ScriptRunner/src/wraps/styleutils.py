from PyQt6.QtWidgets import QStyle, QStyleOption, QStylePainter

def codeaigent_create_style(parent=None):
    return QStyle(parent) if parent else QStyle()

def codeaigent_create_styleoption():
    return QStyleOption()

def codeaigent_create_stylepainter(widget):
    return QStylePainter(widget)
