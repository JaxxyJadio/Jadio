from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout

def codeaigent_vbox(parent=None, **kwargs):
    layout = QVBoxLayout(parent)
    for k, v in kwargs.items():
        setattr(layout, k, v)
    return layout

def codeaigent_hbox(parent=None, **kwargs):
    layout = QHBoxLayout(parent)
    for k, v in kwargs.items():
        setattr(layout, k, v)
    return layout

def codeaigent_grid(parent=None, **kwargs):
    layout = QGridLayout(parent)
    for k, v in kwargs.items():
        setattr(layout, k, v)
    return layout
