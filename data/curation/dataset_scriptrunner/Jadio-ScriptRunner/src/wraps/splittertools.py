from PyQt6.QtWidgets import QSplitter

def codeaigent_make_splitter(orientation, widgets, sizes=None):
    splitter = QSplitter(orientation)
    for w in widgets:
        splitter.addWidget(w)
    if sizes:
        splitter.setSizes(sizes)
    return splitter
