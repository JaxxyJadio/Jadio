from PyQt6.QtWidgets import QListWidget, QListView

def codeaigent_create_listwidget(parent=None):
    return QListWidget(parent)

def codeaigent_create_listview(parent=None, model=None):
    view = QListView(parent)
    if model:
        view.setModel(model)
    return view
