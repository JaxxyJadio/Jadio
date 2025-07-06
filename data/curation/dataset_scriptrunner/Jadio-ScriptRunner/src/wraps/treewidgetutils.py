from PyQt6.QtWidgets import QTreeWidget, QTreeView

def codeaigent_create_treewidget(parent=None):
    return QTreeWidget(parent)

def codeaigent_create_treeview(parent=None, model=None):
    view = QTreeView(parent)
    if model:
        view.setModel(model)
    return view
