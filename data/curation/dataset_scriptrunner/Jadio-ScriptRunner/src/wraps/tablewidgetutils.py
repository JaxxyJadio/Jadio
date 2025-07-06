from PyQt6.QtWidgets import QTableWidget, QTableView

def codeaigent_create_tablewidget(parent=None, rows=0, columns=0):
    table = QTableWidget(rows, columns, parent)
    return table

def codeaigent_create_tableview(parent=None, model=None):
    view = QTableView(parent)
    if model:
        view.setModel(model)
    return view
