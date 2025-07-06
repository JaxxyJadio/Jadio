from PyQt6.QtWidgets import QTabWidget, QWidget

def codeaigent_make_tab_widget(object_name=None):
    tab = QTabWidget()
    if object_name:
        tab.setObjectName(object_name)
    return tab

def codeaigent_add_tab(tab_widget, widget, title):
    tab_widget.addTab(widget, title)
