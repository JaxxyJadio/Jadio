from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem

def codeaigent_create_graphicsview(parent=None):
    return QGraphicsView(parent)

def codeaigent_create_graphicsscene(parent=None):
    return QGraphicsScene(parent)
# QGraphicsItem is abstract, so typically subclassed; provide a utility for custom items if needed.
