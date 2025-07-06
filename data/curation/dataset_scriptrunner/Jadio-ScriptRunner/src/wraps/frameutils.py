from PyQt6.QtWidgets import QFrame

def codeaigent_create_frame(parent=None, shape=None, shadow=None):
    frame = QFrame(parent)
    if shape:
        frame.setFrameShape(shape)
    if shadow:
        frame.setFrameShadow(shadow)
    return frame
