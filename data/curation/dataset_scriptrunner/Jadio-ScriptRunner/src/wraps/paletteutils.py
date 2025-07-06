from PyQt6.QtGui import QPalette

def codeaigent_create_palette():
    return QPalette()

def codeaigent_set_palette_color(palette, role, color):
    palette.setColor(role, color)
    return palette
