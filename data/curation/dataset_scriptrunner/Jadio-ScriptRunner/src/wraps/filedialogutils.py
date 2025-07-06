from PyQt6.QtWidgets import QFileDialog, QColorDialog, QFontDialog

def codeaigent_get_open_filename(parent=None, caption='', directory='', filter=''):
    return QFileDialog.getOpenFileName(parent, caption, directory, filter)

def codeaigent_get_save_filename(parent=None, caption='', directory='', filter=''):
    return QFileDialog.getSaveFileName(parent, caption, directory, filter)

def codeaigent_get_existing_directory(parent=None, caption='', directory=''):
    return QFileDialog.getExistingDirectory(parent, caption, directory)

def codeaigent_get_color(parent=None, initial=None):
    if initial is not None:
        return QColorDialog.getColor(initial, parent)
    else:
        return QColorDialog.getColor(parent=parent)

def codeaigent_get_font(parent=None, initial=None):
    if initial is not None:
        return QFontDialog.getFont(initial, parent)
    else:
        return QFontDialog.getFont(parent=parent)
