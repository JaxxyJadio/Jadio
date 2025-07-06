from PyQt6.QtGui import QKeySequence, QCursor, QRegion, QBitmap

def codeaigent_create_keysequence(seq):
    return QKeySequence(seq)

def codeaigent_create_cursor(shape):
    return QCursor(shape)

def codeaigent_create_region(*args):
    return QRegion(*args)

def codeaigent_create_bitmap(width, height):
    return QBitmap(width, height)
