from PyQt6.QtGui import QMovie

def codeaigent_create_movie(filename=None, parent=None):
    movie = QMovie(filename, parent=parent) if filename else QMovie(parent=parent)
    return movie
