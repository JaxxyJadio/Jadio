from PyQt6.QtWidgets import QSplashScreen
from PyQt6.QtGui import QPixmap

def codeaigent_create_splashscreen(pixmap=None, flags=None):
    splash = QSplashScreen(pixmap or QPixmap(), flags) if flags else QSplashScreen(pixmap or QPixmap())
    return splash
