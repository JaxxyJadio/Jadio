from PyQt6.QtCore import QMimeData

def codeaigent_create_mimedata():
    return QMimeData()

def codeaigent_set_mimedata_text(mimedata, text):
    mimedata.setText(text)
    return mimedata

def codeaigent_get_mimedata_text(mimedata):
    return mimedata.text()
