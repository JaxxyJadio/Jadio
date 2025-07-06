from PyQt6.QtWidgets import QTextBrowser

def codeaigent_make_text_browser(parent=None, object_name=None, **kwargs):
    browser = QTextBrowser(parent)
    if object_name:
        browser.setObjectName(object_name)
    for k, v in kwargs.items():
        setattr(browser, k, v)
    return browser
