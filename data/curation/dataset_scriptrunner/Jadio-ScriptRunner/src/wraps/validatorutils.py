from PyQt6.QtGui import QIntValidator, QDoubleValidator
from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import QRegularExpressionValidator

def codeaigent_create_int_validator(minimum=None, maximum=None, parent=None):
    validator = QIntValidator(parent)
    if minimum is not None:
        validator.setBottom(minimum)
    if maximum is not None:
        validator.setTop(maximum)
    return validator

def codeaigent_create_double_validator(minimum=None, maximum=None, decimals=None, parent=None):
    validator = QDoubleValidator(parent)
    if minimum is not None:
        validator.setBottom(minimum)
    if maximum is not None:
        validator.setTop(maximum)
    if decimals is not None:
        validator.setDecimals(decimals)
    return validator

def codeaigent_create_regex_validator(pattern, parent=None):
    regex = QRegularExpression(pattern)
    return QRegularExpressionValidator(regex, parent)
