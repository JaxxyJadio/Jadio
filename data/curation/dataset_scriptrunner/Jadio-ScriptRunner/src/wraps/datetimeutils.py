from PyQt6.QtWidgets import QDateEdit, QTimeEdit, QDateTimeEdit

def codeaigent_create_dateedit(parent=None, date=None):
    widget = QDateEdit(parent)
    if date:
        widget.setDate(date)
    return widget

def codeaigent_create_timeedit(parent=None, time=None):
    widget = QTimeEdit(parent)
    if time:
        widget.setTime(time)
    return widget

def codeaigent_create_datetimeedit(parent=None, datetime=None):
    widget = QDateTimeEdit(parent)
    if datetime:
        widget.setDateTime(datetime)
    return widget
