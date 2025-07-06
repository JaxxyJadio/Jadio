from PyQt6.QtWidgets import QLCDNumber

def codeaigent_create_lcdnumber(parent=None, digit_count=5, mode=None):
    lcd = QLCDNumber(parent)
    lcd.setDigitCount(digit_count)
    if mode:
        lcd.setMode(mode)
    return lcd
