# Button panel styles
from .mainstyle import PANEL_BG

BUTTON_PANEL_STYLE = f"""
QWidget#ButtonPanel {{
    background: {PANEL_BG};
    margin: 0;
}}
QGridLayout {{
    margin: 0;
    padding: 0;
}}
QPushButton {{
    min-width: 60px;
    max-width: 100px;
    min-height: 28px;
    font-size: 13px;
}}
QPushButton[configured="true"] {{
    background-color: #2e7d32;
    color: #fff;
    font-weight: bold;
}}
QPushButton[configured="false"] {{
    background-color: #424242;
    color: #bdbdbd;
}}
QPushButton:hover {{
    background: #31363b;
}}
QLabel#CliHeader {{
    font-weight: bold;
    font-size: 15px;
    padding: 8px 0;
    background: {PANEL_BG};
    color: #e0e0e0;
    margin: 8px 8px 12px 8px;
}}
QLabel#WarningLabel {{
    color: #FFA500;
    font-weight: bold;
}}
"""

CLI_HEADER_STYLE = """
font-weight: bold;
font-size: 15px;
padding: 8px 0;
background: #141516;
color: #e0e0e0;
margin: 8px 8px 12px 8px;
"""
