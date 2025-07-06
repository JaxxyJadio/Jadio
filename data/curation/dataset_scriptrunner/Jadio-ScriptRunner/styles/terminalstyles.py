# Terminal panel styles
from .mainstyle import PANEL_BG, MAIN_BG

TERMINAL_COLORS = {
    'stdout': '#e0e0e0',
    'stderr': '#ff5252',
}

def get_terminal_color(stream):
    return TERMINAL_COLORS.get(stream, "#FFFFFF")

TERMINAL_PANEL_STYLE = f"""
QWidget#TerminalPanel {{
    background: {PANEL_BG};
    color: #e0e0e0;
    border-radius: 10px;
    box-shadow: 0 2px 16px 0 rgba(0,0,0,0.25);
    margin: 0;
    padding: 0;
}}
QHBoxLayout, QVBoxLayout {{
    margin: 0;
    padding: 0;
}}
QLabel#TerminalTitleLabel {{
    color: #90caf9;
    font-size: 17px;
    font-weight: bold;
    letter-spacing: 1px;
    padding: 8px 0 8px 12px;
    background: transparent;
}}
QPushButton#TerminalClearButton {{
    background: #232629;
    color: #fff;
    font-size: 13px;
    border-radius: 6px;
    padding: 4px 18px;
    margin: 6px 12px 6px 0;
    font-weight: bold;
}}
QPushButton#TerminalClearButton:hover {{
    background: #1565c0;
    color: #fff;
}}
QTextEdit#TerminalTextEdit {{
    background: {PANEL_BG};
    color: #e0e0e0;
    font-family: 'Fira Mono', 'Consolas', 'Courier New', monospace;
    font-size: 14px;
    border: none;
    border-radius: 8px;
    padding: 12px 16px;
    line-height: 1.6;
    outline: none;
    min-height: 180px;
    max-height: 400px;
}}
QTextEdit#TerminalTextEdit[error="true"] {{
    background: #ff525222;
    color: #ff5252;
    border: 1.5px solid #ff5252;
}}
QTextEdit#TerminalTextEdit[warning="true"] {{
    background: #ffa72622;
    color: #ffa726;
    border: 1.5px solid #ffa726;
}}
QTextEdit#TerminalTextEdit[info="true"] {{
    background: #90caf922;
    color: #90caf9;
    border: 1.5px solid #90caf9;
}}
QTextEdit#TerminalTextEdit[success="true"] {{
    background: #2e7d3222;
    color: #2e7d32;
    border: 1.5px solid #2e7d32;
}}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: {PANEL_BG};
    border: none;
    border-radius: 6px;
    width: 12px;
    height: 12px;
    margin: 0px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: #31363b;
    min-height: 24px;
    min-width: 24px;
    border-radius: 6px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    height: 0px;
    width: 0px;
}}
QLabel {{
    color: #e0e0e0;
    font-weight: bold;
}}
QPushButton {{
    background: {MAIN_BG};
    color: #fff;
    font-size: 13px;
    padding: 4px 12px;
    border-radius: 6px;
    margin: 2px 4px;
}}
QPushButton:hover {{
    background: #1565c0;
    color: #fff;
}}
QPushButton:pressed {{
    background: #1976d2;
    color: #fff;
}}
"""