# Editor panel styles (advanced, Monaco-inspired)
from .mainstyle import PANEL_BG, MAIN_BG, ACCENT

EDITOR_BG = PANEL_BG
EDITOR_FG = "#e0e0e0"
EDITOR_LINE_NUMBER = "#5c6370"
EDITOR_LINE_NUMBER_ACTIVE = ACCENT
EDITOR_SELECTION_BG = "#264f78"
EDITOR_CURSOR = ACCENT
EDITOR_MINIMAP_BG = "#181a1b"
EDITOR_GUTTER_BG = "#181a1b"
EDITOR_GUTTER_BORDER = "#232629"
EDITOR_HIGHLIGHT = "#31363b"
EDITOR_ERROR = "#ff5252"
EDITOR_WARNING = "#ffa726"
EDITOR_INFO = "#90caf9"
EDITOR_MATCH = "#1976d2"

EDITOR_STYLE = f"""
QWidget#EditorPanel {{
    background: {EDITOR_BG};
    color: {EDITOR_FG};
    font-family: 'Fira Mono', 'Consolas', 'Courier New', monospace;
    font-size: 15px;
    border: none;
    border-radius: 8px;
    box-shadow: 0 2px 16px 0 rgba(0,0,0,0.25);
}}
QPlainTextEdit#EditorTextEdit {{
    background: {EDITOR_BG};
    color: {EDITOR_FG};
    selection-background-color: {EDITOR_SELECTION_BG};
    selection-color: #fff;
    caret-color: {EDITOR_CURSOR};
    font-family: 'Fira Mono', 'Consolas', 'Courier New', monospace;
    font-size: 15px;
    border: none;
    padding: 12px 16px 12px 48px;
    line-height: 1.6;
    outline: none;
}}
QWidget#EditorGutter {{
    background: {EDITOR_GUTTER_BG};
    color: {EDITOR_LINE_NUMBER};
    border-right: 2px solid {EDITOR_GUTTER_BORDER};
    font-size: 13px;
    min-width: 36px;
    padding: 0 8px;
}}
QWidget#EditorGutter[lineNumberActive="true"] {{
    color: {EDITOR_LINE_NUMBER_ACTIVE};
    font-weight: bold;
}}
QWidget#EditorMinimap {{
    background: {EDITOR_MINIMAP_BG};
    border-left: 1px solid {EDITOR_GUTTER_BORDER};
    min-width: 60px;
    max-width: 80px;
    opacity: 0.85;
}}
QWidget#EditorPanel[error="true"] QPlainTextEdit#EditorTextEdit {{
    background: {EDITOR_ERROR}22;
    border: 1.5px solid {EDITOR_ERROR};
}}
QWidget#EditorPanel[warning="true"] QPlainTextEdit#EditorTextEdit {{
    background: {EDITOR_WARNING}22;
    border: 1.5px solid {EDITOR_WARNING};
}}
QWidget#EditorPanel[info="true"] QPlainTextEdit#EditorTextEdit {{
    background: {EDITOR_INFO}22;
    border: 1.5px solid {EDITOR_INFO};
}}
QWidget#EditorPanel[match="true"] QPlainTextEdit#EditorTextEdit {{
    background: {EDITOR_MATCH}22;
    border: 1.5px solid {EDITOR_MATCH};
}}
QWidget#EditorPanel QScrollBar:vertical {{
    background: {EDITOR_BG};
    width: 12px;
    margin: 0px 0px 0px 0px;
    border: none;
    border-radius: 6px;
}}
QWidget#EditorPanel QScrollBar::handle:vertical {{
    background: {EDITOR_HIGHLIGHT};
    min-height: 24px;
    border-radius: 6px;
}}
QWidget#EditorPanel QScrollBar::add-line:vertical,
QWidget#EditorPanel QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QWidget#EditorPanel QScrollBar:horizontal {{
    background: {EDITOR_BG};
    height: 12px;
    margin: 0px 0px 0px 0px;
    border: none;
    border-radius: 6px;
}}
QWidget#EditorPanel QScrollBar::handle:horizontal {{
    background: {EDITOR_HIGHLIGHT};
    min-width: 24px;
    border-radius: 6px;
}}
QWidget#EditorPanel QScrollBar::add-line:horizontal,
QWidget#EditorPanel QScrollBar::sub-line:horizontal {{
    width: 0px;
}}
"""

# Syntax highlight colors (example, can be expanded)
EDITOR_SYNTAX = {
    'keyword': '#82aaff',
    'string': '#c3e88d',
    'comment': '#5c6370',
    'function': '#82aaff',
    'number': '#f78c6c',
    'type': '#ffcb6b',
    'variable': '#e0e0e0',
    'operator': '#89ddff',
    'builtin': '#ff5370',
    'decorator': '#c792ea',
    'class': '#ffcb6b',
    'constant': '#f78c6c',
    'error': EDITOR_ERROR,
    'warning': EDITOR_WARNING,
    'info': EDITOR_INFO,
    'match': EDITOR_MATCH,
}
