# Code.AIgent panel styles
from .mainstyle import PANEL_BG, MAIN_BG, ACCENT

CODEAIGENT_PANEL_STYLE = f"""
QWidget#CodeAIgentPanel {{
    background: {PANEL_BG};
    margin: 0;
}}
QLabel#CodeAIgentHeader {{
    font-weight: bold;
    font-size: 18px;
    color: {ACCENT};
    padding: 12px 0 8px 0;
    background: transparent;
    margin-bottom: 8px;
    letter-spacing: 1px;
}}
QTextEdit#CodeAIgentTextEdit {{
    background: {PANEL_BG};
    color: #e0e0e0;
    font-family: 'Fira Mono', 'Consolas', 'Courier New', monospace;
    font-size: 14px;
    padding: 10px;
    min-height: 120px;
}}
QPushButton#CodeAIgentAskButton {{
    background: #1976d2;
    color: #fff;
    font-size: 13px;
    font-weight: bold;
    padding: 4px 12px;
    min-width: 30px;
    max-width: 40px;
    margin-right: 10px;
}}
QPushButton#CodeAIgentAskButton:hover {{
    background: #42a5f5;
    color: #232629;
}}
"""

CODEAIGENT_BOTTOM_BAR_STYLE = """
#CodeAIgentBottomBar {
    background: #232629;
    padding: 12px 12px 12px 12px;
    margin-top: 10px;
}
QComboBox#CodeAIgentModelCombo {
    min-width: 90px;
    max-width: 140px;
    background: #141516;
    color: #e0e0e0;
    padding: 4px 8px;
    font-size: 13px;
}
QComboBox#CodeAIgentModelCombo:focus {
    background: #232629;
}
QPushButton#CodeAIgentToolsButton, QPushButton#CodeAIgentAgentAskToggle {
    background: #232629;
    color: #e0e0e0;
    font-size: 13px;
    padding: 6px 14px;
}
QPushButton#CodeAIgentToolsButton:hover, QPushButton#CodeAIgentAgentAskToggle:hover {
    background: #31363b;
    color: #90caf9;
}
QPushButton#CodeAIgentAgentAskToggle:checked {
    background: #1976d2;
    color: #fff;
}
QLineEdit#CodeAIgentPromptEntry {
    background: #141516;
    color: #e0e0e0;
    font-size: 15px;
    padding: 8px 12px;
    min-width: 180px;
}
QLineEdit#CodeAIgentPromptEntry:focus {
    background: #232629;
}
QPushButton#CodeAIgentAskButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1976d2, stop:1 #1565c0);
    color: #fff;
    font-size: 16px;
    font-weight: bold;
    padding: 10px 28px;
    margin-left: 8px;
    min-width: 120px;
}
QPushButton#CodeAIgentAskButton:hover {
    background: #42a5f5;
    color: #232629;
}
"""

CODEAIGENT_TAB_STYLE = f"""
QTabWidget#CodeAIgentInputTabs::pane {{
    background: {MAIN_BG};
    margin-top: 6px;
}}
QTabBar::tab {{
    background: {MAIN_BG};
    color: #bdbdbd;
    min-width: 28px;
    min-height: 24px;
    margin-right: 1px;
    padding: 2px 6px;
    font-size: 12px;
    margin-top: 1px;
}}
QTabBar::tab:selected {{
    background: {PANEL_BG};
    color: {ACCENT};
    font-weight: bold;
}}
QTabBar::tab:hover {{
    background: #31363b;
    color: #e0e0e0;
}}
QTabWidget QTabBar {{
    margin: 0;
    padding: 0;
}}
QTabBar, QTabWidget QTabBar {{
    qproperty-drawBase: 0;
    min-height: 0;
}}
QTabBar::scroller, QTabBar QToolButton {{
    width: 0px;
    height: 0px;
    background: transparent;
}}
QTabWidget::tab-bar {{
    left: 0px;
}}
QTabBar {{
    qproperty-elideMode: ElideRight;
}}
"""

CODEAIGENT_BOTTOM_TAB_STYLE = f"""
QTabWidget#CodeAIgentBottomTabs::pane {{
    background: {MAIN_BG};
    margin-bottom: 4px;
    min-height: 48px;
    max-height: 90px;
}}
QTabBar::tab {{
    min-height: 22px;
    padding: 2px 6px;
    font-size: 11px;
}}
QLabel {{
    background: {PANEL_BG};
    color: #e0e0e0;
    padding: 4px 8px;
    font-size: 13px;
    margin: 4px 0 4px 0;
}}
"""
