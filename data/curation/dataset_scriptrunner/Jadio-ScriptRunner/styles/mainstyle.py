# Main application-wide styles

MAIN_BG = "#232629"
PANEL_BG = "#141516"  # Even darker for floating effect
ACCENT = "#90caf9"

ALL_STYLES = f"""
QWidget {{
    background-color: {MAIN_BG};
    color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}}
QMenuBar, QMenu {{
    background: {MAIN_BG};
    color: #e0e0e0;
}}
QMenu::item:selected {{
    background: #1565c0;
}}
"""
