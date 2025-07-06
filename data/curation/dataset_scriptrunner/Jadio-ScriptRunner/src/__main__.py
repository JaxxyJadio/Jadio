import sys
from .wraps.qtwidgetsutils import (
    codeaigent_create_qapplication, codeaigent_create_qmainwindow, codeaigent_create_qvboxlayout,
    codeaigent_create_qwidget, codeaigent_create_qmenubar
)
from .wraps.qtcoreutils import codeaigent_create_qtimer
from .wraps.qtguiutils import codeaigent_create_qkeysequence, codeaigent_create_qcloseevent
from .wraps.menutools import codeaigent_add_menu, codeaigent_add_action, codeaigent_add_separator
from .wraps.splittertools import codeaigent_make_splitter
from .wraps.statustools import codeaigent_make_status_bar
from .wraps.timertools import codeaigent_make_timer
from .wraps.dialogtools import codeaigent_show_question
from .styles import ALL_STYLES
from .cli_shortcut_panel import ButtonPanel
from .terminalpanel import TerminalPanel
from .dialogs import about_dialog, shortcuts_dialog
from .editorpanel import EditorPanel
from .explorerpanel import ExplorerPanel
from .code_aigent_panel import CodeAIgentPanel
from PyQt6.QtCore import Qt

class CodeAIGent:
    def __init__(self):
        self.window = codeaigent_create_qmainwindow()
        self.window.setWindowTitle("Code-AIgent")
        self.window.setGeometry(100, 100, 1600, 900)
        self.window.setStyleSheet(ALL_STYLES)
        central_widget = codeaigent_create_qwidget()
        self.window.setCentralWidget(central_widget)
        layout = codeaigent_create_qvboxlayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # Instantiate panels and keep references
        self.explorer_panel = ExplorerPanel()
        self.button_panel = ButtonPanel()
        left_splitter = codeaigent_make_splitter(
            Qt.Orientation.Vertical, [self.explorer_panel, self.button_panel], [400, 200]
        )
        self.editor_panel = EditorPanel()
        self.codeaigent_panel = CodeAIgentPanel()
        main_splitter = codeaigent_make_splitter(
            Qt.Orientation.Horizontal, [left_splitter, self.editor_panel, self.codeaigent_panel], [150, 900, 400]
        )
        self.terminal_panel = TerminalPanel()
        vertical_splitter = codeaigent_make_splitter(
            Qt.Orientation.Vertical, [main_splitter, self.terminal_panel], [700, 200]
        )
        layout.addWidget(vertical_splitter)
        # Connect button panel to terminal
        self.button_panel.executeCommand.connect(self.terminal_panel.run_command)
        self.create_menu_bar()
        self.status_bar = codeaigent_make_status_bar(self.window)
        self.setup_timers()
    def create_menu_bar(self):
        menubar = self.window.menuBar()
        file_menu = codeaigent_add_menu(menubar, "File")
        codeaigent_add_action(file_menu, "Save Configuration", codeaigent_create_qkeysequence('Ctrl+S'), self.save_config)
        codeaigent_add_separator(file_menu)
        codeaigent_add_action(file_menu, "Exit", codeaigent_create_qkeysequence('Ctrl+Q'), self.window.close)
        edit_menu = codeaigent_add_menu(menubar, "Edit")
        codeaigent_add_action(edit_menu, "Clear Terminal", codeaigent_create_qkeysequence('Ctrl+L'), self.clear_terminal)
        tools_menu = codeaigent_add_menu(menubar, "Tools")
        codeaigent_add_action(tools_menu, "Stop All Scripts", codeaigent_create_qkeysequence('Ctrl+Shift+C'), self.stop_all_scripts)
        help_menu = codeaigent_add_menu(menubar, "Help")
        codeaigent_add_action(help_menu, "About", None, self.about)
        codeaigent_add_action(help_menu, "Keyboard Shortcuts", None, self.shortcuts)
    def setup_timers(self):
        self.auto_save_timer = codeaigent_make_timer(30000, self.auto_save)
        self.status_timer = codeaigent_make_timer(1000, self.update_status)
    def save_config(self):
        self.button_panel.save_config()
        self.status_bar.showMessage("Configuration saved", 2000)
    def auto_save(self):
        self.button_panel.save_config()
    def clear_terminal(self):
        self.terminal_panel.clear_terminal()
    def stop_all_scripts(self):
        self.terminal_panel.stop_all_scripts()
    def update_status(self):
        configured = self.button_panel.get_configured_count()
        running = len([w for w in getattr(self.terminal_panel, 'workers', []) if w.isRunning()])
        unsaved = self.button_panel.get_unsaved_status()
        status = f"Configured Buttons: {configured} | Running Scripts: {running}"
        if unsaved:
            status += " • Unsaved changes"
        self.status_bar.showMessage(status)
    def about(self):
        about_dialog.show_about(self.window)
    def shortcuts(self):
        shortcuts_dialog.show_shortcuts(self.window)
    def closeEvent(self, a0):
        running = len([w for w in getattr(self.terminal_panel, 'workers', []) if w.isRunning()])
        unsaved = self.button_panel.get_unsaved_status()
        if running > 0:
            reply = codeaigent_show_question(self.window, "Scripts Running", "There are running scripts. Stop all and exit?")
            if reply == 65536:
                if a0:
                    a0.ignore()
                return
            self.terminal_panel.stop_all_scripts()
        if unsaved:
            reply = codeaigent_show_question(self.window, "Unsaved Changes", "You have unsaved changes. Exit without saving?")
            if reply == 65536:
                if a0:
                    a0.ignore()
                return
        if a0:
            a0.accept()

def main():
    app = codeaigent_create_qapplication(sys.argv)
    app.setApplicationName("Jadio-ScriptRunner")
    codeaigent = CodeAIGent()
    codeaigent.window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
