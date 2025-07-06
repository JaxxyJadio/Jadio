from wraps import (
    codeaigent_add_tab,
    codeaigent_copy_to_clipboard,
    codeaigent_create_checkbox,
    codeaigent_create_doublespinbox,
    codeaigent_create_frame,
    codeaigent_create_groupbox,
    codeaigent_create_listview,
    codeaigent_create_listwidget,
    codeaigent_create_progressbar,
    codeaigent_create_radiobutton,
    codeaigent_create_scrollarea,
    codeaigent_create_slider,
    codeaigent_create_spinbox,
    codeaigent_create_tableview,
    codeaigent_create_tablewidget,
    codeaigent_create_toolbar,
    codeaigent_create_toolbutton,
    codeaigent_create_treeview,
    codeaigent_create_treewidget,
    codeaigent_get_clipboard_text,
    codeaigent_get_color,
    codeaigent_get_existing_directory,
    codeaigent_get_font,
    codeaigent_get_open_filename,
    codeaigent_get_save_filename,
    codeaigent_grid,
    codeaigent_hbox,
    codeaigent_load_image,
    codeaigent_load_pixmap,
    codeaigent_make_button,
    codeaigent_make_combo_box,
    codeaigent_make_label,
    codeaigent_make_line_edit,
    codeaigent_make_panel_header,
    codeaigent_make_plain_text_edit,
    codeaigent_make_splitter,
    codeaigent_make_status_bar,
    codeaigent_make_tab_widget,
    codeaigent_make_text_browser,
    codeaigent_make_text_edit,
    codeaigent_make_timer,
    codeaigent_make_widget,
    codeaigent_show_question,
    codeaigent_vbox
)
    codeaigent_add_tab,
    codeaigent_copy_to_clipboard,
    codeaigent_create_checkbox,
    codeaigent_create_doublespinbox,
    codeaigent_create_frame,
    codeaigent_create_groupbox,
    codeaigent_create_listview,
    codeaigent_create_listwidget,
    codeaigent_create_progressbar,
    codeaigent_create_radiobutton,
    codeaigent_create_scrollarea,
    codeaigent_create_slider,
    codeaigent_create_spinbox,
    codeaigent_create_tableview,
    codeaigent_create_tablewidget,
    codeaigent_create_toolbar,
    codeaigent_create_toolbutton,
    codeaigent_create_treeview,
    codeaigent_create_treewidget,
    codeaigent_get_clipboard_text,
    codeaigent_get_color,
    codeaigent_get_existing_directory,
    codeaigent_get_font,
    codeaigent_get_open_filename,
    codeaigent_get_save_filename,
    codeaigent_grid,
    codeaigent_hbox,
    codeaigent_load_image,
    codeaigent_load_pixmap,
    codeaigent_make_button,
    codeaigent_make_combo_box,
    codeaigent_make_label,
    codeaigent_make_line_edit,
    codeaigent_make_panel_header,
    codeaigent_make_plain_text_edit,
    codeaigent_make_splitter,
    codeaigent_make_status_bar,
    codeaigent_make_tab_widget,
    codeaigent_make_text_browser,
    codeaigent_make_text_edit,
    codeaigent_make_timer,
    codeaigent_make_widget,
    codeaigent_show_question,
    codeaigent_vbox
)
import os
import json
    codeaigent_QWidget,
    codeaigent_QGridLayout,
    codeaigent_QPushButton,
    codeaigent_QVBoxLayout,
    codeaigent_QLineEdit,
    codeaigent_QLabel,
    codeaigent_QDialogButtonBox,
    codeaigent_QMessageBox,
    codeaigent_QGraphicsOpacityEffect,
    codeaigent_QHBoxLayout,
    codeaigent_QPropertyAnimation,
    codeaigent_QEasingCurve,
)
from .dialogs import button_config_dialog
from .styles.styles import TERMINAL_COLORS, BUTTON_PANEL_STYLE

CONFIG_FILE = "config/button_config.json"
GRID_ROWS = 6  # 6 rows x 3 columns = 18 buttons
GRID_COLS = 3

class AnimatedButton(codeaigent_QPushButton()):
    def __init__(self, row, col, parent=None):
        super().__init__(parent)
        self.row = row
        self.col = col
        self._configured = False
        self._default_text = f"{row*GRID_COLS+col+1}"
        self.setText(self._default_text)
        self.setProperty("configured", False)
        self.opacity_effect = codeaigent_QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.animation = codeaigent_QPropertyAnimation(self.opacity_effect, b"opacity")
        self._opacity = 1.0
    def setConfigured(self, value):
        self._configured = value
        self.setProperty("configured", value)
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)
    def isConfigured(self):
        return self._configured
    def setOpacity(self, value):
        self._opacity = value
        self.opacity_effect.setOpacity(value)
    def getOpacity(self):
        return self._opacity
    opacity = property(getOpacity, setOpacity)
    def animateClick(self):
        self.animation.stop()
        self.animation.setDuration(200)
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.5)
        self.animation.setEasingCurve(codeaigent_QEasingCurve.Type.InOutQuad)
        try:
            self.animation.finished.disconnect()
        except Exception:
            pass
        self.animation.finished.connect(self.restoreOpacity)
        self.animation.start()
    def restoreOpacity(self):
        self.animation.stop()
        self.animation.setDuration(200)
        self.animation.setStartValue(0.5)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(codeaigent_QEasingCurve.Type.InOutQuad)
        try:
            self.animation.finished.disconnect()
        except Exception:
            pass
        self.animation.start()

class ButtonPanel(codeaigent_QWidget()):
    executeCommand = codeaigent_create_pyqtsignal(str, str)  # name, command
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ButtonPanel")
        self.setStyleSheet(BUTTON_PANEL_STYLE)
        self.has_unsaved_changes = False
        self.grid = codeaigent_QGridLayout()
        self.grid.setSpacing(2)
        self.buttons = []
        self.configs = [[{"name": "", "command": ""} for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        # Add CLI Shortcuts header
        main_layout = codeaigent_QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        header = codeaigent_QLabel('CLI Shortcuts')
        header.setObjectName("CliHeader")
        header.setAlignment(0x0004)  # Qt.AlignmentFlag.AlignCenter
        main_layout.addWidget(header)
        for row in range(GRID_ROWS):
            row_buttons = []
            for col in range(GRID_COLS):
                btn = AnimatedButton(row, col, self)
                btn.setFixedWidth(80)
                btn.clicked.connect(lambda checked, r=row, c=col: self.handle_left_click(r, c))
                btn.setContextMenuPolicy(3)  # Qt.ContextMenuPolicy.CustomContextMenu
                btn.customContextMenuRequested.connect(lambda pos, r=row, c=col: self.handle_right_click(r, c))
                self.grid.addWidget(btn, row, col)
                row_buttons.append(btn)
            self.buttons.append(row_buttons)
        main_layout.addLayout(self.grid)
        self.warning_label = codeaigent_QLabel('⚠️ Unsaved changes')
        self.warning_label.setObjectName("WarningLabel")
        self.warning_label.setVisible(False)
        main_layout.addWidget(self.warning_label)
        self.load_config()
    def set_unsaved(self, value: bool):
        self.has_unsaved_changes = value
        self.warning_label.setVisible(value)
    def get_unsaved_status(self):
        return self.has_unsaved_changes
    def handle_left_click(self, row, col):
        config = self.configs[row][col]
        if config["command"]:
            self.buttons[row][col].animateClick()
            self.executeCommand.emit(config["name"], config["command"])
        else:
            codeaigent_QMessageBox.information(self, "Not Configured", "This button is not configured.")
    def handle_right_click(self, row, col):
        config = self.configs[row][col]
        name, command = button_config_dialog.show_button_config_dialog(config["name"], config["command"], self)
        if name is not None and command is not None:
            self.configs[row][col] = {"name": name, "command": command}
            btn = self.buttons[row][col]
            btn.setText(name if name else btn._default_text)
            btn.setConfigured(bool(name and command))
            self.set_unsaved(True)
            self.save_config()  # Save immediately after change
    def save_config(self):
        data = [[self.configs[r][c] for c in range(GRID_COLS)] for r in range(GRID_ROWS)]
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.set_unsaved(False)
    def duplicate_config(self, src_row, src_col, dst_row, dst_col):
        if 0 <= src_row < GRID_ROWS and 0 <= src_col < GRID_COLS and 0 <= dst_row < GRID_ROWS and 0 <= dst_col < GRID_COLS:
            self.configs[dst_row][dst_col] = self.configs[src_row][src_col].copy()
            conf = self.configs[dst_row][dst_col]
            btn = self.buttons[dst_row][dst_col]
            btn.setText(conf["name"] if conf["name"] else btn._default_text)
            btn.setConfigured(bool(conf["name"] and conf["command"]))
            self.set_unsaved(True)
            self.save_config()
    def save_config_to_file(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.configs, f, indent=2)
        self.set_unsaved(False)
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in range(min(GRID_ROWS, len(data))):
                for c in range(min(GRID_COLS, len(data[r]))):
                    conf = data[r][c]
                    self.configs[r][c] = conf
                    btn = self.buttons[r][c]
                    btn.setText(conf["name"] if conf["name"] else btn._default_text)
                    btn.setConfigured(bool(conf["name"] and conf["command"]))
        self.set_unsaved(False)
    def get_configured_count(self):
        return sum(1 for row in self.configs for conf in row if conf["name"] and conf["command"])
    def load_config_from_file(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in range(min(GRID_ROWS, len(data))):
                for c in range(min(GRID_COLS, len(data[r]))):
                    conf = data[r][c]
                    self.configs[r][c] = conf
                    btn = self.buttons[r][c]
                    btn.setText(conf["name"] if conf["name"] else btn._default_text)
                    btn.setConfigured(bool(conf["name"] and conf["command"]))
            self.save_config()