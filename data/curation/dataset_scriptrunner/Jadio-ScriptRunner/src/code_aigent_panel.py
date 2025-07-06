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
from .styles.aigentstyle import CODEAIGENT_PANEL_STYLE, CODEAIGENT_TAB_STYLE, CODEAIGENT_BOTTOM_TAB_STYLE
from .wraps.panelbase import PanelBase

class CodeAIgentPanel(PanelBase):
    def __init__(self, parent=None):
        super().__init__(
            object_name="CodeAIgentPanel",
            style_sheet=CODEAIGENT_PANEL_STYLE + CODEAIGENT_TAB_STYLE + CODEAIGENT_BOTTOM_TAB_STYLE,
            header="Code.AIgent",
            parent=parent
        )
        self.text_edit = QTextEdit()
        self.text_edit.setObjectName("CodeAIgentTextEdit")
        self._layout.addWidget(self.text_edit)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("CodeAIgentInputTabs")
        self._layout.addWidget(self.tabs)
        self.prompt_entry = QLineEdit()
        self.prompt_entry.setObjectName("CodeAIgentPromptEntry")
        self.ask_button = QPushButton("Ask")
        self.ask_button.setObjectName("CodeAIgentAskButton")
        self.tools_button = QPushButton("Tools")
        self.tools_button.setObjectName("CodeAIgentToolsButton")
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("CodeAIgentModelCombo")
        self.agent_ask_toggle = QPushButton("Agent Ask")
        self.agent_ask_toggle.setObjectName("CodeAIgentAgentAskToggle")
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setObjectName("CodeAIgentBottomTabs")
        self._layout.addWidget(self.bottom_tabs)
        # Layout setup for tabs and controls can be further modularized
