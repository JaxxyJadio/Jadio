class CodeAigentTabPanelMixin:
    def codeaigent_add_tab(self, tab_widget, widget, title):
        tab_widget.addTab(widget, title)
    def codeaigent_set_tab_enabled(self, tab_widget, index, enabled):
        tab_widget.setTabEnabled(index, enabled)
