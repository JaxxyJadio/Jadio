class CodeAigentListPanelMixin:
    def codeaigent_refresh_list(self, list_widget, items):
        list_widget.clear()
        for item in items:
            list_widget.addItem(item)
