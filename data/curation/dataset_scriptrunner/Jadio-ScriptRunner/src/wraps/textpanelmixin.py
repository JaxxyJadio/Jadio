class CodeAigentTextPanelMixin:
    codeaigent_text_attr = "text_edit"  # Override in subclass if needed

    def codeaigent_get_text(self):
        text_widget = getattr(self, self.codeaigent_text_attr, None)
        if text_widget is not None:
            return text_widget.toPlainText()
        raise AttributeError(f"{self.__class__.__name__} missing attribute '{self.codeaigent_text_attr}'")

    def codeaigent_set_text(self, text):
        text_widget = getattr(self, self.codeaigent_text_attr, None)
        if text_widget is not None:
            text_widget.setPlainText(text)
        else:
            raise AttributeError(f"{self.__class__.__name__} missing attribute '{self.codeaigent_text_attr}'")
