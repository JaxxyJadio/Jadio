import sys
import os
import subprocess
import signal
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from .styles.terminalstyles import TERMINAL_PANEL_STYLE, get_terminal_color
import shutil
from .shells.powershell import PowerShellRunner
from .shells.base_shell import BaseShellRunner
from html import escape

DEBUG_SHELL = False  # Set to True for verbose shell debug output

class CommandWorker(QThread):
    outputReceived = pyqtSignal(str, str)  # (text, stream)
    finished = pyqtSignal(int)  # exit code
    started = pyqtSignal()
    def __init__(self, command, parent=None):
        super().__init__(parent)
        self.command = command
        self.runner = None
    def run(self):
        self.started.emit()
        exit_code = -1
        if sys.platform.startswith('win'):
            self.runner = PowerShellRunner(self.command, debug=DEBUG_SHELL)
        elif sys.platform == "darwin":  # macOS
            shell_candidates = ["zsh", "bash", "sh"]
            self.runner = BaseShellRunner(self.command, shell_candidates, debug=DEBUG_SHELL)
        else:  # Linux and others
            shell_candidates = ["bash", "sh"]
            self.runner = BaseShellRunner(self.command, shell_candidates, debug=DEBUG_SHELL)
        if self.runner is None or (hasattr(self.runner, "shell_path") and self.runner.shell_path is None):
            self.outputReceived.emit("No suitable shell found on this system.", 'stderr')
            self.finished.emit(-1)
            return
        exit_code = self.runner.run(self.outputReceived.emit)
        self.finished.emit(exit_code)
    def stop(self):
        if self.runner:
            self.runner.stop()

class TerminalPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TerminalPanel")
        self.setStyleSheet(TERMINAL_PANEL_STYLE)
        self.workers = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QHBoxLayout()
        self.title_label = QLabel("Terminal")
        self.title_label.setObjectName("TerminalTitleLabel")
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("TerminalClearButton")
        self.clear_btn.clicked.connect(self.clear_terminal)
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(self.clear_btn)
        layout.addLayout(header)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setObjectName("TerminalTextEdit")
        layout.addWidget(self.text_edit)
    def append_output(self, text, stream):
        color = get_terminal_color(stream)
        html = f'<span style="color:{color}">{escape(text)}</span>'
        # Set style state for error/warning/info/success
        if stream == 'stderr':
            self.text_edit.setProperty('error', True)
            self.text_edit.setProperty('warning', False)
            self.text_edit.setProperty('info', False)
            self.text_edit.setProperty('success', False)
        elif stream == 'warning':
            self.text_edit.setProperty('error', False)
            self.text_edit.setProperty('warning', True)
            self.text_edit.setProperty('info', False)
            self.text_edit.setProperty('success', False)
        elif stream == 'info':
            self.text_edit.setProperty('error', False)
            self.text_edit.setProperty('warning', False)
            self.text_edit.setProperty('info', True)
            self.text_edit.setProperty('success', False)
        elif stream == 'success':
            self.text_edit.setProperty('error', False)
            self.text_edit.setProperty('warning', False)
            self.text_edit.setProperty('info', False)
            self.text_edit.setProperty('success', True)
        else:
            self.text_edit.setProperty('error', False)
            self.text_edit.setProperty('warning', False)
            self.text_edit.setProperty('info', False)
            self.text_edit.setProperty('success', False)
        # Force style refresh
        self.text_edit.setStyleSheet(self.text_edit.styleSheet())
        self.text_edit.append(html)
    def run_command(self, command):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.append_output(f"{timestamp} $ {command}", 'stdout')
        worker = CommandWorker(command)
        worker.outputReceived.connect(self.append_output)
        worker.finished.connect(lambda code, cmd=command: self.on_command_finished(code, cmd))
        self.workers.append(worker)
        worker.start()
    def on_command_finished(self, exit_code, command):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.append_output(f"{timestamp} [exit code: {exit_code}]", 'stdout' if exit_code == 0 else 'stderr')
        self.workers = [w for w in self.workers if w.isRunning()]
    def stop_all_scripts(self):
        for worker in self.workers:
            worker.stop()
        self.workers = [w for w in self.workers if w.isRunning()]
        self.append_output("Stopped all running scripts.", 'stderr')
    def clear_terminal(self):
        self.text_edit.clear()

# Note: On Cygwin/MSYS bash on Windows, process signals may not work as expected. This is a known limitation.
