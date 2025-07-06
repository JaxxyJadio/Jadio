import sys
import subprocess
import signal

class PowerShellRunner:
    def __init__(self, command, debug=False):
        self.command = command
        self.process = None
        self._stopping = False
        self.debug = debug
        self.shell_path = "powershell"  # Added for compatibility

    def run(self, output_callback):
        shell_cmd = ["powershell", "-NoProfile", "-Command", self.command]
        if self.debug:
            output_callback(f"Launching shell: {shell_cmd}", 'stdout')
        output_callback("Using shell: powershell", 'stdout')
        try:
            self.process = subprocess.Popen(
                shell_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform.startswith('win') else 0
            )
            if self.process.stdout:
                for line in self.process.stdout:
                    if self._stopping:
                        break
                    output_callback(line.rstrip(), 'stdout')
            if self.process.stderr:
                for line in self.process.stderr:
                    if self._stopping:
                        break
                    output_callback(line.rstrip(), 'stderr')
            self.process.wait()
            return self.process.returncode
        except Exception as e:
            output_callback(str(e), 'stderr')
            return -1

    def stop(self):
        self._stopping = True
        if self.process:
            try:
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception:
                pass
