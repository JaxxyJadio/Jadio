import subprocess
import shutil
import os
import signal

class BaseShellRunner:
    def __init__(self, command, shell_candidates, debug=False, log_to_file=False, log_path=None):
        self.command = command
        self.process = None
        self._stopping = False
        self.shell_path = self._find_shell(shell_candidates)
        self.debug = debug
        self.log_to_file = log_to_file
        self.log_path = log_path
        self._buffer = []
        self._batch_size = 20

    def _find_shell(self, candidates):
        for shell in candidates:
            if shutil.which(shell):
                return shell
        return None

    def run(self, output_callback):
        if not self.shell_path:
            output_callback("No suitable shell found on this system.", 'stderr')
            return -1
        shell_cmd = [self.shell_path, "-c", self.command]
        if self.debug:
            output_callback(f"Launching shell: {shell_cmd}", 'stdout')
        output_callback(f"Using shell: {self.shell_path}", 'stdout')
        log_file = open(self.log_path, 'a', encoding='utf-8') if self.log_to_file and self.log_path else None
        try:
            self.process = subprocess.Popen(
                shell_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid if os.name == 'posix' else None
            )
            # Buffered reading for stdout
            stdout = self.process.stdout
            if stdout is not None:
                for chunk in iter(lambda: stdout.read(4096), ''):
                    if self._stopping or not chunk:
                        break
                    self._buffer.extend(chunk.splitlines())
                    if len(self._buffer) >= self._batch_size:
                        batch = "\n".join(self._buffer)
                        output_callback(batch, 'stdout')
                        if log_file:
                            log_file.write(batch + '\n')
                        self._buffer.clear()
                # Flush any remaining lines
                if self._buffer:
                    batch = "\n".join(self._buffer)
                    output_callback(batch, 'stdout')
                    if log_file:
                        log_file.write(batch + '\n')
                    self._buffer.clear()
            # Buffered reading for stderr
            stderr = self.process.stderr
            if stderr is not None:
                err_buffer = []
                for chunk in iter(lambda: stderr.read(4096), ''):
                    if self._stopping or not chunk:
                        break
                    err_buffer.extend(chunk.splitlines())
                    if len(err_buffer) >= self._batch_size:
                        batch = "\n".join(err_buffer)
                        output_callback(batch, 'stderr')
                        if log_file:
                            log_file.write(batch + '\n')
                        err_buffer.clear()
                if err_buffer:
                    batch = "\n".join(err_buffer)
                    output_callback(batch, 'stderr')
                    if log_file:
                        log_file.write(batch + '\n')
            self.process.wait()
            if log_file:
                log_file.close()
            return self.process.returncode
        except Exception as e:
            output_callback(str(e), 'stderr')
            if log_file:
                log_file.write(str(e) + '\n')
                log_file.close()
            return -1

    def stop(self):
        self._stopping = True
        if self.process:
            try:
                if os.name == 'posix':
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()
            except Exception:
                pass
