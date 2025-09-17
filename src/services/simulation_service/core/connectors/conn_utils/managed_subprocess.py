import subprocess
import threading
from typing import Callable, Mapping, Optional


class ManagedSubprocess:
    def __init__(
        self,
        command_args: list[str],
        stream_reader_func: Callable,
        logger_info_func: Callable,
        logger_error_func: Callable,
        text: bool = True,
        env: Optional[Mapping[str, str]] = None,
    ):
        self.command_args = command_args
        self.text = text
        self.stream_reader_func = stream_reader_func
        self.logger_info_func = logger_info_func
        self.logger_error_func = logger_error_func
        self.logger_warning_func = logger_error_func
        self.env = dict(env) if env is not None else None

        self.process: subprocess.Popen | None = None
        self.stdout_thread: threading.Thread | None = None
        self.stderr_thread: threading.Thread | None = None
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []

    def __enter__(self) -> subprocess.Popen:
        try:
            command = " ".join(self.command_args)
        except Exception:
            command = str(self.command_args)
        cwd = self.env.get("PWD") if self.env else None
        if cwd:
            self.logger_info_func(f"Starting subprocess: {command} (cwd={cwd})")
        else:
            self.logger_info_func(f"Starting subprocess: {command}")
        try:
            popen_kwargs: dict = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": self.text,
                "cwd": cwd,
            }

            # Pass env only if explicitly provided; do not include when None.
            if self.env is not None:
                popen_kwargs["env"] = self.env

            self.process = subprocess.Popen(
                self.command_args,
                **popen_kwargs,
            )
        except Exception:
            self.logger_error_func(
                f"Failed to start subprocess with command: {command}"
            )
            self.process = None
            raise  # Re-raise the exception from Popen

        if self.process.stdout is None or self.process.stderr is None:
            if self.process:
                self.process.kill()
                self.process.wait()
            raise RuntimeError("Subprocess stdout/stderr streams are not available.")

        self.stdout_thread = threading.Thread(
            target=self.stream_reader_func,
            args=(self.process.stdout, self.stdout_lines, self.logger_info_func),
            daemon=True,
        )
        self.stderr_thread = threading.Thread(
            target=self.stream_reader_func,
            args=(self.process.stderr, self.stderr_lines, self.logger_error_func),
            daemon=True,
        )

        self.stdout_thread.start()
        self.stderr_thread.start()

        return self.process

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process and self.process.poll() is None:
            self.logger_warning_func = getattr(
                self, "logger_warning_func", self.logger_error_func
            )
            self.logger_warning_func(
                f"Subprocess (PID: {self.process.pid}) still running upon exiting context. Exception: {exc_type}. Terminating."
            )
            try:
                self.process.terminate()
            except Exception as e:
                self.logger_warning_func(f"Error sending terminate to subprocess: {e}")
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger_warning_func(
                    f"Subprocess (PID: {self.process.pid}) did not terminate gracefully. Killing."
                )
                try:
                    self.process.kill()
                except Exception as e:
                    self.logger_warning_func(f"Error sending kill to subprocess: {e}")
                self.process.wait()

        # Join threads to ensure all output is processed and resources are released
        if self.stdout_thread and self.stdout_thread.is_alive():
            self.stdout_thread.join(timeout=2)
        if self.stderr_thread and self.stderr_thread.is_alive():
            self.stderr_thread.join(timeout=2)

        return False
