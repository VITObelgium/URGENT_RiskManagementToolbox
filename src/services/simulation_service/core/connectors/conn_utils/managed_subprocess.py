from __future__ import annotations

import subprocess
import threading
from collections.abc import Mapping
from pathlib import Path
from collections.abc import Callable


def _default_stream_reader(stream, lines: list[str], log_func: Callable) -> None:
    """Read lines from a stream, appending to `lines` and calling `log_func`."""
    try:
        for line in stream:
            stripped = line.rstrip("\n")
            lines.append(stripped)
            log_func(stripped)
    except ValueError:
        # Stream closed externally; nothing to do.
        pass
    finally:
        stream.close()


class ManagedSubprocess:
    """
    Context manager that launches a subprocess and captures its stdout/stderr
    via background reader threads.
    """

    def __init__(
        self,
        command_args: list[str],
        logger_info_func: Callable,
        logger_error_func: Callable,
        logger_warning_func: Callable | None = None,
        stream_reader_func: Callable = _default_stream_reader,
        text: bool = True,
        env: Mapping[str, str] | None = None,
        thread_name_prefix: str = "",
        cwd: str | Path | None = None,
    ):
        self.command_args = command_args
        self.stream_reader_func = stream_reader_func
        self.text = text
        self.logger_info_func = logger_info_func
        self.logger_error_func = logger_error_func
        self.logger_warning_func = logger_warning_func or logger_error_func
        self.env = dict(env) if env is not None else None
        self.cwd = str(cwd) if cwd is not None else None
        self.thread_name_prefix = thread_name_prefix

        self.process: subprocess.Popen | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []

    def __enter__(self) -> subprocess.Popen:
        try:
            command = " ".join(self.command_args)
        except Exception:
            command = str(self.command_args)
        self.logger_info_func("Starting subprocess...")
        try:
            popen_kwargs: dict = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": self.text,
                "cwd": self.cwd,
            }

            # Pass env only if explicitly provided; do not include when None.
            if self.env is not None:
                popen_kwargs["env"] = self.env

            self.process = subprocess.Popen(
                self.command_args,
                **popen_kwargs,
            )
        except Exception:
            self.log_error(f"Failed to start subprocess: {' '.join(self.command_args)}")
            raise

        if self.process.stdout is None or self.process.stderr is None:
            self.process.kill()
            self.process.wait()
            raise RuntimeError("Subprocess stdout/stderr streams are not available.")

        prefix = self.thread_name_prefix or "subprocess"
        self._stdout_thread = threading.Thread(
            target=self.stream_reader_func,
            args=(self.process.stdout, self.stdout_lines, self.log_info),
            daemon=True,
            name=f"{prefix}:stdout",
        )
        self._stderr_thread = threading.Thread(
            target=self.stream_reader_func,
            args=(self.process.stderr, self.stderr_lines, self.log_error),
            daemon=True,
            name=f"{prefix}:stderr",
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.process is not None and self.process.poll() is None:
            self.logger_warning_func(
                f"Subprocess (PID {self.process.pid}) still running on context exit "
                f"(exc={exc_type}). Terminating."
            )
            try:
                self.process.terminate()
            except OSError as e:
                self.logger_warning_func(f"terminate() failed: {e}")

            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger_warning_func(
                    f"Subprocess (PID {self.process.pid}) did not terminate "
                    "gracefully after 5 s — killing."
                )
                try:
                    self.process.kill()
                except OSError as e:
                    self.logger_warning_func(f"kill() failed: {e}")
                self.process.wait()

        if self._stdout_thread is not None:
            self._stdout_thread.join()
        if self._stderr_thread is not None:
            self._stderr_thread.join()

    @property
    def stdout_thread(self) -> threading.Thread | None:
        return self._stdout_thread

    @property
    def stderr_thread(self) -> threading.Thread | None:
        return self._stderr_thread

    @property
    def returncode(self) -> int | None:
        return self.process.returncode if self.process else None

    def wait(self, timeout: float | None = None) -> int:
        if self.process is None:
            raise RuntimeError("Subprocess has not been started.")
        self.process.wait(timeout=timeout)
        if self._stdout_thread:
            self._stdout_thread.join()
        if self._stderr_thread:
            self._stderr_thread.join()
        if self.returncode is None:
            raise RuntimeError("Subprocess finished but returncode is None.")
        return self.returncode
