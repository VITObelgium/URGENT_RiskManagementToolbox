import subprocess
import threading
from collections.abc import Mapping
from typing import Callable


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

    The critical guarantee: by the time ``__exit__`` returns, *all* output
    has been appended to ``stdout_lines`` / ``stderr_lines``.  This prevents
    the race where the caller inspects those lists immediately after the
    process exits but before the reader threads have finished draining the
    pipes.

    Parameters
    ----------
    command_args:
        Argument list passed directly to ``subprocess.Popen``.
    log_info / log_error / log_warning:
        Callables used for structured logging.  ``log_warning`` defaults to
        ``log_error`` when omitted.
    stream_reader_func:
        Optional override for the stream-reader callable; defaults to
        ``_default_stream_reader``.  Signature must be
        ``(stream, lines: list[str], log_func: Callable) -> None``.
    text:
        Passed through to ``Popen``; ``True`` decodes streams as text.
    env:
        Optional environment mapping.  When provided it completely replaces
        the inherited environment (same as ``Popen`` semantics).
    thread_name_prefix:
        Prefix for the reader thread names (aids debugging).
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
    ):
        self.command_args = command_args
        self.stream_reader_func = stream_reader_func
        self.log_info = logger_info_func
        self.log_error = logger_error_func
        self.log_warning = logger_warning_func or logger_error_func
        self.text = text
        self.env: dict[str, str] | None = dict(env) if env is not None else None
        self.thread_name_prefix = thread_name_prefix

        self.process: subprocess.Popen | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "ManagedSubprocess":
        cwd = self.env.get("PWD") if self.env else None
        self.log_info(f"Starting subprocess: {' '.join(self.command_args)}")

        popen_kwargs: dict = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": self.text,
            "cwd": cwd,
        }
        if self.env is not None:
            popen_kwargs["env"] = self.env

        try:
            self.process = subprocess.Popen(self.command_args, **popen_kwargs)
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

        # Return self so callers can inspect stdout_lines / stderr_lines after
        # the block, and optionally access self.process for the returncode.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # ----------------------------------------------------------------
        # 1. Ensure the process has actually finished.
        # ----------------------------------------------------------------
        if self.process is not None and self.process.poll() is None:
            self.log_warning(
                f"Subprocess (PID {self.process.pid}) still running on context exit "
                f"(exc={exc_type}). Terminating."
            )
            try:
                self.process.terminate()
            except OSError as e:
                self.log_warning(f"terminate() failed: {e}")

            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.log_warning(
                    f"Subprocess (PID {self.process.pid}) did not terminate "
                    "gracefully after 5 s — killing."
                )
                try:
                    self.process.kill()
                except OSError as e:
                    self.log_warning(f"kill() failed: {e}")
                self.process.wait()

        # ----------------------------------------------------------------
        # 2. CRITICAL: join reader threads *before* returning.
        #    This guarantees stdout_lines / stderr_lines are fully populated
        #    by the time the caller's `with` block ends, eliminating the race
        #    condition where rc=0 but the output lists appear empty.
        # ----------------------------------------------------------------
        if self._stdout_thread is not None:
            self._stdout_thread.join()  # no timeout — drain completely
        if self._stderr_thread is not None:
            self._stderr_thread.join()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def stdout_thread(self) -> threading.Thread | None:
        """Return the background thread reading stdout."""
        return self._stdout_thread

    @property
    def stderr_thread(self) -> threading.Thread | None:
        """Return the background thread reading stderr."""
        return self._stderr_thread

    @property
    def returncode(self) -> int | None:
        """Return the process exit code, or None if not yet finished."""
        return self.process.returncode if self.process else None

    def wait(self, timeout: float | None = None) -> int:
        """
        Block until the subprocess exits and *all* output has been captured.

        Raises ``subprocess.TimeoutExpired`` if ``timeout`` is given and
        the process does not finish in time.
        """
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
