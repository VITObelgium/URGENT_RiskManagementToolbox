import io
import subprocess

import pytest

from services.simulation_service.core.connectors.conn_utils.managed_subprocess import (
    ManagedSubprocess,
    _default_stream_reader,
)


def _noop_stream_reader(stream, lines, log_func):
    """Minimal stream reader that drains a stream into lines list."""
    for line in stream:
        lines.append(line)


def _make_managed(command_args, **kwargs) -> ManagedSubprocess:
    return ManagedSubprocess(
        command_args=command_args,
        stream_reader_func=_noop_stream_reader,
        logger_info_func=lambda msg: None,
        logger_error_func=lambda msg: None,
        **kwargs,
    )


class TestManagedSubprocessBasic:
    def test_starts_subprocess_and_returns_popen(self):
        ms = _make_managed(["echo", "hello"])
        with ms as m:
            assert isinstance(m, ManagedSubprocess)
            assert isinstance(m.process, subprocess.Popen)
            m.process.wait()

    def test_stdout_lines_collected(self):
        ms = _make_managed(["echo", "hello_test"])
        with ms as m:
            m.process.wait()
        # Threads are automatically joined on __exit__
        assert any("hello_test" in line for line in m.stdout_lines)

    def test_exit_with_no_exception(self):
        ms = _make_managed(["echo", "done"])
        with ms as m:
            m.process.wait()

    def test_default_thread_names_set(self):
        ms = _make_managed(["echo", "hi"])
        with ms as _:
            assert ms.stdout_thread is not None
            assert ms.stderr_thread is not None
            assert ms.stdout_thread.name == "subprocess:stdout"
            assert ms.stderr_thread.name == "subprocess:stderr"

    def test_custom_thread_name_prefix(self):
        ms = _make_managed(["echo", "hi"], thread_name_prefix="myworker")
        with ms as _:
            assert ms.stdout_thread is not None
            assert ms.stdout_thread.name == "myworker:stdout"

    def test_env_passed_to_subprocess(self):
        import os

        env = dict(os.environ)
        env["TEST_MANAGED_SUBPROCESS"] = "yes"
        ms = _make_managed(
            ["printenv", "TEST_MANAGED_SUBPROCESS"],
            env=env,
        )
        with ms as m:
            m.process.wait()
        # automatically joined on exit
        assert any("yes" in line for line in ms.stdout_lines)

    def test_no_env_uses_default(self):
        """No env kwarg should succeed without passing env to Popen."""
        ms = _make_managed(["echo", "default_env"])
        with ms as m:
            m.process.wait()


class TestManagedSubprocessFailure:
    def test_invalid_command_raises(self):
        ms = _make_managed(["this_command_does_not_exist_xyz"])
        with pytest.raises(FileNotFoundError):
            with ms:
                pass

    def test_process_is_none_after_failed_start(self):
        ms = _make_managed(["this_command_does_not_exist_xyz"])
        try:
            with ms:
                pass
        except FileNotFoundError:
            pass
        assert ms.process is None


class TestManagedSubprocessExit:
    def test_exit_terminates_still_running_process(self):
        """If process is still running when context exits, it should be terminated."""
        ms = _make_managed(["sleep", "100"])
        try:
            with ms:
                # Exit the context immediately while the process is still running
                pass
        except Exception:
            pass
        # After context exit, process should be terminated
        if ms.process is not None:
            ms.process.wait(timeout=5)
            assert ms.process.poll() is not None

    def test_exit_joins_threads(self):
        ms = _make_managed(["echo", "join_test"])
        with ms as m:
            m.process.wait()
        # After exit, threads should be done
        if ms.stdout_thread:
            assert not ms.stdout_thread.is_alive()
        if ms.stderr_thread:
            assert not ms.stderr_thread.is_alive()

    def test_exit_returns_false(self):
        """__exit__ should return None (not suppress exceptions)."""
        ms = _make_managed(["echo", "hi"])
        with ms as m:
            m.process.wait()
        result = ms.__exit__(None, None, None)
        assert result is None


class TestDefaultStreamReader:
    def test_reads_lines_into_list(self):
        lines: list[str] = []
        logged: list[str] = []
        stream = io.StringIO("line1\nline2\nline3\n")
        _default_stream_reader(stream, lines, logged.append)
        assert lines == ["line1", "line2", "line3"]

    def test_calls_log_func_for_each_line(self):
        logged: list[str] = []
        stream = io.StringIO("a\nb\n")
        _default_stream_reader(stream, [], logged.append)
        assert logged == ["a", "b"]

    def test_stream_closed_value_error_is_swallowed(self):
        """ValueError raised mid-iteration (stream closed) must be ignored."""

        class _ClosingStream:
            def __iter__(self):
                yield "first"
                raise ValueError("I/O operation on closed file")

            def close(self):
                pass

        lines: list[str] = []
        _default_stream_reader(_ClosingStream(), lines, lambda _: None)
        assert lines == ["first"]

    def test_stream_close_called_in_finally(self):
        """stream.close() must be called even when no exception occurs."""
        closed = []

        class _TrackingStream:
            def __iter__(self):
                return iter(["x\n"])

            def close(self):
                closed.append(True)

        _default_stream_reader(_TrackingStream(), [], lambda _: None)
        assert closed == [True]


class TestManagedSubprocessWait:
    def test_wait_returns_returncode(self):
        ms = _make_managed(["true"])
        with ms as m:
            rc = m.wait()
        assert rc == 0

    def test_wait_raises_if_not_started(self):
        ms = _make_managed(["echo", "hi"])
        with pytest.raises(RuntimeError, match="not been started"):
            ms.wait()

    def test_returncode_is_none_before_start(self):
        ms = _make_managed(["echo", "hi"])
        assert ms.returncode is None


class TestManagedSubprocessLogMethods:
    def test_log_info_calls_info_func(self):
        logged: list[str] = []
        ms = ManagedSubprocess(
            command_args=["echo", "hi"],
            logger_info_func=logged.append,
            logger_error_func=lambda _: None,
        )
        ms.log_info("hello")
        assert "hello" in logged

    def test_log_warning_defaults_to_error_func(self):
        """When no logger_warning_func is given, warning falls back to error func."""
        errors: list[str] = []
        ms = ManagedSubprocess(
            command_args=["echo", "hi"],
            logger_info_func=lambda _: None,
            logger_error_func=errors.append,
        )
        ms.log_warning("warn msg")
        assert "warn msg" in errors
