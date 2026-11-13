"""Tests for SubprocessRunner helpers and edge-case paths in runner.py."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.simulation_service.core.connectors.common import SimulationStatus
from services.simulation_service.core.connectors.runner import (
    SubprocessRunner,
    _build_env,
    _log_process_failure,
    _merge_results,
    _safe_call,
    _tail,
    _terminate_process,
)

_RUNNER = "services.simulation_service.core.connectors.runner"


# ---------------------------------------------------------------------------
# _tail
# ---------------------------------------------------------------------------


def test_tail_returns_last_n_lines():
    lines = [str(i) for i in range(200)]
    result = _tail(lines, n=5)
    assert result == "195\n196\n197\n198\n199"


def test_tail_empty_lines():
    assert _tail([], n=10) == ""


# ---------------------------------------------------------------------------
# _safe_call
# ---------------------------------------------------------------------------


def test_safe_call_returns_value_on_success():
    result = _safe_call(lambda: 42)
    assert result == 42


def test_safe_call_returns_none_on_exception():
    result = _safe_call(lambda: 1 / 0)
    assert result is None


# ---------------------------------------------------------------------------
# _merge_results
# ---------------------------------------------------------------------------


def test_merge_results_matching_keys():
    defaults = {"a": 1.0, "b": 2.0}
    broadcast = {"a": 10.0, "b": 20.0}
    merged = _merge_results(defaults, broadcast)
    assert merged == {"a": 10.0, "b": 20.0}


def test_merge_results_key_mismatch_raises():
    defaults = {"a": 1.0}
    broadcast = {"b": 2.0}
    with pytest.raises(KeyError, match="Missing in connector"):
        _merge_results(defaults, broadcast)


# ---------------------------------------------------------------------------
# _log_process_failure
# ---------------------------------------------------------------------------


def test_log_process_failure_oom_kill(caplog):
    manager = MagicMock()
    manager.stdout_lines = ["out"]
    manager.stderr_lines = ["err"]
    with caplog.at_level("ERROR"):
        _log_process_failure(-9, manager)
    assert (
        "OOM" in caplog.text
        or "killed" in caplog.text.lower()
        or "rc=-9" in caplog.text
    )


def test_log_process_failure_normal_exit(caplog):
    manager = MagicMock()
    manager.stdout_lines = []
    manager.stderr_lines = []
    with caplog.at_level("ERROR"):
        _log_process_failure(1, manager)
    assert "rc=1" in caplog.text


# ---------------------------------------------------------------------------
# _terminate_process
# ---------------------------------------------------------------------------


def test_terminate_process_already_finished():
    """Should not call terminate() if process already exited."""
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 0  # already done
    _terminate_process(mock_proc)
    mock_proc.terminate.assert_not_called()


def test_terminate_process_kills_after_timeout():
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running
    mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)
    _terminate_process(mock_proc, graceful=True)
    mock_proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# _build_env
# ---------------------------------------------------------------------------


def test_build_env_sets_worker_subprocess_flag():
    env = _build_env(None, "thread")
    assert env["URGENT_WORKER_SUBPROCESS"] == "1"


def test_build_env_docker_mode_sets_pwd(tmp_path: Path):
    env = _build_env(tmp_path, "docker")
    assert env["PWD"] == str(tmp_path)
    assert env["SIM_MODEL_DIR"] == str(tmp_path)


def test_build_env_thread_mode_with_repo_root_sets_pythonpath(tmp_path: Path):
    env = _build_env(tmp_path, "thread", repo_root=tmp_path)
    assert str(tmp_path / "src") in env.get("PYTHONPATH", "")


# ---------------------------------------------------------------------------
# SubprocessRunner — error exception paths
# ---------------------------------------------------------------------------


class TestSubprocessRunnerErrorPaths:
    def _make_runner(self, results_parser=None, command_builder=None):
        return SubprocessRunner(
            worker_simulation_timeout_seconds=10,
            command_builder=command_builder or (lambda cfg: ["echo", "hi"]),
            results_parser=results_parser or (lambda wd, s: {"cost": 0.0}),
        )

    def test_docker_mode_no_sim_model_dir_returns_exception(self):
        runner = self._make_runner()
        with patch.dict("os.environ", {"RUNNER_MODE": "docker"}, clear=False):
            if "SIM_MODEL_DIR" in __import__("os").environ:
                __import__("os").environ.pop("SIM_MODEL_DIR", None)
            with patch.dict("os.environ", {}, clear=False) as env:
                env.pop("SIM_MODEL_DIR", None)
                status, _ = runner.run({"config": {}}, {"cost": float("nan")})
        assert status == SimulationStatus.EXCEPTION

    def test_key_error_returns_exception(self):
        """Broadcast results with wrong keys → KeyError → EXCEPTION status."""

        def bad_parser(work_dir, stdout):
            return {"wrong_key": 1.0}  # mismatches default key "cost"

        runner = self._make_runner(results_parser=bad_parser)

        with patch(f"{_RUNNER}.static_workspace") as mock_ws:
            mock_ws.return_value.__enter__ = lambda s: Path("/tmp")
            mock_ws.return_value.__exit__ = MagicMock(return_value=False)

            mock_mgr = MagicMock()
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_mgr.process = mock_proc
            mock_mgr.stdout_lines = ["result: 1.0"]
            mock_mgr.stderr_lines = []
            mock_mgr.__enter__ = lambda s: s
            mock_mgr.__exit__ = MagicMock(return_value=False)

            def _factory(*a, **kw):
                return mock_mgr

            runner._managed_subprocess_factory = _factory
            runner._results_parser = bad_parser

            # Override _wait_for_process to return None (process finished normally)
            with patch.object(runner, "_wait_for_process", return_value=None):
                status, _ = runner.run({"config": {}}, {"cost": float("nan")})

        assert status == SimulationStatus.EXCEPTION

    def test_file_not_found_returns_exception(self):
        """FileNotFoundError in subprocess start → EXCEPTION status."""
        runner = self._make_runner(command_builder=lambda cfg: ["nonexistent_cmd_xyz"])

        with patch(f"{_RUNNER}.static_workspace") as mock_ws:
            mock_ws.return_value.__enter__ = lambda s: None
            mock_ws.return_value.__exit__ = MagicMock(return_value=False)

            with patch(f"{_RUNNER}.ManagedSubprocess") as mock_ms_cls:
                mock_ms = MagicMock()
                mock_ms.__enter__ = MagicMock(
                    side_effect=FileNotFoundError("not found")
                )
                mock_ms.__exit__ = MagicMock(return_value=False)
                mock_ms_cls.return_value = mock_ms

                status, _ = runner.run({"config": {}}, {"cost": float("nan")})

        assert status == SimulationStatus.EXCEPTION
