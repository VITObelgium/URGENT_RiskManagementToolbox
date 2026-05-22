from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Callable

from logger import get_logger, stream_reader

from .common import (
    JsonPath,
    SimulationResults,
    SimulationStatus,
)
from .conn_utils import (
    ManagedSubprocess,
    docker_job_workspace,
    static_workspace,
)

logger = get_logger("threading-worker", filename=__name__)

_LOG_TAIL_LINES = 100
_GRACEFUL_TERMINATE_TIMEOUT = 5
_STOP_TERMINATE_TIMEOUT = 3
_THREAD_JOIN_TIMEOUT = 2
_POLL_STEP = 0.25


class SimulationRunner(Protocol):
    def run(
        self,
        config: JsonPath,
        user_cost_function_with_default_values: SimulationResults,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]: ...


class SubprocessRunner:
    """Run the simulation using the existing ManagedSubprocess logic."""

    def __init__(
        self,
        worker_simulation_timeout_seconds: int,
        repo_root_getter: Callable[[], Path] | None = None,
        worker_id_getter: Callable[[], str | None] | None = None,
        managed_subprocess_factory: Callable[..., ManagedSubprocess] | None = None,
        broadcast_results_parser: Callable[[str], SimulationResults] | None = None,
    ):
        if broadcast_results_parser is None:
            raise RuntimeError(
                "SubprocessRunner requires a broadcast_results_parser to be provided"
            )

        self._managed_subprocess_factory = (
            managed_subprocess_factory or _default_subprocess_factory
        )
        self._broadcast_results_parser = broadcast_results_parser
        self._repo_root_getter = repo_root_getter
        self._worker_id_getter = worker_id_getter
        self._timeout_duration = worker_simulation_timeout_seconds

    def run(
        self,
        config: JsonPath,
        user_cost_function_with_default_values: SimulationResults,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]:
        runner_mode = os.environ.get("RUNNER_MODE", "thread").lower()

        repo_root: Path | None = None
        if self._repo_root_getter is not None:
            root_result = _safe_call(self._repo_root_getter)
            if isinstance(root_result, Path):
                repo_root = root_result

        worker_id: str | None = None
        if self._worker_id_getter is not None:
            id_result = _safe_call(self._worker_id_getter)
            if isinstance(id_result, str):
                worker_id = id_result

        _cleanup_stale_caches()

        workspace_mgr = self._get_workspace_manager(runner_mode, repo_root, worker_id)

        if workspace_mgr is None:
            return SimulationStatus.EXCEPTION, user_cost_function_with_default_values

        try:
            with workspace_mgr as work_dir:
                return self._execute_in_workspace(
                    config,
                    user_cost_function_with_default_values,
                    work_dir,
                    worker_id,
                    runner_mode,
                    stop,
                    repo_root=repo_root,
                )
        except KeyError as e:
            logger.exception(
                f"Broadcast results keys do not match user cost-function keys. "
                f"Expected keys: {sorted(user_cost_function_with_default_values.keys())}. "
                f"Raw error: {e}"
            )
            return SimulationStatus.EXCEPTION, user_cost_function_with_default_values
        except FileNotFoundError as e:
            logger.exception(
                f"Failed to prepare simulation workspace or command not found: {e}"
            )
            return SimulationStatus.EXCEPTION, user_cost_function_with_default_values
        except Exception as e:
            logger.exception(f"Error running simulation subprocess: {e}")
            return SimulationStatus.EXCEPTION, user_cost_function_with_default_values

    def _get_workspace_manager(
        self,
        runner_mode: str,
        repo_root: Path | None,
        worker_id: str | None,
    ):
        if runner_mode == "docker":
            template_dir_raw = os.environ.get("SIM_MODEL_DIR")
            if not template_dir_raw:
                logger.error("Docker runner mode requires SIM_MODEL_DIR to be set.")
                return None
            return docker_job_workspace(Path(template_dir_raw))

        # Thread mode
        if repo_root is None or worker_id is None:
            logger.debug(
                "Thread runner mode is missing repo_root or worker_id; "
                "running without an isolated worker workspace."
            )
            return static_workspace(None)

        run_id = os.environ.get("URGENT_RUN_ID", "default")
        work_dir = (
            repo_root / f"orchestration_files_{run_id}" / f".worker_{worker_id}_temp"
        )
        return static_workspace(work_dir)

    def _execute_in_workspace(
        self,
        config: JsonPath,
        defaults: SimulationResults,
        work_dir: Path | None,
        worker_id: str | None,
        runner_mode: str,
        stop: threading.Event | None,
        repo_root: Path | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]:
        command = _build_command(config, runner_mode)
        env = _build_env(work_dir, runner_mode, repo_root=repo_root)

        if work_dir is not None:
            work_dir.mkdir(parents=True, exist_ok=True)

        tw_logger = get_logger("threading-worker")
        manager = self._managed_subprocess_factory(
            command_args=command,
            stream_reader_func=stream_reader,
            logger_info_func=tw_logger.info,
            logger_error_func=tw_logger.error,
            env=env,
            cwd=work_dir,
            thread_name_prefix=f"worker-{worker_id}" if worker_id else None,
        )

        with manager as _:
            popen = manager.process
            if popen is None:
                return SimulationStatus.FAILED, defaults

            status = self._wait_for_process(popen, manager, stop)
            if status is not None:
                return status, defaults

            if popen.returncode != 0:
                _log_process_failure(popen.returncode or 1, manager)
                return SimulationStatus.FAILED, defaults

            return self._parse_results(manager, defaults)

    def _wait_for_process(
        self,
        process: subprocess.Popen,
        manager: ManagedSubprocess,
        stop: threading.Event | None,
    ) -> SimulationStatus | None:
        waited = 0.0
        while True:
            if stop is not None and stop.is_set():
                logger.warning("Stop requested; terminating simulation subprocess.")
                _terminate_process(process, graceful=False)
                return SimulationStatus.FAILED

            try:
                process.wait(timeout=_POLL_STEP)
                return None  # Success
            except subprocess.TimeoutExpired:
                waited += _POLL_STEP
                if waited >= self._timeout_duration:
                    logger.warning(
                        f"Subprocess timed out after {self._timeout_duration} seconds. Terminating."
                    )
                    _terminate_process(process, graceful=True)
                    _join_output_threads(manager)
                    return SimulationStatus.TIMEOUT

    def _parse_results(
        self, manager: ManagedSubprocess, defaults: SimulationResults
    ) -> tuple[SimulationStatus, SimulationResults]:
        parser = self._broadcast_results_parser
        full_stdout = "\n".join(manager.stdout_lines)
        broadcast_results = parser(full_stdout)  # type: ignore

        if not broadcast_results:
            logger.error(
                "Subprocess exited rc=0 but broadcast results are empty. "
                "The simulation likely crashed silently or produced no broadcast output.\n"
                f"Stdout tail:\n{_tail(manager.stdout_lines)}\n"
                f"Stderr tail:\n{_tail(manager.stderr_lines)}"
            )
            return SimulationStatus.EXCEPTION, defaults

        merged = _merge_results(defaults, broadcast_results)
        return SimulationStatus.SUCCESS, merged


class ThreadRunner:
    """Lightweight wrapper that sets thread mode environment and delegates to SubprocessRunner."""

    def __init__(self, subprocess_runner: SubprocessRunner):
        self._subprocess_runner = subprocess_runner

    def run(
        self,
        config: JsonPath,
        user_cost_function_with_default_values: SimulationResults,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]:
        return self._subprocess_runner.run(
            config, user_cost_function_with_default_values, stop
        )


def _tail(lines: list[str], n: int = _LOG_TAIL_LINES) -> str:
    return "\n".join(lines[-n:])


def _safe_call[T](fn: Callable[[], T]) -> T | None:
    try:
        return fn()
    except Exception as e:
        func_name = getattr(fn, "__name__", str(fn))
        logger.warning(f"Safe call to '{func_name}' failed: {e}")
        return None


def _cleanup_stale_caches() -> None:
    try:
        for p in Path.cwd().glob("obl_point_data_*.pkl"):
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                logger.debug(f"Could not remove cache file {p}: {e}")
    except OSError as e:
        logger.warning(
            f"Failed to scan/remove old 'obl_point_data_*.pkl' caches; proceeding anyway. Error: {e}"
        )


def _default_subprocess_factory(*args: Any, **kwargs: Any) -> ManagedSubprocess:
    return ManagedSubprocess(*args, **kwargs)


def _build_command(config: JsonPath, runner_mode: str) -> list[str]:
    if runner_mode == "docker":
        return [sys.executable, "-u", "main.py", config]
    return ["pixi", "run", "-e", "worker", "python", "-u", "main.py", config]


def _build_env(
    work_dir: Path | None, runner_mode: str, repo_root: Path | None = None
) -> dict[str, str]:
    env = os.environ.copy()
    env["URGENT_WORKER_SUBPROCESS"] = "1"
    if work_dir is not None:
        env["PWD"] = str(work_dir)
        if runner_mode == "docker":
            env["SIM_MODEL_TEMPLATE_DIR"] = os.environ.get("SIM_MODEL_DIR", "")
            env["SIM_MODEL_DIR"] = str(work_dir)
    if runner_mode != "docker" and repo_root is not None:
        src_path = str(repo_root / "src")
        plugins_path = str(repo_root / "plugins")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{src_path}{os.pathsep}{plugins_path}{os.pathsep}{existing}".rstrip(
                os.pathsep
            )
        )
    return env


def _terminate_process(process: subprocess.Popen, graceful: bool = True) -> None:
    """Terminate or kill a running process depending on the gracefulness requirement."""
    if process.poll() is not None:
        return
    timeout = _GRACEFUL_TERMINATE_TIMEOUT if graceful else _STOP_TERMINATE_TIMEOUT
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Subprocess did not terminate gracefully. Killing.")
        process.kill()
        process.wait()


def _join_output_threads(manager: ManagedSubprocess) -> None:
    for thread in (manager.stdout_thread, manager.stderr_thread):
        if thread and thread.is_alive():
            thread.join(timeout=_THREAD_JOIN_TIMEOUT)


def _log_process_failure(returncode: int, manager: ManagedSubprocess) -> None:
    stdout_tail = _tail(manager.stdout_lines)
    stderr_tail = _tail(manager.stderr_lines)

    if returncode == -9:
        logger.error(
            "Simulation subprocess was killed (rc=-9). This is often due to OOM kill. "
            f"Consider lowering worker_count or reducing model size.\n"
            f"Stdout tail:\n{stdout_tail}\nStderr tail:\n{stderr_tail}"
        )
    else:
        logger.error(
            f"Simulation subprocess failed rc={returncode}.\n"
            f"Stdout tail:\n{stdout_tail}\nStderr tail:\n{stderr_tail}"
        )

    if (
        "BlockingIOError" in stderr_tail
        and "h5py" in stderr_tail
        and "Unable to synchronously create file" in stderr_tail
    ):
        logger.error(
            "Detected HDF5 file locking error from h5py. Ensure each worker runs "
            "in an isolated directory and no other process holds the same HDF5 file open."
        )


def _merge_results(
    defaults: SimulationResults, broadcast_results: SimulationResults
) -> SimulationResults:
    default_keys = set(defaults.keys())
    result_keys = set(broadcast_results.keys())

    if default_keys != result_keys:
        missing = sorted(default_keys - result_keys)
        extra = sorted(result_keys - default_keys)
        raise KeyError(
            f"Broadcast results keys do not match user cost-function keys. "
            f"Missing in connector={missing}, extra in connector={extra}"
        )

    return {**defaults, **broadcast_results}
