from __future__ import annotations

import os
import subprocess
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Protocol
from collections.abc import Callable
import logging

from logger import get_logger, stream_reader

from services.simulation_service.core.connectors.common import (
    ConnectorInterface,
    JsonPath,
    SimulationResults,
    SimulationStatus,
)
from services.simulation_service.core.connectors.conn_utils import (
    ManagedSubprocess,
    docker_job_workspace,
    static_workspace,
)

logger = get_logger("threading-worker", filename=__name__)
module_logger = logging.getLogger(__name__)

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
    """Run the simulation as a managed subprocess."""

    def __init__(
        self,
        worker_simulation_timeout_seconds: int,
        command_builder: Callable[[JsonPath], list[str]],
        results_parser: Callable[[Path | None, str], SimulationResults],
        repo_root_getter: Callable[[], Path] | None = None,
        worker_id_getter: Callable[[], str | None] | None = None,
        managed_subprocess_factory: Callable[..., ManagedSubprocess] | None = None,
        pre_run_hook: Callable[[Path | None], None] | None = None,
    ):
        self._managed_subprocess_factory = (
            managed_subprocess_factory or _default_subprocess_factory
        )
        self._results_parser = results_parser
        self._command_builder = command_builder
        self._repo_root_getter = repo_root_getter
        self._worker_id_getter = worker_id_getter
        self._timeout_duration = worker_simulation_timeout_seconds
        self._pre_run_hook = pre_run_hook

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
        if self._pre_run_hook is not None:
            self._pre_run_hook(work_dir)

        command = self._command_builder(config)
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

            return self._parse_results(manager, work_dir, defaults)

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
                return None  # process finished
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
        self,
        manager: ManagedSubprocess,
        work_dir: Path | None,
        defaults: SimulationResults,
    ) -> tuple[SimulationStatus, SimulationResults]:
        full_stdout = "\n".join(manager.stdout_lines)
        broadcast_results = self._results_parser(work_dir, full_stdout)

        if not broadcast_results:
            logger.error(
                "Subprocess exited rc=0 but results are empty. "
                "The simulation likely crashed silently or produced no output.\n"
                f"Stdout tail:\n{_tail(manager.stdout_lines)}\n"
                f"Stderr tail:\n{_tail(manager.stderr_lines)}"
            )
            return SimulationStatus.EXCEPTION, defaults

        merged = _merge_results(defaults, broadcast_results)
        return SimulationStatus.SUCCESS, merged


class SubprocessConnectorInterface(ConnectorInterface, ABC):
    """Base class for connectors that execute their simulation as a subprocess."""

    ConnectorName: ClassVar[str] = "unnamed_subprocess_connector"

    @classmethod
    @abstractmethod
    def build_command(cls, config_path: JsonPath) -> list[str]:
        """Build the subprocess command for this simulator."""
        ...

    @classmethod
    @abstractmethod
    def parse_results(cls, work_dir: Path | None, stdout: str) -> SimulationResults:
        """Parse simulation results from the workspace.

        Use ``stdout`` for simulators that broadcast results via stdout.
        Use ``work_dir`` to read output files for file-based simulators (e.g. Eclipse).
        """
        ...

    @classmethod
    def pre_run(cls, work_dir: Path | None) -> None:
        """Called before the subprocess starts. Override for per-run setup."""

    @classmethod
    def run(
        cls,
        config_path: JsonPath,
        user_cost_function_with_default_values: SimulationResults,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]:
        timeout = int(os.environ.get("WORKER_SIMULATION_TIMEOUT_SECONDS", "900"))

        subprocess_runner = SubprocessRunner(
            worker_simulation_timeout_seconds=timeout,
            command_builder=cls.build_command,
            results_parser=cls.parse_results,
            managed_subprocess_factory=lambda *a, **k: ManagedSubprocess(*a, **k),
            repo_root_getter=cls._repo_root,
            worker_id_getter=cls._current_worker_id,
            pre_run_hook=cls.pre_run,
        )

        return subprocess_runner.run(
            config_path, user_cost_function_with_default_values, stop
        )

    @staticmethod
    def _repo_root() -> Path:
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "pyproject.toml").exists():
                return parent
        raise RuntimeError("Failed to locate repository root directory.")

    @staticmethod
    def _current_worker_id() -> str | None:
        name = threading.current_thread().name
        if name.startswith("worker-"):
            return name.split("-", 1)[1]
        return os.environ.get("SIM_WORKER_ID")


def _tail(lines: list[str], n: int = _LOG_TAIL_LINES) -> str:
    return "\n".join(lines[-n:])


def _safe_call[T](fn: Callable[[], T]) -> T | None:
    try:
        return fn()
    except Exception as e:
        func_name = getattr(fn, "__name__", str(fn))
        logger.warning(f"Safe call to '{func_name}' failed: {e}")
        return None


def _default_subprocess_factory(*args: Any, **kwargs: Any) -> ManagedSubprocess:
    return ManagedSubprocess(*args, **kwargs)


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
    if process.poll() is not None:
        return
    timeout = _GRACEFUL_TERMINATE_TIMEOUT if graceful else _STOP_TERMINATE_TIMEOUT
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Subprocess did not terminate gracefully. Killing.")
        process.kill()
        try:
            process.wait()
        except subprocess.TimeoutExpired:
            logger.warning("Subprocess did not exit after kill.")


def _join_output_threads(manager: ManagedSubprocess) -> None:
    for thread in (manager.stdout_thread, manager.stderr_thread):
        if thread and thread.is_alive():
            thread.join(timeout=_THREAD_JOIN_TIMEOUT)


def _log_process_failure(returncode: int, manager: ManagedSubprocess) -> None:
    stdout_tail = _tail(manager.stdout_lines)
    stderr_tail = _tail(manager.stderr_lines)

    if returncode == -9:
        message = (
            "Simulation subprocess was killed (rc=-9). This is often due to OOM kill. "
            f"Consider lowering worker_count or reducing model size.\n"
            f"Stdout tail:\n{stdout_tail}\nStderr tail:\n{stderr_tail}"
        )
    else:
        message = (
            f"Simulation subprocess failed rc={returncode}.\n"
            f"Stdout tail:\n{stdout_tail}\nStderr tail:\n{stderr_tail}"
        )
    logger.error(message)
    if not logger.propagate:
        module_logger.error(message)


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
