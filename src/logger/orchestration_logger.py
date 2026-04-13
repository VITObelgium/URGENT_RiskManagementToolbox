import logging
import shutil
import subprocess
from logging import Logger
from pathlib import Path

from logger.utils import get_external_console_logging, get_services, get_services_id

TAIL_LINES = 100
WORKER_PREFIX = "simulation-worker"
SIM_SERVICE_NAME = "simulation-server"
TERMINAL_WINDOW_ID = 999  # big number to avoid conflicts with user terminals

module_logger = logging.getLogger(__name__)


def _start_external_xterm_log_terminal(title: str, command: str) -> None:
    """Start an xterm window with the specified log command."""
    subprocess.Popen(["xterm", "-hold", "-T", title, "-e", "bash", "-c", command])


def _start_external_log_terminal(title: str, command: str) -> bool:
    """Best-effort external terminal launch for log tails.
    Returns True when a terminal was successfully opened, otherwise False.
    """

    wt_path = shutil.which("wt.exe")
    wsl_path = shutil.which("wsl.exe")
    try:
        if wt_path and wsl_path:
            subprocess.run(
                [
                    wt_path,
                    "-w",
                    str(TERMINAL_WINDOW_ID),
                    "--title",
                    title,
                    wsl_path,
                    "bash",
                    "-lc",
                    command,
                ],
                check=True,
            )
            return True

        if shutil.which("xterm"):
            _start_external_xterm_log_terminal(title, command)
            return True

        module_logger.info(
            "External docker log console requested, but no supported terminal launcher was found. Continuing with file logging only."
        )
        return False
    except Exception as e:
        module_logger.warning(
            "Failed to open external terminal for logging: %s. Continuing with file logging only.",
            e,
        )
        return False


def _start_file_logging(command: str) -> None:
    """Start a background process that logs output to a file."""
    buffered_command = f"stdbuf -oL -eL {command}"
    subprocess.Popen(
        ["bash", "-c", buffered_command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def log_docker_logs(logger: Logger, disperse_worker_logs: bool = True) -> None:
    """Fetch and log docker logs for workers and simulation service."""
    assert logger is not None, "Logger must be initialized before logging Docker logs."

    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "log"
    log_dir.mkdir(exist_ok=True)

    try:
        services = get_services(logger)
    except RuntimeError as e:
        logger.error(f"Error fetching Docker services: {e}")
        return

    logger.info(f"Found services: {services}")
    if not services:
        logger.warning("No services found. Ensure Docker Compose is running.")
        return

    worker_services = sorted([s for s in services if s.startswith(WORKER_PREFIX)])
    sim_log_path = (log_dir / "docker_simulation_server.log").resolve()

    logger.info(f"Logging Docker worker services: {worker_services}")

    if worker_services:
        if disperse_worker_logs:
            for worker_service in worker_services:
                worker_log_path = (
                    log_dir / f"docker_{worker_service.replace('-', '_')}.log"
                ).resolve()
                worker_cmd = (
                    f"docker logs --tail {TAIL_LINES} --follow {worker_service}"
                )

                if get_external_console_logging() and _start_external_log_terminal(
                    f"Docker Logs: {worker_service}",
                    f"{worker_cmd} | stdbuf -oL -eL tee '{worker_log_path}'",
                ):
                    logger.info(
                        f"Opened external terminal for {worker_service} logs (also logging to {worker_log_path})."
                    )
                elif get_external_console_logging():
                    logger.info(
                        "External terminal logging unavailable for %s; continuing with file logging only.",
                        worker_service,
                    )

                _start_file_logging(f"{worker_cmd} > '{worker_log_path}'")
                logger.info(f"Logging {worker_service} logs to file: {worker_log_path}")
        else:
            worker_log_path = (log_dir / "docker_workers.log").resolve()
            worker_services = sorted(
                [s for s in get_services_id() if s.startswith(WORKER_PREFIX)]
            )

            logger.info(f"Aggregating logs for worker services: {worker_services}")
            worker_cmd = f"docker compose logs --tail {TAIL_LINES} --follow {' '.join(worker_services)}"

            if get_external_console_logging() and _start_external_log_terminal(
                "Docker Worker Logs",
                f"{worker_cmd} | stdbuf -oL -eL tee '{worker_log_path}'",
            ):
                logger.info(
                    f"Opened external terminal for aggregated worker logs (also logging to {worker_log_path})."
                )
            elif get_external_console_logging():
                logger.info(
                    "External terminal logging unavailable for aggregated worker logs; continuing with file logging only."
                )

            _start_file_logging(f"{worker_cmd} > '{worker_log_path}'")
            logger.info(f"Logging aggregated worker logs to file: {worker_log_path}")

    if SIM_SERVICE_NAME in services:
        sim_cmd = f"docker logs --tail {TAIL_LINES} --follow {SIM_SERVICE_NAME}"

        if get_external_console_logging() and _start_external_log_terminal(
            "Docker Simulation Server Logs",
            f"{sim_cmd} | stdbuf -oL -eL tee '{sim_log_path}'",
        ):
            logger.info(
                f"Opened external terminal for simulation logs (also logging to {sim_log_path})."
            )
        elif get_external_console_logging():
            logger.info(
                "External terminal logging unavailable for the simulation server; continuing with file logging only."
            )

        _start_file_logging(f"{sim_cmd} > '{sim_log_path}'")
        logger.info(f"Logging simulation logs to file: {sim_log_path}")
    elif SIM_SERVICE_NAME not in services and any(
        s.startswith(WORKER_PREFIX) for s in services
    ):
        logger.warning(
            f"Simulation service '{SIM_SERVICE_NAME}' not found. Skipping simulation server logs."
        )
