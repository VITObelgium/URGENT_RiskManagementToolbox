import subprocess
from logging import Logger
from pathlib import Path

from logger.utils import get_external_console_logging, get_services, get_services_id

TAIL_LINES = 100
WORKER_PREFIX = "core-worker"
SIM_SERVICE_NAME = "simulation_server"


def _start_external_xterm_log_terminal(title: str, command: str) -> None:
    """Start an xterm window with the specified log command."""
    subprocess.Popen(["xterm", "-hold", "-T", title, "-e", "bash", "-c", command])


def _start_external_log_terminal(title: str, command: str) -> None:
    """Start an external terminal session with the specified log command."""

    try:
        escaped_command = command.replace('"', '\\"')
        wt_command = (
            f'wt.exe new-tab --title "{title}" wsl.exe bash -c "{escaped_command}"'
        )
        subprocess.run(wt_command, shell=True)
    except (OSError, subprocess.SubprocessError):
        _start_external_xterm_log_terminal(
            title,
            command,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to open external terminal for logging: {e}") from e


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
                    log_dir / f'docker_{worker_service.replace("-", "_")}.log'
                ).resolve()
                worker_cmd = (
                    f"docker logs --tail {TAIL_LINES} --follow {worker_service}"
                )

                if get_external_console_logging():
                    _start_external_log_terminal(
                        f"Docker Logs: {worker_service}",
                        f"{worker_cmd} | stdbuf -oL -eL tee '{worker_log_path}'",
                    )
                    logger.info(
                        f"Opened external terminal for {worker_service} logs (also logging to {worker_log_path})."
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

            if get_external_console_logging():
                _start_external_log_terminal(
                    "Docker Worker Logs",
                    f"{worker_cmd} | stdbuf -oL -eL tee '{worker_log_path}'",
                )
                logger.info(
                    f"Opened external terminal for aggregated worker logs (also logging to {worker_log_path})."
                )

            _start_file_logging(f"{worker_cmd} > '{worker_log_path}'")
            logger.info(f"Logging aggregated worker logs to file: {worker_log_path}")

    if SIM_SERVICE_NAME in services:
        sim_cmd = f"docker logs --tail {TAIL_LINES} --follow {SIM_SERVICE_NAME}"

        if get_external_console_logging():
            _start_external_log_terminal(
                "Docker Simulation Server Logs",
                f"{sim_cmd} | stdbuf -oL -eL tee '{sim_log_path}'",
            )
            logger.info(
                f"Opened external terminal for simulation logs (also logging to {sim_log_path})."
            )

        _start_file_logging(f"{sim_cmd} > '{sim_log_path}'")
        logger.info(f"Logging simulation logs to file: {sim_log_path}")
    elif SIM_SERVICE_NAME not in services and any(
        s.startswith(WORKER_PREFIX) for s in services
    ):
        logger.warning(
            f"Simulation service '{SIM_SERVICE_NAME}' not found. Skipping simulation server logs."
        )
