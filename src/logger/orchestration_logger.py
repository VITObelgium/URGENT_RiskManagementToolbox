import subprocess
from logging import Logger
from pathlib import Path
from typing import List

import docker

from logger.utils import get_external_console_logging

TAIL_LINES = 100
WORKER_PREFIX = "worker"
SIM_SERVICE_NAME = "simulation_server"


def get_services() -> List[str]:
    """Return a list of all docker compose service names using Docker SDK."""
    try:
        client = docker.from_env()
        containers = client.containers.list()

        # Extract service names from running containers
        services = []
        for container in containers:
            if "com.docker.compose.service" in container.labels:
                service_name = container.labels["com.docker.compose.service"]
                if service_name not in services:
                    services.append(service_name)

        return services
    except Exception as e:
        print(f"Unexpected error in get_services: {e}")
        return []


def _start_external_xterm_log_terminal(title: str, command: str) -> None:
    """Start an xterm window with the specified log command."""
    subprocess.Popen(["xterm", "-hold", "-T", title, "-e", "bash", "-c", command])


def _start_external_log_terminal(title: str, command: str) -> None:
    """Start a tmux session with the specified log command."""

    session_name = f"log_{title.replace(' ', '_').lower()}"

    try:
        escaped_command = command.replace('"', '\\"')
        wt_command = (
            f'wt.exe new-tab --title "{title}" wsl.exe bash -c "{escaped_command}"'
        )
        subprocess.run(wt_command, shell=True)
    except (OSError, subprocess.SubprocessError):
        _start_external_xterm_log_terminal(
            title, f"tmux attach-session -t {session_name}"
        )


def _start_file_logging(command: str) -> None:
    """Start a background process that logs output to a file."""
    buffered_command = f"stdbuf -oL -eL {command}"
    subprocess.Popen(
        ["bash", "-c", buffered_command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def log_docker_logs(logger: Logger) -> None:
    """Fetch and log docker logs for workers and simulation service."""
    assert logger is not None, "Logger must be initialized before logging Docker logs."

    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "log"
    log_dir.mkdir(exist_ok=True)

    services = get_services()
    logger.info(f"Found services: {services}")
    if not services:
        logger.warning("No services found. Ensure Docker Compose is running.")
        return

    worker_services = sorted([s for s in services if s.startswith(WORKER_PREFIX)])

    worker_log_path = (log_dir / "docker_workers.log").resolve()
    sim_log_path = (log_dir / "docker_simulation_server.log").resolve()

    try:
        for session in subprocess.check_output(
            ["tmux", "list-sessions"], text=True, stderr=subprocess.DEVNULL
        ).splitlines():
            if session.startswith("log_"):
                session_name = session.split(":")[0]
                logger.info(f"Killing existing tmux session: {session_name}")
                subprocess.Popen(["tmux", "kill-session", "-t", session_name])
    except subprocess.CalledProcessError:
        pass

    if worker_services:
        worker_cmd = f"docker compose logs --tail {TAIL_LINES} --follow {' '.join(worker_services)}"

        if get_external_console_logging():
            _start_external_log_terminal(
                "Docker Worker Logs",
                f"{worker_cmd} | stdbuf -oL -eL tee '{worker_log_path}'",
            )
            logger.info(
                f"Opened external terminal for worker logs (also logging to {worker_log_path})."
            )

        _start_file_logging(f"{worker_cmd} > '{worker_log_path}'")
        logger.info(f"Logging worker logs to file: {worker_log_path}")

    sim_cmd = f"docker logs --tail {TAIL_LINES} --follow {SIM_SERVICE_NAME}"

    if get_external_console_logging():
        _start_external_log_terminal(
            "Docker Simulation Service Logs",
            f"{sim_cmd} | stdbuf -oL -eL tee '{sim_log_path}'",
        )
        logger.info(
            f"Opened external terminal for simulation logs (also logging to {sim_log_path})."
        )

    _start_file_logging(f"{sim_cmd} > '{sim_log_path}'")
    logger.info(f"Logging simulation logs to file: {sim_log_path}")
