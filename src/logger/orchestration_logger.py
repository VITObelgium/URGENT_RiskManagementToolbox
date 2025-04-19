import subprocess
from logging import Logger
from pathlib import Path

from logger.utils import get_external_console_logging


def log_docker_logs(logger: Logger):
    """Fetch and log docker logs in a separate terminal window or only to file."""
    assert logger is not None, "Logger must be initialized before logging Docker logs."
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "log"
    docker_log_path = log_dir / "docker_compose.log"
    abs_log_path = docker_log_path.resolve()

    docker_command = "docker compose logs --tail 100 --follow"

    if get_external_console_logging():
        tee_command = f"{docker_command} | tee '{abs_log_path}'"
        xterm_command = [
            "xterm",
            "-hold",
            "-T",
            "Docker Logs",
            "-e",
            "bash",
            "-c",
            tee_command,
        ]
        try:
            subprocess.Popen(xterm_command)
            logger.info(
                "Opened external terminal for Docker logs (also logging to file)."
            )
        except (OSError, subprocess.CalledProcessError) as log_err:
            logger.error(
                "Error opening external terminal for docker logs:\n%s",
                getattr(log_err, "stderr", str(log_err)),
            )
            raise
    else:
        file_command = f"{docker_command} > '{abs_log_path}'"
        try:
            subprocess.Popen(
                ["bash", "-c", file_command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Logging Docker logs only to file: %s", abs_log_path)
        except (OSError, subprocess.CalledProcessError) as log_err:
            logger.error(
                "Error logging docker logs to file:\n%s",
                getattr(log_err, "stderr", str(log_err)),
            )
            raise
