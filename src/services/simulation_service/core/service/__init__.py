import os
import subprocess
from pathlib import Path

from logger.u_logger import configure_logger, get_logger
from services.simulation_service.core.service.simulation_service import (
    SimulationService,
    simulation_cluster_contex_manager,
)

configure_logger()
logger = get_logger(__name__)
_is_initialized = False


def _initialize_images() -> None:
    """
    Initializes Docker images for the simulation service, if not already done.
    """
    global _is_initialized

    if _is_initialized:
        logger.info(
            "Docker images have already been initialized. Skipping reinitialization."
        )
        return

    current_file = Path(__file__)
    core_dir = current_file.parent.parent

    if not core_dir.exists():
        logger.error(
            "Core directory not found at %s. Initialization aborted.", core_dir
        )
        raise FileNotFoundError(f"Core directory not found at {core_dir}")

    logger.info("Starting Docker image initialization in core directory: %s", core_dir)

    original_dir = os.getcwd()
    os.chdir(core_dir)

    try:
        logger.info("Running 'docker compose build' in progress mode 'plain'...")
        result = subprocess.run(
            [
                "docker",
                "compose",
                "build",
                "--progress",
                "plain",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.debug("Docker compose build output:\n%s", result.stdout)
        logger.info("Docker compose build completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(
            "Docker compose build failed with error:\n%s", e.stderr, exc_info=True
        )
        raise
    finally:
        _is_initialized = True
        os.chdir(original_dir)
        logger.info("Reverted to the original working directory: %s", original_dir)


_initialize_images()

__all__ = ["SimulationService", "simulation_cluster_contex_manager"]
