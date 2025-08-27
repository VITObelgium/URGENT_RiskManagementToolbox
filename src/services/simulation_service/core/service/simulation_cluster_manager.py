import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from logger import get_logger, log_docker_logs
from services.simulation_service.core.service.grpc_stub_manager import GrpcStubManager

logger = get_logger(__name__)


@contextmanager
def core_directory() -> Generator[None, None, None]:
    """
    A context manager to temporarily switch to the core directory and restore the original directory upon exit.

    Yields:
        None
    """
    current_file = Path(__file__)
    core_dir = current_file.parent.parent
    logger.debug("Attempting to switch to core directory: %s", core_dir)

    if not core_dir.exists():
        logger.error("Core directory not found at %s", core_dir)
        raise FileNotFoundError(f"Core directory not found at {core_dir}")

    original_dir = os.getcwd()
    logger.debug("Original working directory: %s", original_dir)

    try:
        os.chdir(core_dir)
        logger.debug("Switched to core directory: %s", os.getcwd())
        yield
    finally:
        os.chdir(original_dir)
        logger.debug("Reverted to original working directory: %s", original_dir)


class SimulationClusterManager:
    @staticmethod
    def build_simulation_cluster_images() -> None:
        logger.info("Building simulation cluster images...")
        with core_directory():
            try:
                result = subprocess.run(
                    ["docker", "compose", "build", "--progress", "plain", "--force-rm"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.debug("Docker build output:\n%s", result.stdout)
                logger.info("Simulation cluster images successfully built.")
            except subprocess.CalledProcessError as e:
                logger.error("Error building simulation cluster images:\n%s", e.stderr)
                raise

    @staticmethod
    def prune_dangling_images():
        # Clean up dangling images before starting cluster.
        # Note: dangling images are images with both <none> as their repository and tag
        try:
            logger.info("In the meantime: Pruning dangling Docker images...")
            process = subprocess.Popen(
                ["docker", "image", "prune", "-f"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            logger.debug("Started dangling image pruning process (non-blocking)")
            return process
        except Exception as e:
            logger.warning("Failed to prune Docker images: %s", str(e))
            raise

    @staticmethod
    def start_simulation_cluster(
        worker_count: int = 3, run_with_web_app: bool = False
    ) -> None:
        """
        Start the Docker-based simulation cluster with the specified worker count.
        """
        logger.info(f"Starting the simulation cluster with {worker_count} workers...")

        with core_directory():
            build_command = [
                "docker",
                "compose",
                "up",
                "-d",
                "--scale",
                f"worker={worker_count}",
            ]
            if run_with_web_app:
                build_command = [
                    "docker",
                    "compose",
                    "--profile",
                    "webui",
                    "up",
                    "-d",
                    "--scale",
                    f"worker={worker_count}",
                ]
                logger.info("Web application will be started for visualization.")

            try:
                result = subprocess.run(
                    build_command,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.debug("Docker compose output:\n%s", result.stdout)
                logger.info("Simulation cluster successfully started.")
                log_docker_logs(logger)
            except Exception as e:
                logger.error(
                    "Error starting the simulation cluster: %s",
                    getattr(e, "stderr", str(e)),
                )
                raise

            except subprocess.CalledProcessError as e:
                logger.error("Error starting the simulation cluster:\n%s", e.stderr)
                raise

    @staticmethod
    def shutdown_simulation_cluster() -> None:
        """
        Shutdown the Docker-based simulation cluster.
        """
        logger.info("Attempting to shut down the simulation cluster...")
        GrpcStubManager.close()
        with core_directory():
            try:
                subprocess.run(
                    ["docker", "compose", "down"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info("Simulation cluster successfully shut down.")
            except subprocess.CalledProcessError as e:
                logger.error(
                    "Error shutting down the simulation cluster:\n%s", e.stderr
                )
                raise


@contextmanager
def simulation_cluster_context_manager(
    worker_count: int = 3, run_with_web_app: bool = False
):
    """
    Context manager for managing the simulation cluster lifecycle.
    """
    logger.info("Entering simulation cluster context.")
    SimulationClusterManager.build_simulation_cluster_images()
    SimulationClusterManager.prune_dangling_images()
    SimulationClusterManager.start_simulation_cluster(worker_count, run_with_web_app)
    try:
        yield
    finally:
        SimulationClusterManager.shutdown_simulation_cluster()
        logger.info("Exited simulation cluster context.")
