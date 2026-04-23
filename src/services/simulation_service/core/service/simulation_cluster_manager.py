import os
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Generator

from logger import get_logger, log_docker_logs
from services.simulation_service.core.config import get_simulation_config
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
    def wait_for_server_readiness(timeout: float | None = None, interval: float = 0.25):
        config = get_simulation_config()
        deadline = time.time() + (
            timeout if timeout is not None else config.server_startup_timeout
        )
        last_error: OSError | None = None

        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(interval)
                try:
                    sock.connect((config.server_host, config.server_port))
                    logger.info(
                        "Docker simulation server is ready on %s:%s",
                        config.server_host,
                        config.server_port,
                    )
                    return
                except OSError as exc:
                    last_error = exc
            time.sleep(interval)

        detail = f" Last socket error: {last_error}" if last_error else ""
        raise TimeoutError(
            "Docker simulation cluster server did not become ready within "
            f"{timeout if timeout is not None else config.server_startup_timeout} seconds."
            f"{detail}"
        )

    @staticmethod
    def build_simulation_cluster_images() -> None:
        logger.info("Building simulation cluster images...")
        with core_directory():
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "--progress",
                        "plain",
                        "build",
                        "--force-rm",
                    ],
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
    def start_simulation_cluster(worker_count: int = 3) -> None:
        """
        Start the Docker-based simulation cluster with the specified worker count.
        """
        logger.info("Starting the simulation cluster with %d workers...", worker_count)

        with core_directory():
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "up",
                        "-d",
                        "--scale",
                        f"worker={worker_count}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.debug("Docker compose output:\n%s", result.stdout)
                SimulationClusterManager.wait_for_server_readiness()
                logger.info("Simulation cluster successfully started.")
                log_docker_logs(logger)
            except subprocess.CalledProcessError as e:
                logger.error("Error starting the simulation cluster:\n%s", e.stderr)
                raise
            except Exception as e:
                logger.error("Error starting the simulation cluster: %s", str(e))
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
def simulation_cluster_context_manager(worker_count: int = 3):
    """
    Context manager for managing the simulation cluster lifecycle.
    """
    logger.info("Entering simulation cluster context.")
    SimulationClusterManager.build_simulation_cluster_images()
    SimulationClusterManager.prune_dangling_images()
    SimulationClusterManager.start_simulation_cluster(worker_count)
    try:
        yield
    finally:
        SimulationClusterManager.shutdown_simulation_cluster()
        logger.info("Exited simulation cluster context.")
