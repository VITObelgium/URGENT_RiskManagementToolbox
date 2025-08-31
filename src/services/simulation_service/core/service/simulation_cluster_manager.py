import subprocess
from contextlib import contextmanager

from logger import get_logger, log_docker_logs
from services.simulation_service.core.service.grpc_stub_manager import GrpcStubManager
from services.simulation_service.core.service.utils import core_directory

logger = get_logger(__name__)


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
    def start_simulation_cluster(worker_count: int = 3) -> None:
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
def simulation_cluster_context_manager(worker_count: int = 3):
    """
    Context manager for managing the simulation cluster lifecycle.
    """
    logger.info("Entering simulation cluster context...")
    SimulationClusterManager.build_simulation_cluster_images()
    SimulationClusterManager.prune_dangling_images()
    SimulationClusterManager.start_simulation_cluster(worker_count)
    try:
        yield
    finally:
        SimulationClusterManager.shutdown_simulation_cluster()
        logger.info("Exited simulation cluster context.")
