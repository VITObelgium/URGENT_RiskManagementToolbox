import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Sequence

import grpc

from logger import get_logger, log_docker_logs
from services.simulation_service.core.infrastructure.generated import (
    simulation_messaging_pb2 as sm,
)
from services.simulation_service.core.infrastructure.generated import (
    simulation_messaging_pb2_grpc as sm_grpc,
)
from services.simulation_service.core.models import (
    SimulationCase,
    SimulationResults,
    SimulationServiceRequest,
    SimulationServiceResponse,
    WellManagementServiceResult,
)
from services.simulation_service.core.utils.converters import json_to_str, str_to_json

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


class SimulationService:
    _SERVER_HOST = "localhost"
    _SERVER_PORT = 50051
    _WORKER_COUNT = 3

    @staticmethod
    def start_simulation_cluster() -> None:
        """
        Start the Docker-based simulation cluster with the specified worker count.
        """
        logger.info(
            "Attempting to start simulation cluster on host %s, port %d...",
            SimulationService._SERVER_HOST,
            SimulationService._SERVER_PORT,
        )
        with core_directory():
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "up",
                        "-d",
                        "--scale",
                        f"worker={SimulationService._WORKER_COUNT}",
                    ],
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

    @staticmethod
    def process_request(request_dict: dict[str, Any]) -> SimulationServiceResponse:
        """
        Process a simulation service request with provided cases.

        Args:
            request_dict (dict[str, Any]): The request payload.

        Returns:
            SimulationServiceResponse: The response with completed simulation cases.
        """
        logger.debug("Received simulation request: %s", request_dict)
        request = SimulationServiceRequest(**request_dict)

        simulation_cases_request = request.simulation_cases

        logger.info(
            "Processing %d simulation cases on the cluster...",
            len(simulation_cases_request),
        )

        simulation_cases_response = SimulationService._perform_simulations_on_cluster(
            simulation_cases_request
        )

        logger.info("Simulation processing completed.")
        return SimulationServiceResponse(
            simulation_cases=list(simulation_cases_response)
        )

    @staticmethod
    def transfer_simulation_model(simulation_model_archive: bytes | str) -> None:
        """
        Transfer the simulation model archive to the cluster.

        Args:
            simulation_model_archive (bytes | str): The model archive as bytes or a file path.
        """
        logger.info("Transferring simulation model archive...")
        match simulation_model_archive:
            case bytes():
                SimulationService._transfer_model_archive(simulation_model_archive)
            case str():
                path_to_archive = Path(simulation_model_archive)
                SimulationService._transfer_model_archive_from_path(path_to_archive)
            case _:
                raise TypeError(
                    f"Invalid type for model archive, actual {type(simulation_model_archive)}. "
                    "Expected bytes or str."
                )

    @staticmethod
    def _transfer_model_archive(simulation_model_archive: bytes) -> None:
        """
        Send the simulation model archive to the cluster via gRPC.

        Args:
            simulation_model_archive (bytes): The archive content as bytes.
        """
        grpc_target = (
            f"{SimulationService._SERVER_HOST}:{SimulationService._SERVER_PORT}"
        )
        logger.info("Establishing gRPC connection to %s...", grpc_target)

        with grpc.insecure_channel(grpc_target) as channel:
            stub = sm_grpc.SimulationMessagingStub(channel)

            try:
                logger.info("Sending simulation model archive to the cluster...")
                cluster_response = stub.TransferSimulationModel(
                    sm.SimulationModel(package_archive=simulation_model_archive)
                )
                logger.info(
                    "Cluster responded with message: %s", cluster_response.message
                )
            except grpc.RpcError as e:
                logger.error("Error transferring simulation model: %s", e)
                raise

    @staticmethod
    def _transfer_model_archive_from_path(simulation_model_archive: Path) -> None:
        """
        Read a simulation model archive from a file and transfer it to the cluster.

        Args:
            simulation_model_archive (Path): Path to the archive file.
        """
        if not simulation_model_archive.is_file():
            raise FileNotFoundError(
                f"Simulation model file not found at {simulation_model_archive}"
            )

        logger.info(
            "Reading simulation model archive from file: %s", simulation_model_archive
        )
        content = simulation_model_archive.read_bytes()
        SimulationService._transfer_model_archive(content)

    @staticmethod
    def _perform_simulations_on_cluster(
        cases: Sequence[SimulationCase],
    ) -> Sequence[SimulationCase]:
        """
        Perform simulations on the cluster and return the responses.

        Args:
            cases (Sequence[SimulationCase]): The simulation cases to process.

        Returns:
            Sequence[SimulationCase]: The processed simulation cases.
        """
        grpc_target = (
            f"{SimulationService._SERVER_HOST}:{SimulationService._SERVER_PORT}"
        )
        logger.info("Connecting to simulation cluster at %s...", grpc_target)

        with grpc.insecure_channel(grpc_target) as channel:
            stub = sm_grpc.SimulationMessagingStub(channel)

            logger.info("Sending simulation cases to the cluster...")
            simulations_inputs = [SimulationService._to_grpc(case) for case in cases]
            simulations_request = sm.Simulations(simulations=simulations_inputs)

            try:
                cluster_response = stub.PerformSimulations(simulations_request)
                logger.info("Simulations completed on the cluster.")
            except grpc.RpcError as e:
                logger.error("Error performing simulations: %s", e)
                raise

            return [
                SimulationService._from_grpc(sim)
                for sim in cluster_response.simulations
            ]

    @staticmethod
    def _to_grpc(case: SimulationCase) -> sm.Simulation:
        """
        Convert a simulation case to the gRPC format.

        Args:
            case (SimulationCase): The simulation case to convert.

        Returns:
            sm.Simulation: The gRPC-compatible simulation object.
        """
        return sm.Simulation(
            input=sm.SimulationInput(wells=case.wells.model_dump_json()),
            control_vector=sm.SimulationControlVector(
                content=json_to_str(case.control_vector)
            ),
        )

    @staticmethod
    def _from_grpc(simulation: sm.Simulation) -> SimulationCase:
        """
        Convert a gRPC simulation result back to a SimulationCase.

        Args:
            simulation (sm.Simulation): The simulation result in gRPC format.

        Returns:
            SimulationCase: The simulation case object.
        """
        return SimulationCase(
            wells=WellManagementServiceResult(**str_to_json(simulation.input.wells)),
            results=SimulationResults(**str_to_json(simulation.result.result)),
            control_vector=str_to_json(simulation.control_vector.content),
        )


@contextmanager
def simulation_cluster_context_manager():
    """
    Context manager for managing the simulation cluster lifecycle.
    """
    logger.info("Entering simulation cluster context.")
    SimulationService.start_simulation_cluster()
    try:
        yield
    finally:
        SimulationService.shutdown_simulation_cluster()
        logger.info("Exited simulation cluster context.")
