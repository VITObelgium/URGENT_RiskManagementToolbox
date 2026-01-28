from collections.abc import Sequence
from pathlib import Path
from typing import Any

import grpc

from logger import get_logger
from services.simulation_service.core.config import get_simulation_config
from services.simulation_service.core.infrastructure.generated import (
    simulation_messaging_pb2 as sm,
)
from services.simulation_service.core.models import (
    SimulationCase,
    SimulationResults,
    SimulationServiceRequest,
    SimulationServiceResponse,
)
from services.simulation_service.core.service.grpc_stub_manager import GrpcStubManager
from services.simulation_service.core.utils.converters import json_to_str, str_to_json
from services.well_management_service.core.models import WellManagementServiceResponse

logger = get_logger(__name__)


class SimulationService:
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
        _config = get_simulation_config()

        with GrpcStubManager.get_stub(
            _config.server_host,
            _config.server_port,
        ) as stub:
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
        logger.info("Processing %d simulation cases on the cluster...", len(cases))
        _config = get_simulation_config()

        with GrpcStubManager.get_stub(
            _config.server_host,
            _config.server_port,
        ) as stub:
            simulations_inputs = [SimulationService._to_grpc(case) for case in cases]
            simulations_request = sm.Simulations(simulations=simulations_inputs)

            try:
                logger.info(
                    "Performing simulations on the cluster (that may take a while, check server and workers logs for progress)..."
                )
                cluster_response = stub.PerformSimulations(simulations_request)
                logger.info("Simulations completed on the cluster.")
            except grpc.RpcError as e:
                logger.error("Error performing simulations: %s", e)
                raise
            except KeyboardInterrupt:
                logger.warning("Simulation interrupted by user.")
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
        # Handle empty or invalid results (e.g., from failed/timed-out simulations)
        result_str = simulation.result.result
        if result_str:
            result_dict = str_to_json(result_str)
            if not result_dict or "Heat" not in result_dict:
                logger.warning(
                    "Simulation result missing required 'Heat' field, using NaN. Result: %s",
                    result_dict,
                )
                results = SimulationResults(Heat=float("nan"))
            else:
                results = SimulationResults(**result_dict)
        else:
            logger.warning("Simulation has empty result, using NaN")
            results = SimulationResults(Heat=float("nan"))

        return SimulationCase(
            wells=WellManagementServiceResponse(**str_to_json(simulation.input.wells)),
            results=SimulationResults(**str_to_json(simulation.result.result)),
            control_vector=str_to_json(simulation.control_vector.content),
        )
