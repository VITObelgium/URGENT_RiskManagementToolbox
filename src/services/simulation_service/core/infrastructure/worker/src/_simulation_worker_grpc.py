import asyncio
import io
import os
import uuid
from zipfile import ZipFile

import generated.simulation_messaging_pb2 as sm
import generated.simulation_messaging_pb2_grpc as sm_grpc
import grpc.aio
from connectors.common import SimulationResults
from connectors.factory import ConnectorFactory
from utils.converters import json_to_str

from logger.u_logger import configure_logger, get_logger

configure_logger()

logger = get_logger(__name__)

channel_options = [
    ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
    ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
]


def _run_simulator(simulation_job) -> SimulationResults:
    logger.info(f"Simulator type: {simulation_job.simulator}")
    connector = ConnectorFactory.get_connector(simulation_job.simulator)
    return connector.run(simulation_job.simulation.input.wells)


async def request_simulation_job(stub, worker_id):
    """Request a simulation job from the server."""
    return await stub.RequestSimulationJob(sm.RequestJob(worker_id=worker_id))


async def request_simulation_model(stub, worker_id):
    """Request a simulation model archive from server."""
    return await stub.RequestSimulationModel(sm.RequestModel(worker_id=worker_id))


async def submit_simulation_job(stub, simulation_job, simulation_result, status):
    """Submit the simulation result back to the server."""
    simulation_result_as_string = json_to_str(simulation_result)

    simulation_job.simulation.result.result = simulation_result_as_string
    simulation_job.status = status

    return await stub.SubmitSimulationJob(simulation_job)


async def handle_simulation_job(stub, simulation_job):
    """Handle a single simulation job."""
    try:
        logger.info("Starting simulation....")
        # Run the simulator in a thread-safe, non-blocking way
        simulation_result = await asyncio.to_thread(_run_simulator, simulation_job)
        status = sm.JobStatus.SUCCESS
        logger.info("Simulation completed successfully.")
    except Exception as e:
        logger.error(f"Simulation failed due to: {e}")
        simulation_result = {"error": str(e)}  # Use a meaningful result for failure
        status = sm.JobStatus.FAILED

    response = await submit_simulation_job(
        stub, simulation_job, simulation_result, status
    )
    logger.info(
        f"Worker {simulation_job.worker_id}: Received server acknowledgment: {response.message}"
    )


async def try_unpacking_model_archive(package_archive) -> bool:
    extract_path = "/app"
    try:
        with ZipFile(io.BytesIO(package_archive), "r") as simulation_model_archive:
            simulation_model_archive.extractall(extract_path)
        return True
    except Exception as e:
        logger.error(f"Unpacking failed due to: {e}")
        return False


async def ask_for_simulation_model(worker_id: str) -> None:
    has_simulation_model = False
    while not has_simulation_model:
        try:
            server_host = os.environ.get("SERVER_HOST", "localhost")
            server_port = os.environ.get("SERVER_PORT", "50051")

            grpc_target = f"{server_host}:{server_port}"
            async with grpc.aio.insecure_channel(
                grpc_target, options=channel_options
            ) as channel:
                stub = sm_grpc.SimulationMessagingStub(channel)
                logger.info(
                    f"Worker {worker_id}: Requesting a simulation model archive..."
                )
                simulation_model = await request_simulation_model(stub, worker_id)

                if simulation_model.status == sm.ModelStatus.NO_MODEL_AVAILABLE:
                    logger.info(
                        f"Worker {worker_id}: No simulation model available. Retrying in 5 seconds..."
                    )
                    await asyncio.sleep(5)  # Non-blocking sleep before retrying
                    continue

                if simulation_model.status == sm.ModelStatus.ON_SERVER:
                    logger.info(
                        f"Worker {worker_id}: Received simulation model archive."
                    )

                    if not await try_unpacking_model_archive(
                        simulation_model.package_archive
                    ):
                        logger.warning(
                            f"Worker {worker_id}: Corrupted simulation model archive. Retrying in 5 seconds..."
                        )
                        await asyncio.sleep(5)  # Non-blocking sleep before retrying

                    else:
                        logger.info(
                            f"Worker {worker_id}: Unpacking simulation model archive successful."
                        )
                        has_simulation_model = True

        except grpc.RpcError as e:
            # Handle gRPC connection errors
            logger.error(
                f"Worker {worker_id}: Unable to connect to server due to {e}. Retrying in 5 seconds..."
            )
            await asyncio.sleep(5)


async def run_simulation_loop(worker_id: str) -> None:
    while True:
        try:
            server_host = os.environ.get("SERVER_HOST", "localhost")
            server_port = os.environ.get("SERVER_PORT", "50051")

            grpc_target = f"{server_host}:{server_port}"
            async with grpc.aio.insecure_channel(
                grpc_target, options=channel_options
            ) as channel:
                stub = sm_grpc.SimulationMessagingStub(channel)

                logger.info(f"Worker {worker_id}: Requesting a job...")
                simulation_job = await request_simulation_job(stub, worker_id)

                # Check the job's status and react accordingly
                if simulation_job.status == sm.JobStatus.NO_JOB_AVAILABLE:
                    logger.info(
                        f"Worker {worker_id}: No simulation jobs available. Retrying in 5 seconds..."
                    )
                    await asyncio.sleep(5)  # Non-blocking sleep before retrying
                    continue
                elif simulation_job.status == sm.JobStatus.ERROR:
                    logger.error(
                        f"Worker {worker_id}: Server returned an error for the job request."
                    )
                    break
                elif simulation_job.status == sm.JobStatus.JOBSTATUS_UNSPECIFIED:
                    logger.info(
                        f"Worker {worker_id}: Received unspecified status. Ignoring this job."
                    )
                    break

                # Process the job if a valid one is received
                logger.info(f"Worker {worker_id}: Received new job {simulation_job}")
                await handle_simulation_job(stub, simulation_job)

        except grpc.RpcError as e:
            # Handle gRPC connection errors
            logger.error(
                f"Worker {worker_id}: Unable to connect to server due to {e}. Retrying in 5 seconds..."
            )
            await asyncio.sleep(5)


async def main():
    worker_id = uuid.uuid4().hex
    await ask_for_simulation_model(worker_id)
    await asyncio.gather(run_simulation_loop(worker_id))


if __name__ == "__main__":
    asyncio.run(main())
