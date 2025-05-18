import asyncio
import io
import os
import uuid
from zipfile import BadZipFile, ZipFile

import generated.simulation_messaging_pb2 as sm
import generated.simulation_messaging_pb2_grpc as sm_grpc
import grpc.aio
from connectors.common import SimulationResults, SimulationStatus
from connectors.factory import ConnectorFactory
from utils.converters import json_to_str

from logger.u_logger import configure_logger, get_logger

configure_logger()

logger = get_logger(__name__)

channel_options = [
    ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
    ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
]


def _run_simulator(simulation_job) -> [SimulationStatus, SimulationResults]:
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


async def handle_simulation_job(stub, simulation_job, worker_id):
    """
    Handle a single simulation job with proper error handling and logging.

    Args:
        stub: The gRPC stub used for communicating with the server
        simulation_job: The simulation job to be processed
        worker_id: Unique identifier for worker processing the job
    Returns:
        None
    """
    job_id = simulation_job.job_id

    logger.info(f"Worker {worker_id}: Starting simulation {job_id}...")
    logger.debug(f"Worker {worker_id}: Simulation job: {simulation_job}")

    simulation_status, simulation_result = await asyncio.to_thread(
        _run_simulator, simulation_job
    )

    if simulation_status == SimulationStatus.SUCCESS:
        status = sm.JobStatus.SUCCESS
        logger.info(f"Worker {worker_id}: Simulation {job_id} completed successfully")
    elif simulation_status == SimulationStatus.TIMEOUT:
        status = sm.JobStatus.TIMEOUT
        logger.warning(f"Worker {worker_id}: Simulation {job_id} timed out")
    elif simulation_status == SimulationStatus.FAILED:
        status = sm.JobStatus.ERROR
        logger.error(f"Worker {worker_id}: Simulation {job_id} failed")
    else:
        status = sm.JobStatus.ERROR
        logger.error(f"Worker {worker_id}: Simulation {job_id} failed")

    response = await submit_simulation_job(
        stub, simulation_job, simulation_result, status
    )
    logger.debug(
        f"Worker {worker_id}: Job {job_id} result submitted. Server response: {response.message}"
    )


async def try_unpacking_model_archive(package_archive) -> bool:
    """
    Unpack the simulation model archive to the application directory.
    Assumes the extract path already exists.

    Args:
        package_archive: Binary content of the simulation model archive

    Returns:
        bool: True if unpacking was successful, False otherwise
    """
    extract_path = "/app"

    # Validate inputs before attempting extraction
    if not package_archive:
        logger.error("Cannot unpack empty archive")
        return False

    try:
        # Use a context manager for the BytesIO object for proper resource management
        with io.BytesIO(package_archive) as archive_data:
            with ZipFile(archive_data, "r") as simulation_model_archive:
                # Validate the archive structure before extraction
                if any(
                    name.startswith("/") or ".." in name
                    for name in simulation_model_archive.namelist()
                ):
                    logger.error("Archive contains potentially unsafe paths")
                    return False

                # Extract all files (assumes extract_path exists)
                simulation_model_archive.extractall(extract_path)

        return True

    except BadZipFile:
        logger.error("Invalid zip archive format")
        return False
    except PermissionError:
        logger.error(f"Permission denied when extracting to {extract_path}")
        return False
    except FileNotFoundError:
        logger.error(f"Extraction path {extract_path} does not exist")
        return False
    except Exception as e:
        logger.error(f"Unpacking failed due to: {str(e)}")
        return False


async def ask_for_simulation_model(stub, worker_id: str) -> None:
    """
    Request and unpack simulation model from the server.
    Will keep retrying until successful.

    Args:
        stub: The gRPC stub to use for communication
        worker_id: Unique identifier for this worker
    """
    RETRY_DELAY = 5  # Seconds to wait between retries

    while True:
        try:
            logger.info(f"Worker {worker_id}: Requesting simulation model archive...")
            simulation_model = await request_simulation_model(stub, worker_id)

            if simulation_model.status == sm.ModelStatus.NO_MODEL_AVAILABLE:
                logger.info(
                    f"Worker {worker_id}: No simulation model available. Retrying in {RETRY_DELAY} seconds..."
                )
                await asyncio.sleep(RETRY_DELAY)
                continue

            if simulation_model.status != sm.ModelStatus.ON_SERVER:
                logger.warning(
                    f"Worker {worker_id}: Received unexpected model status: {simulation_model.status}. Retrying..."
                )
                await asyncio.sleep(RETRY_DELAY)
                continue

            logger.info(f"Worker {worker_id}: Received simulation model archive.")

            if await try_unpacking_model_archive(simulation_model.package_archive):
                logger.info(
                    f"Worker {worker_id}: Unpacking simulation model archive successful."
                )
                return  # Successfully retrieved and unpacked model
            else:
                logger.warning(
                    f"Worker {worker_id}: Corrupted simulation model archive. Retrying in {RETRY_DELAY} seconds..."
                )
                await asyncio.sleep(RETRY_DELAY)

        except grpc.RpcError as e:
            logger.error(
                f"Worker {worker_id}: Unable to connect to server due to {e}. Retrying in {RETRY_DELAY} seconds..."
            )
            await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(
                f"Worker {worker_id}: Unexpected error while requesting model: {str(e)}. Retrying in {RETRY_DELAY} seconds..."
            )
            await asyncio.sleep(RETRY_DELAY)


async def run_simulation_loop(stub, worker_id) -> None:
    """
    Main simulation processing loop that continually requests and processes jobs.

    Args:
        stub: The gRPC stub for server communication
        worker_id: Unique identifier for this worker
    """
    retry_delay = 5  # Fixed delay between retries in seconds

    while True:
        try:
            logger.info(f"Worker {worker_id}: Requesting a job...")
            simulation_job = await request_simulation_job(stub, worker_id)

            # Handle different job statuses
            if simulation_job.status == sm.JobStatus.NO_JOB_AVAILABLE:
                logger.info(
                    f"Worker {worker_id}: No simulation jobs available. Retrying in {retry_delay} seconds..."
                )
                await asyncio.sleep(retry_delay)
                continue

            elif simulation_job.status == sm.JobStatus.ERROR:
                logger.error(
                    f"Worker {worker_id}: Server returned an error for the job request. Retrying..."
                )
                await asyncio.sleep(retry_delay)
                continue

            elif simulation_job.status == sm.JobStatus.JOBSTATUS_UNSPECIFIED:
                logger.warning(
                    f"Worker {worker_id}: Received unspecified status. Retrying..."
                )
                await asyncio.sleep(retry_delay)
                continue

            # Process the job if a valid one is received
            logger.info(
                f"Worker {worker_id}: Processing job {simulation_job.job_id}..."
            )
            await handle_simulation_job(stub, simulation_job, worker_id)

        except grpc.RpcError as e:
            logger.error(
                f"Worker {worker_id}: gRPC error: {str(e)}. Retrying in {retry_delay} seconds..."
            )
            await asyncio.sleep(retry_delay)

        except Exception as e:
            logger.exception(f"Worker {worker_id}: Unexpected error: {str(e)}")
            await asyncio.sleep(retry_delay)


async def create_channel():
    """Create and return a gRPC channel to the server."""
    server_host = os.environ.get("SERVER_HOST", "localhost")
    server_port = os.environ.get("SERVER_PORT", "50051")
    grpc_target = f"{server_host}:{server_port}"

    return grpc.aio.insecure_channel(grpc_target, options=channel_options)


async def main():
    worker_id = str(uuid.uuid4().hex)[:8]

    # Create a single channel that will be reused for all gRPC requests
    async with await create_channel() as channel:
        # Create a stub using the shared channel
        stub = sm_grpc.SimulationMessagingStub(channel)

        # Use the same stub for both simulation model retrieval and job processing
        await ask_for_simulation_model(stub, worker_id)
        await run_simulation_loop(stub, worker_id)


if __name__ == "__main__":
    asyncio.run(main())
