import asyncio
import io
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple
from zipfile import BadZipFile, ZipFile

import grpc.aio

import services.simulation_service.core.infrastructure.generated.simulation_messaging_pb2 as sm
import services.simulation_service.core.infrastructure.generated.simulation_messaging_pb2_grpc as sm_grpc
from logger import get_logger
from services.simulation_service.core.connectors.common import (
    SimulationResults,
    SimulationStatus,
)
from services.simulation_service.core.connectors.factory import ConnectorFactory
from services.simulation_service.core.utils.converters import json_to_str

logger = get_logger("worker")

channel_options = [
    ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
    ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
]

# TODO: refactor or store in config
MODEL_RETRY_DELAY_SEC = 5
JOB_RETRY_DELAY_SEC = 5


async def _sleep_with_stop(
    delay: float, stop_flag: threading.Event | None = None, granularity: float = 0.1
) -> None:
    """Sleep up to `delay` seconds but return early if `stop_flag` is set.

    Uses small async sleep slices to remain responsive during shutdown.
    """
    if delay <= 0:
        return
    if stop_flag is None:
        await asyncio.sleep(delay)
        return
    remaining = float(delay)
    step = max(0.01, float(granularity))
    while remaining > 0:
        if stop_flag.is_set():
            return
        await asyncio.sleep(min(step, remaining))
        remaining -= step


def _compute_worker_temp_dir(worker_id: str | int) -> Path:
    """Compute the absolute temp directory path for a given worker.

    Accept both string and integer worker identifiers, since gRPC uses strings for IDs
    while local paths only need a stable textual representation.
    """
    repo_root = Path(__file__).resolve().parents[7]
    return repo_root / f"orchestration_files/.worker_{str(worker_id)}_temp"


def _get_shared_venv_dir() -> Path:
    """Return the shared venv directory under orchestration_files.

    All workers on this host will share this environment to reduce setup time.
    """
    repo_root = Path(__file__).resolve().parents[7]
    return repo_root / "orchestration_files/.venv_darts"


def _venv_python_path(venv_dir: Path) -> Path:
    # Linux/macOS layout
    return venv_dir / "bin/python"


def _acquire_venv_lock(venv_dir: Path, timeout: float = 600.0) -> bool:
    """Best-effort file lock to avoid concurrent venv creation/installation.

    Creates a lock directory atomically. If already exists, wait for ready marker or
    the lock to be released up to timeout seconds.
    """
    lock_dir = venv_dir / ".venv_lock"
    ready_marker = venv_dir / ".venv_ready"
    try:
        lock_dir.mkdir(parents=True, exist_ok=False)
        return True
    except FileExistsError:
        # Someone else is preparing the venv; wait for completion
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            if ready_marker.exists() and _venv_python_path(venv_dir).exists():
                return False
            if not lock_dir.exists():
                return False
            time.sleep(0.5)
        return False


def _release_venv_lock(venv_dir: Path) -> None:
    lock_dir = venv_dir / ".venv_lock"
    try:
        if lock_dir.exists():
            shutil.rmtree(lock_dir, ignore_errors=True)
    except Exception:
        pass


def _prepare_shared_venv(worker_id: str | int) -> Path | None:
    """Create or reuse a shared Python venv and install model requirements.

    - Else tries python3.10, else current interpreter.
        - Installs requirements from the shared worker requirements file under
            src/services/simulation_service/core/infrastructure/worker/worker_requirements.txt
            so that local relative paths (like the OpenDarts wheel) resolve correctly.
    Returns the path to the venv python if successful, else None.
    """
    venv_dir = _get_shared_venv_dir()
    venv_dir.mkdir(parents=True, exist_ok=True)
    venv_python = _venv_python_path(venv_dir)
    ready_marker = venv_dir / ".venv_ready"

    # If already prepared, return venv path
    if venv_python.exists() and ready_marker.exists():
        logger.info("Worker %s: Using existing shared venv at %s", worker_id, venv_dir)
        return venv_python

    got_lock = _acquire_venv_lock(venv_dir)
    try:
        # Re-check after acquiring/contending for the lock
        if venv_python.exists() and ready_marker.exists():
            return venv_python

        # Prefer python3.10 if available (for DARTS compatibility), else fallback
        base = (
            shutil.which("python3.10")
            or sys.executable
            or shutil.which("python3")
            or "python3"
        )

        logger.info(
            "Worker %s: Preparing shared venv using base interpreter: %s",
            worker_id,
            base,
        )

        if not venv_python.exists():
            try:
                subprocess.run([base, "-m", "venv", str(venv_dir)], check=True)
            except Exception as e:
                logger.error("Worker %s: Failed to create venv: %s", worker_id, e)
                return None

        try:
            subprocess.run(
                [
                    str(venv_python),
                    "-m",
                    "pip",
                    "install",
                    "-U",
                    "pip",
                    "setuptools",
                    "wheel",
                ],
                check=True,
            )
        except Exception as e:
            logger.warning(
                "Worker %s: Failed to upgrade pip tooling in venv (%s); proceeding.",
                worker_id,
                e,
            )

        req_path = Path(__file__).resolve().parents[1] / "worker_requirements.txt"
        logger.info("Worker %s: Installing requirements from %s", worker_id, req_path)
        install_cwd = str(req_path.parent)

        logger.debug(
            "Worker %s: Running pip install with cwd=%s", worker_id, install_cwd
        )
        try:
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "-r", str(req_path)],
                check=True,
                cwd=install_cwd,
            )
        except Exception as e:
            logger.error(
                "Worker %s: pip install -r %s failed: %s", worker_id, req_path, e
            )
            return None

        # Quick validation step of darts importability
        try:
            test = subprocess.run(
                [str(venv_python), "-c", "import darts, sys; print(sys.version)"],
                capture_output=True,
                text=True,
            )
            if test.returncode != 0:
                logger.error(
                    "Worker %s: DARTS not importable in venv; stderr: %s",
                    worker_id,
                    test.stderr[-500:],
                )
                return None
        except Exception as e:
            logger.error("Worker %s: Failed to validate venv import: %s", worker_id, e)
            return None

        try:
            ready_marker.touch(exist_ok=True)
        except Exception:
            pass

        logger.info("Worker %s: Shared venv ready at %s", worker_id, venv_dir)
        return venv_python
    finally:
        if got_lock:
            _release_venv_lock(venv_dir)


def _run_simulator(
    simulation_job, stop_flag: threading.Event | None = None
) -> Tuple[SimulationStatus, SimulationResults]:
    connector = ConnectorFactory.get_connector(simulation_job.simulator)
    return connector.run(simulation_job.simulation.input.wells, stop=stop_flag)


async def request_simulation_job(stub, worker_id):
    """Request a simulation job from the server."""
    return await stub.RequestSimulationJob(sm.RequestJob(worker_id=worker_id))


async def request_simulation_model(stub, worker_id: str):
    """Request a simulation model archive from server."""
    return await stub.RequestSimulationModel(sm.RequestModel(worker_id=worker_id))


async def submit_simulation_job(stub, simulation_job, simulation_result, status):
    """Submit the simulation result back to the server."""
    simulation_result_as_string = json_to_str(simulation_result)

    simulation_job.simulation.result.result = simulation_result_as_string
    simulation_job.status = status

    return await stub.SubmitSimulationJob(simulation_job)


async def handle_simulation_job(
    stub, simulation_job, worker_id, stop_flag: threading.Event | None = None
):
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

    def _run_simulator_with_logging(sim_job, wid, stop):
        """Wrapper to ensure logs emitted inside to_thread map to worker file.

        We temporarily set the current thread name to match the worker thread
        (e.g., "worker-3"). Our per-thread file handlers in process logger are
        filtered by threadName, so this routes logs from the simulator and any
        nested libraries into the correct `log/worker_{wid}.log` file.
        """
        try:
            th = threading.current_thread()
            old_name = th.name
            th.name = f"worker-{wid}"
        except Exception:
            old_name = None
        try:
            return _run_simulator(sim_job, stop)
        finally:
            if old_name is not None:
                try:
                    threading.current_thread().name = old_name
                except Exception:
                    pass

    with worker_file_context(worker_id):
        simulation_status, simulation_result = await asyncio.to_thread(
            _run_simulator_with_logging, simulation_job, worker_id, stop_flag
        )

    if simulation_status == SimulationStatus.SUCCESS:
        status = sm.JobStatus.SUCCESS
        logger.info(f"Worker {worker_id}: Simulation {job_id} completed successfully")
    elif simulation_status == SimulationStatus.TIMEOUT:
        status = sm.JobStatus.TIMEOUT
        logger.warning(f"Worker {worker_id}: Simulation {job_id} timed out")
    elif simulation_status == SimulationStatus.FAILED:
        status = sm.JobStatus.FAILED
        logger.error(f"Worker {worker_id}: Simulation {job_id} failed")
    else:
        status = sm.JobStatus.ERROR
        logger.error(
            f"Worker {worker_id}: Simulation {job_id} encountered an unknown status: {simulation_status}"
        )

    response = await submit_simulation_job(
        stub, simulation_job, simulation_result, status
    )
    logger.debug(
        f"Worker {worker_id}: Job {job_id} result submitted. Server response: {response.message}"
    )


async def try_unpacking_model_archive(
    package_archive: bytes, worker_id: str | int
) -> bool:
    """
    Unpack the simulation model archive to the application directory.
    Ensures the extract path exists.

    Args:
        package_archive: Binary content of the simulation model archive

    Returns:
        bool: True if unpacking was successful, False otherwise
    """
    extract_path = Path(
        os.environ.get("SIM_MODEL_DIR") or _compute_worker_temp_dir(worker_id)
    )
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

                extract_path.mkdir(parents=True, exist_ok=True)
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


async def ask_for_simulation_model(
    stub, worker_id: str, stop_flag: threading.Event | None = None
) -> None:
    """
    Request and unpack simulation model from the server.
    Will keep retrying until successful.

    Args:
        stub: The gRPC stub to use for communication
        worker_id: Unique identifier for this worker
    """
    while True:
        if stop_flag is not None and stop_flag.is_set():
            logger.info(f"Worker {worker_id}: Stop requested during model acquisition.")
            return
        try:
            logger.info(f"Worker {worker_id}: Requesting simulation model archive...")
            simulation_model = await request_simulation_model(stub, worker_id)

            if simulation_model.status == sm.ModelStatus.NO_MODEL_AVAILABLE:
                logger.info(
                    f"Worker {worker_id}: No simulation model available. Retrying in {MODEL_RETRY_DELAY_SEC} seconds..."
                )
                await _sleep_with_stop(MODEL_RETRY_DELAY_SEC, stop_flag)
                if stop_flag is not None and stop_flag.is_set():
                    return
                continue

            if simulation_model.status != sm.ModelStatus.ON_SERVER:
                logger.warning(
                    f"Worker {worker_id}: Received unexpected model status: {simulation_model.status}. Retrying in {MODEL_RETRY_DELAY_SEC} seconds..."
                )
                await _sleep_with_stop(MODEL_RETRY_DELAY_SEC, stop_flag)
                if stop_flag is not None and stop_flag.is_set():
                    return
                continue

            logger.info(f"Worker {worker_id}: Received simulation model archive.")

            if await try_unpacking_model_archive(
                simulation_model.package_archive, worker_id
            ):
                logger.info(
                    f"Worker {worker_id}: Unpacking simulation model archive successful."
                )

                # Prepare or reuse a shared Python venv and set interpreter for DARTS
                try:
                    venv_python = await asyncio.to_thread(
                        _prepare_shared_venv, worker_id
                    )
                    if venv_python:
                        logger.info(
                            "Worker %s: Using venv python at %s for DARTS runs.",
                            worker_id,
                            venv_python,
                        )
                    else:
                        logger.warning(
                            "Worker %s: Proceeding without venv; DARTS may fail if dependencies are missing.",
                            worker_id,
                        )
                except Exception as e:
                    logger.warning(
                        "Worker %s: Unexpected error while preparing venv: %s",
                        worker_id,
                        e,
                    )
                return  # Successfully retrieved and unpacked model
            else:
                logger.warning(
                    f"Worker {worker_id}: Corrupted simulation model archive. Retrying in {MODEL_RETRY_DELAY_SEC} seconds..."
                )
                await _sleep_with_stop(MODEL_RETRY_DELAY_SEC, stop_flag)
                if stop_flag is not None and stop_flag.is_set():
                    return

        except grpc.RpcError as e:
            logger.error(
                f"Worker {worker_id}: Unable to connect to server due to {e}. Retrying in {MODEL_RETRY_DELAY_SEC} seconds..."
            )
            await _sleep_with_stop(MODEL_RETRY_DELAY_SEC, stop_flag)
            if stop_flag is not None and stop_flag.is_set():
                return
        except Exception as e:
            logger.warning(
                f"Worker {worker_id}: Unexpected error while requesting model: {str(e)}. Retrying in {MODEL_RETRY_DELAY_SEC} seconds..."
            )
            await _sleep_with_stop(MODEL_RETRY_DELAY_SEC, stop_flag)
            if stop_flag is not None and stop_flag.is_set():
                return


async def run_simulation_loop(
    stub, worker_id: str, stop_flag: threading.Event | None = None
) -> None:
    """
    Main simulation processing loop that continually requests and processes jobs.

    Args:
        stub: The gRPC stub for server communication
        worker_id: Unique identifier for this worker
    """
    retry_delay = JOB_RETRY_DELAY_SEC  # Fixed delay between retries in seconds

    while True:
        if stop_flag is not None and stop_flag.is_set():
            logger.info(f"Worker {worker_id}: Stop requested, exiting loop.")
            break
        try:
            logger.info(f"Worker {worker_id}: Requesting a job...")
            simulation_job = await request_simulation_job(stub, worker_id)

            # Handle different job statuses
            if simulation_job.status == sm.JobStatus.NO_JOB_AVAILABLE:
                logger.info(
                    f"Worker {worker_id}: No simulation jobs available. Retrying in {retry_delay} seconds..."
                )
                await _sleep_with_stop(retry_delay, stop_flag)
                if stop_flag is not None and stop_flag.is_set():
                    break
                continue

            elif simulation_job.status == sm.JobStatus.ERROR:
                logger.error(
                    f"Worker {worker_id}: Server returned an error for the job request. Retrying..."
                )
                await _sleep_with_stop(retry_delay, stop_flag)
                if stop_flag is not None and stop_flag.is_set():
                    break
                continue

            elif simulation_job.status == sm.JobStatus.JOBSTATUS_UNSPECIFIED:
                logger.warning(
                    f"Worker {worker_id}: Received unspecified status. Retrying..."
                )
                await _sleep_with_stop(retry_delay, stop_flag)
                if stop_flag is not None and stop_flag.is_set():
                    break
                continue

            # Process the job if a valid one is received
            logger.info(
                f"Worker {worker_id}: Processing job {simulation_job.job_id}..."
            )
            await handle_simulation_job(stub, simulation_job, worker_id, stop_flag)

        except grpc.RpcError as e:
            logger.error(
                f"Worker {worker_id}: gRPC error: {str(e)}. Retrying in {retry_delay} seconds..."
            )
            await _sleep_with_stop(retry_delay, stop_flag)
            if stop_flag is not None and stop_flag.is_set():
                break

        except Exception as e:
            logger.exception(f"Worker {worker_id}: Unexpected error: {str(e)}")
            await _sleep_with_stop(retry_delay, stop_flag)
            if stop_flag is not None and stop_flag.is_set():
                break


def create_channel():
    """Create and return a gRPC channel to the server."""
    server_host = os.environ.get("SERVER_HOST", "localhost")
    server_port = os.environ.get("SERVER_PORT", "50051")
    grpc_target = f"{server_host}:{server_port}"

    return grpc.aio.insecure_channel(grpc_target, options=channel_options)


async def main(stop_flag: threading.Event | None = None, worker_id: str | None = None):
    worker_id = worker_id or str(uuid.uuid4().hex)[:8]

    loop = asyncio.get_running_loop()
    _warned = {"done": False}

    def _ignore_poller_eagain(loop, context):
        exc = context.get("exception")
        message = context.get("message", "")
        if isinstance(exc, BlockingIOError) and getattr(exc, "errno", None) == 11:
            if not _warned["done"]:
                logger.debug(
                    "Worker %s: suppressing gRPC AIO poller EAGAIN noise (BlockingIOError 11)",
                    worker_id,
                )
                _warned["done"] = True
            return
        if "PollerCompletionQueue._handle_events" in message:
            if not _warned["done"]:
                logger.debug(
                    "Worker %s: suppressing gRPC AIO poller _handle_events noise via message",
                    worker_id,
                )
                _warned["done"] = True
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_ignore_poller_eagain)

    # Create a single channel that will be reused for all gRPC requests
    async with create_channel() as channel:
        # Create a stub using the shared channel
        stub = sm_grpc.SimulationMessagingStub(channel)

        # Use the same stub for both simulation model retrieval and job processing
        # gRPC RequestModel expects a string worker_id; keep as string
        await ask_for_simulation_model(stub, worker_id, stop_flag)
        await run_simulation_loop(stub, worker_id, stop_flag)


if __name__ == "__main__":
    asyncio.run(main())


@contextmanager
def worker_file_context(worker_id: str | int):
    """
    Context manager to create and clean up a temporary directory for worker files.
    Args:
        worker_id (int): Unique identifier for the worker to create a distinct directory.
    """
    prev_cwd = Path.cwd()
    temp_dir = _compute_worker_temp_dir(worker_id)

    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(temp_dir)
        logger.info(f"Worker {worker_id}: Changed working directory to {temp_dir}")
        yield
    finally:
        os.chdir(prev_cwd)
