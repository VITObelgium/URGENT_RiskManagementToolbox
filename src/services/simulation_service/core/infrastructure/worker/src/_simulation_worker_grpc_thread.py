from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import grpc.aio

import services.simulation_service.core.infrastructure.generated.simulation_messaging_pb2 as sm
import services.simulation_service.core.infrastructure.generated.simulation_messaging_pb2_grpc as sm_grpc
from logger import get_logger
from services.simulation_service.core.config import get_simulation_config
from services.simulation_service.core.connectors.common import (
    SimulationResults,
    SimulationStatus,
)
from services.simulation_service.core.connectors.factory import ConnectorFactory
from services.simulation_service.core.infrastructure.worker.src.utils import (
    compute_worker_temp_dir,
    sleep_with_stop,
)
from services.simulation_service.core.utils.converters import json_to_str

logger = get_logger("threading-worker", filename=__name__)

MODEL_RETRY_DELAY_SEC = 5
JOB_RETRY_DELAY_SEC = 5


def _is_expected_shutdown_rpc_error(e: BaseException) -> bool:
    try:
        if isinstance(e, grpc.aio.AioRpcError):
            code = e.code()
            return code in (
                grpc.aio.StatusCode.UNAVAILABLE,
                grpc.aio.StatusCode.CANCELLED,
            )
    except Exception:
        pass
    msg = str(e)
    return "Connection refused" in msg or "failed to connect to all addresses" in msg


def _run_simulator(
    simulation_job,
    stop_flag: threading.Event | None = None,
    *,
    working_dir: Path | None = None,
) -> tuple[SimulationStatus, SimulationResults]:
    connector = ConnectorFactory.get_connector(simulation_job.simulator)

    user_cost_function = json.loads(simulation_job.simulation.result.result)

    tf = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
    try:
        json.dump(simulation_job.simulation.input.wells, tf)
        tf.flush()
        tf_path = tf.name
        tf.close()

        env_patch: dict[str, str] = {}
        if working_dir is not None:
            env_patch["WORKER_CWD"] = str(working_dir)

        old_env = {k: os.environ.get(k) for k in env_patch}
        os.environ.update(env_patch)
        try:
            return connector.run(tf_path, user_cost_function, stop=stop_flag)
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    finally:
        try:
            os.unlink(tf_path)
        except OSError:
            pass


async def request_simulation_job(stub, worker_id):
    return await stub.RequestSimulationJob(sm.RequestJob(worker_id=worker_id))


async def request_simulation_model(stub, worker_id: str):
    return await stub.RequestSimulationModel(sm.RequestModel(worker_id=worker_id))


async def submit_simulation_job(stub, simulation_job, simulation_result, status):
    simulation_result_as_string = json_to_str(simulation_result)
    simulation_job.simulation.result.result = simulation_result_as_string
    simulation_job.status = status
    return await stub.SubmitSimulationJob(simulation_job)


async def handle_simulation_job(
    stub, simulation_job, worker_id: str, stop_flag: threading.Event | None = None
):
    job_id = simulation_job.job_id
    working_dir = compute_worker_temp_dir(worker_id)
    working_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Worker %s: Starting simulation %s...", worker_id, job_id)
    logger.debug("Worker %s: Simulation job: %s", worker_id, simulation_job)

    def _run_in_thread():
        th = threading.current_thread()
        old_name = th.name
        th.name = f"worker-{worker_id}"
        try:
            return _run_simulator(simulation_job, stop_flag, working_dir=working_dir)
        finally:
            th.name = old_name

    simulation_status, simulation_result = await asyncio.to_thread(_run_in_thread)

    _STATUS_MAP = {
        SimulationStatus.SUCCESS: (
            sm.JobStatus.SUCCESS,
            logger.info,
            "completed successfully",
        ),
        SimulationStatus.TIMEOUT: (sm.JobStatus.TIMEOUT, logger.warning, "timed out"),
        SimulationStatus.FAILED: (sm.JobStatus.FAILED, logger.error, "failed"),
        SimulationStatus.EXCEPTION: (
            sm.JobStatus.EXCEPTION,
            logger.exception,
            "encountered an exception",
        ),
    }
    status, log_fn, msg = _STATUS_MAP.get(
        simulation_status,
        (
            sm.JobStatus.ERROR,
            logger.error,
            f"encountered unknown status: {simulation_status}",
        ),
    )
    log_fn("Worker %s: Simulation %s %s", worker_id, job_id, msg)

    try:
        response = await submit_simulation_job(
            stub, simulation_job, simulation_result, status
        )
        logger.debug(
            "Worker %s: Job %s result submitted. Server response: %s",
            worker_id,
            job_id,
            response.message,
        )
    except grpc.aio.AioRpcError as e:
        if (
            stop_flag is not None and stop_flag.is_set()
        ) or _is_expected_shutdown_rpc_error(e):
            logger.info(
                "Worker %s: submit failed because server is shutting down (code=%s). Exiting job handler.",
                worker_id,
                getattr(e, "code", lambda: None)(),
            )
            return
        logger.error("Worker %s: submit failed due to gRPC error: %s", worker_id, e)
        raise


async def try_unpacking_model_archive(
    package_archive: bytes, worker_id: str | int
) -> bool:
    extract_path = Path(
        os.environ.get("SIM_MODEL_DIR") or compute_worker_temp_dir(worker_id)
    )

    if not package_archive:
        logger.error("Cannot unpack empty archive")
        return False

    def _extract() -> bool:
        try:
            with io.BytesIO(package_archive) as archive_data:
                with ZipFile(archive_data, "r") as zf:
                    if any(
                        name.startswith("/") or ".." in name for name in zf.namelist()
                    ):
                        logger.error("Archive contains potentially unsafe paths")
                        return False
                    extract_path.mkdir(parents=True, exist_ok=True)
                    zf.extractall(extract_path)
            return True
        except BadZipFile:
            logger.error("Invalid zip archive format")
            return False
        except PermissionError:
            logger.error("Permission denied when extracting to %s", extract_path)
            return False
        except FileNotFoundError:
            logger.error("Extraction path %s does not exist", extract_path)
            return False
        except Exception as e:
            logger.error("Unpacking failed due to: %s", e)
            return False

    return await asyncio.to_thread(_extract)


async def _retry_with_stop(
    delay: float,
    stop_flag: threading.Event | None,
    worker_id: str,
    context: str,
) -> bool:
    await sleep_with_stop(delay, stop_flag)
    if stop_flag is not None and stop_flag.is_set():
        logger.info("Worker %s: Stop requested during %s.", worker_id, context)
        return True
    return False


async def ask_for_simulation_model(
    stub, worker_id: str, stop_flag: threading.Event | None = None
) -> None:
    while True:
        if stop_flag is not None and stop_flag.is_set():
            logger.info(
                "Worker %s: Stop requested during model acquisition.", worker_id
            )
            return

        try:
            logger.info("Worker %s: Requesting simulation model archive...", worker_id)
            simulation_model = await request_simulation_model(stub, worker_id)

            if simulation_model.status == sm.ModelStatus.NO_MODEL_AVAILABLE:
                logger.info(
                    "Worker %s: No simulation model available. Retrying in %d seconds...",
                    worker_id,
                    MODEL_RETRY_DELAY_SEC,
                )
            elif simulation_model.status != sm.ModelStatus.ON_SERVER:
                logger.warning(
                    "Worker %s: Unexpected model status: %s. Retrying in %d seconds...",
                    worker_id,
                    simulation_model.status,
                    MODEL_RETRY_DELAY_SEC,
                )
            else:
                logger.info("Worker %s: Received simulation model archive.", worker_id)

                if await try_unpacking_model_archive(
                    simulation_model.package_archive, worker_id
                ):
                    logger.info(
                        "Worker %s: Unpacking simulation model archive successful.",
                        worker_id,
                    )
                    return
                else:
                    logger.warning(
                        "Worker %s: Corrupted simulation model archive. Retrying in %d seconds...",
                        worker_id,
                        MODEL_RETRY_DELAY_SEC,
                    )

        except grpc.aio.AioRpcError as e:
            if (
                stop_flag is not None and stop_flag.is_set()
            ) or _is_expected_shutdown_rpc_error(e):
                logger.info(
                    "Worker %s: server unavailable during model acquisition (code=%s). Exiting.",
                    worker_id,
                    e.code(),
                )
                return
            logger.error(
                "Worker %s: Unable to connect to server: %s. Retrying in %d seconds...",
                worker_id,
                e,
                MODEL_RETRY_DELAY_SEC,
            )
        except Exception as e:
            logger.warning(
                "Worker %s: Unexpected error while requesting model: %s. Retrying in %d seconds...",
                worker_id,
                e,
                MODEL_RETRY_DELAY_SEC,
            )

        if await _retry_with_stop(
            MODEL_RETRY_DELAY_SEC, stop_flag, worker_id, "model acquisition"
        ):
            return


async def run_simulation_loop(
    stub, worker_id: str, stop_flag: threading.Event | None = None
) -> None:
    while True:
        if stop_flag is not None and stop_flag.is_set():
            logger.info("Worker %s: Stop requested, exiting loop.", worker_id)
            break

        try:
            logger.debug("Worker %s: Requesting a job...", worker_id)
            simulation_job = await request_simulation_job(stub, worker_id)

            if simulation_job.status == sm.JobStatus.EXCEPTION:
                logger.critical(
                    "Worker %s: Received EXCEPTION status. Shutting down...", worker_id
                )
                if stop_flag is not None:
                    stop_flag.set()
                break

            elif simulation_job.status == sm.JobStatus.NO_JOB_AVAILABLE:
                logger.debug(
                    "Worker %s: Long-poll timed out, re-requesting...", worker_id
                )
                continue

            elif simulation_job.status == sm.JobStatus.ERROR:
                logger.error(
                    "Worker %s: Server returned an error for the job request. Retrying...",
                    worker_id,
                )

            elif simulation_job.status == sm.JobStatus.JOBSTATUS_UNSPECIFIED:
                logger.warning(
                    "Worker %s: Received unspecified status. Retrying...", worker_id
                )

            else:
                logger.info(
                    "Worker %s: Processing job %s...", worker_id, simulation_job.job_id
                )
                await handle_simulation_job(stub, simulation_job, worker_id, stop_flag)
                continue

        except grpc.aio.AioRpcError as e:
            if (
                stop_flag is not None and stop_flag.is_set()
            ) or _is_expected_shutdown_rpc_error(e):
                logger.info(
                    "Worker %s: server unavailable (code=%s). Exiting loop.",
                    worker_id,
                    e.code(),
                )
                break
            logger.error(
                "Worker %s: gRPC error: %s. Retrying in %d seconds...",
                worker_id,
                e,
                JOB_RETRY_DELAY_SEC,
            )
        except Exception as e:
            logger.exception("Worker %s: Unexpected error: %s", worker_id, e)

        if await _retry_with_stop(
            JOB_RETRY_DELAY_SEC, stop_flag, worker_id, "job loop"
        ):
            break


def _create_channel():
    config = get_simulation_config()
    return grpc.aio.insecure_channel(config.grpc_target, options=config.channel_options)


async def main(stop_flag: threading.Event | None = None, worker_id: str | None = None):
    mode = os.getenv("OPEN_DARTS_RUNNER", "thread").lower()
    logger.info("Worker starting with OPEN_DARTS_RUNNER=%s", mode)
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
                    "Worker %s: suppressing gRPC AIO poller _handle_events noise",
                    worker_id,
                )
                _warned["done"] = True
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_ignore_poller_eagain)

    async with _create_channel() as channel:
        stub = sm_grpc.SimulationMessagingStub(channel)
        await ask_for_simulation_model(stub, worker_id, stop_flag)
        await run_simulation_loop(stub, worker_id, stop_flag)


if __name__ == "__main__":
    asyncio.run(main())


@contextmanager
def worker_file_context(worker_id: str | int):
    temp_dir = compute_worker_temp_dir(worker_id)
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Worker %s: temp dir ready at %s", worker_id, temp_dir)
    yield
