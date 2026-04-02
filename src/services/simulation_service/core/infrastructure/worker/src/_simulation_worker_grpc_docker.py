"""
NOTE:
This module must be aligned with python 3.10 syntax, as open-darts whl requires it.
"""

from __future__ import annotations

import asyncio
import io
import os
import uuid
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import grpc.aio

import services.simulation_service.core.infrastructure.generated.simulation_messaging_pb2 as sm
import services.simulation_service.core.infrastructure.generated.simulation_messaging_pb2_grpc as sm_grpc
from logger import get_logger
from services.simulation_service.core.config import get_simulation_config
from services.simulation_service.core.infrastructure.worker.src._simulation_worker_grpc_thread import (
    run_simulation_loop,
)
from services.simulation_service.core.infrastructure.worker.src.utils import (
    get_worker_runtime_dir,
    stage_worker_runtime,
)

logger = get_logger("docker-worker", filename=__name__)


def _resolve_worker_id(explicit_worker_id: str | None = None) -> str:
    """Resolve worker ID from explicit arg, env var, or container hostname."""
    if explicit_worker_id:
        return explicit_worker_id
    if os.environ.get("SIM_WORKER_ID"):
        return str(os.environ["SIM_WORKER_ID"])

    # Docker containers get hostnames like "worker.1.abc123"
    hostname = os.environ.get("HOSTNAME")
    if hostname:
        return hostname.split(".", 1)[0]
    return str(uuid.uuid4().hex)[:8]


def _prepare_worker_runtime(worker_id: str) -> Path:
    """Prepare and stage the worker runtime directory for Docker mode."""
    os.environ["SIM_WORKER_ID"] = worker_id
    runtime_dir = stage_worker_runtime(worker_id, reset=True)
    os.environ["SIM_MODEL_DIR"] = str(runtime_dir)
    logger.info("Worker %s: staged runtime directory at %s", worker_id, runtime_dir)
    return runtime_dir


async def try_unpacking_model_archive(package_archive: bytes, worker_id: str) -> bool:
    """
    Unpack the simulation model archive to the worker's runtime directory.

    Args:
        package_archive: Binary content of the simulation model archive
        worker_id: Worker identifier for determining extraction path

    Returns:
        bool: True if unpacking was successful, False otherwise
    """
    extract_path = get_worker_runtime_dir(worker_id)

    if not package_archive:
        logger.error("Cannot unpack empty archive")
        return False

    try:
        with io.BytesIO(package_archive) as archive_data:
            with ZipFile(archive_data, "r") as simulation_model_archive:
                extract_path.mkdir(parents=True, exist_ok=True)
                simulation_model_archive.extractall(extract_path)

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
        logger.error("Unpacking failed due to: %s", str(e))
        return False


async def request_simulation_model(stub, worker_id: str):
    return await stub.RequestSimulationModel(sm.RequestModel(worker_id=worker_id))


async def ask_for_simulation_model(stub, worker_id: str) -> None:
    """
    Request and unpack simulation model from the server.
    Will keep retrying until successful.

    Args:
        stub: The gRPC stub to use for communication
        worker_id: Unique identifier for this worker
    """
    RETRY_DELAY = 5

    while True:
        try:
            logger.info("Worker %s: Requesting simulation model archive...", worker_id)
            simulation_model = await request_simulation_model(stub, worker_id)

            if simulation_model.status == sm.ModelStatus.NO_MODEL_AVAILABLE:
                logger.info(
                    "Worker %s: No simulation model available. Retrying in %d seconds...",
                    worker_id,
                    RETRY_DELAY,
                )
                await asyncio.sleep(RETRY_DELAY)
                continue

            if simulation_model.status != sm.ModelStatus.ON_SERVER:
                logger.warning(
                    "Worker %s: Received unexpected model status: %s. Retrying in %d seconds...",
                    worker_id,
                    simulation_model.status,
                    RETRY_DELAY,
                )
                await asyncio.sleep(RETRY_DELAY)
                continue

            logger.info("Worker %s: Received simulation model archive.", worker_id)

            if await try_unpacking_model_archive(
                simulation_model.package_archive, worker_id
            ):
                logger.info(
                    "Worker %s: Unpacking simulation model archive successful.",
                    worker_id,
                )
                return  # Success
            else:
                logger.warning(
                    "Worker %s: Corrupted simulation model archive. Retrying in %d seconds...",
                    worker_id,
                    RETRY_DELAY,
                )
                await asyncio.sleep(RETRY_DELAY)

        except grpc.aio.AioRpcError as e:
            logger.error(
                "Worker %s: Unable to connect to server due to %s. Retrying in %d seconds...",
                worker_id,
                e,
                RETRY_DELAY,
            )
            await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(
                "Worker %s: Unexpected error while requesting model: %s. Retrying in %d seconds...",
                worker_id,
                str(e),
                RETRY_DELAY,
            )
            await asyncio.sleep(RETRY_DELAY)


def _create_channel():
    config = get_simulation_config()
    return grpc.aio.insecure_channel(config.grpc_target, options=config.channel_options)


async def main(worker_id: str | None = None):
    worker_id = _resolve_worker_id(worker_id)
    logger.info("Worker %s starting in Docker mode", worker_id)

    _prepare_worker_runtime(worker_id)

    async with _create_channel() as channel:
        stub = sm_grpc.SimulationMessagingStub(channel)
        await ask_for_simulation_model(stub, worker_id)

        # Reuse the simulation loop from thread worker (no stop flag in Docker mode)
        await run_simulation_loop(stub, worker_id, stop_flag=None)


if __name__ == "__main__":
    asyncio.run(main())
