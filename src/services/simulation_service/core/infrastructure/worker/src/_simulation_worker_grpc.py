from __future__ import annotations

import asyncio
import os
import threading

from logger import get_logger

logger = get_logger("worker-dispatcher", filename=__name__)


async def main(stop_flag: threading.Event | None = None, worker_id: str | None = None):
    mode = os.getenv("OPEN_DARTS_RUNNER", "thread").lower()
    if mode == "docker":
        logger.info("Worker dispatcher: selecting docker-mode entrypoint")
        from services.simulation_service.core.infrastructure.worker.src._simulation_worker_grpc_docker import (
            main as docker_main,
        )

        await docker_main(worker_id=worker_id)

    elif mode == "thread":
        logger.info("Worker dispatcher: selecting thread-mode entrypoint")
        from services.simulation_service.core.infrastructure.worker.src._simulation_worker_grpc_thread import (
            main as thread_main,
        )

        await thread_main(stop_flag=stop_flag, worker_id=worker_id)

    else:
        raise ValueError(f"Unknown OPEN_DARTS_RUNNER mode: {mode}")


if __name__ == "__main__":
    asyncio.run(main())
