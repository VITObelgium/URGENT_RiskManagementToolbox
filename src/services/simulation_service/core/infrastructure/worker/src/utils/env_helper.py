from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

from utils import find_repo_root


def get_run_id() -> str:
    """Return the current run ID from the environment, or 'default' if not set."""
    return os.environ.get("URGENT_RUN_ID", "default")


def compute_worker_temp_dir(worker_id: str | int) -> Path:
    """Compute the absolute temp directory path for a given worker."""
    repo_root = find_repo_root()
    run_id = get_run_id()
    return (
        repo_root / f"orchestration_files_{run_id}" / f".worker_{str(worker_id)}_temp"
    )


async def sleep_with_stop(
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
