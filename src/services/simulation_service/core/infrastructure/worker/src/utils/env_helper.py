import asyncio
import threading
from pathlib import Path


def _find_repo_root(marker_file: str = "pyproject.toml") -> Path:
    """Dynamically find the repository root by searching for a marker file (e.g., pyproject.toml).

    This avoids hardcoding parent levels and works regardless of script location.
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / marker_file).exists():
            return parent
    raise RuntimeError(f"Repository root with {marker_file} not found from {__file__}")


def compute_worker_temp_dir(worker_id: str | int) -> Path:
    """Compute the absolute temp directory path for a given worker."""
    repo_root = _find_repo_root()
    return repo_root / f"orchestration_files/.worker_{str(worker_id)}_temp"


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
