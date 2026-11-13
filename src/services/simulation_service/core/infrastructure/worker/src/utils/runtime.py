from __future__ import annotations

import os
import shutil
from pathlib import Path

from utils import find_repo_root
from services.simulation_service.core.infrastructure.worker.src.utils.env_helper import (
    compute_worker_temp_dir,
)


def get_worker_runtime_dir(worker_id: str | int) -> Path:
    return Path(os.environ.get("SIM_MODEL_DIR") or compute_worker_temp_dir(worker_id))


def stage_worker_runtime(worker_id: str | int, *, reset: bool = False) -> Path:
    runtime_dir = get_worker_runtime_dir(worker_id)

    if reset:
        shutil.rmtree(runtime_dir, ignore_errors=True)

    runtime_dir.mkdir(parents=True, exist_ok=True)

    repo_root = find_repo_root()
    runtime_sources = (
        (
            repo_root / "plugins" / "connectors",
            runtime_dir / "connectors",
        ),
        (repo_root / "src" / "logger", runtime_dir / "logger"),
    )

    for source, destination in runtime_sources:
        if not source.exists():
            raise FileNotFoundError(
                f"Required worker runtime source not found: {source}"
            )
        shutil.copytree(source, destination, dirs_exist_ok=True)

    return runtime_dir
