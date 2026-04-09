"""
NOTE:
This module must be aligned with python 3.10 syntax, as open-darts whl requires it.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Generator


@contextmanager
def static_workspace(work_dir: Path | None) -> Generator[Path | None, None, None]:
    """Context manager for thread mode that just yields the work directory unchanged."""
    yield work_dir


@contextmanager
def docker_job_workspace(template_dir: Path) -> Generator[Path, None, None]:
    """
    Context manager that creates an isolated workspace for a Docker simulation job.

    Creates a temporary copy of the template directory, prepares it for execution,
    yields the workspace path, and cleans up afterwards.

    Args:
        template_dir: Path to the template directory containing the simulation model

    Yields:
        Path to the prepared workspace directory

    Raises:
        FileNotFoundError: If template_dir doesn't exist
    """
    if not template_dir.exists():
        raise FileNotFoundError(
            f"Docker simulation template directory does not exist: {template_dir}"
        )

    workspace_prefix = f".{template_dir.name.lstrip('.')}_job_"
    workspace_dir = Path(
        tempfile.mkdtemp(prefix=workspace_prefix, dir=str(template_dir.parent))
    )
    try:
        shutil.rmtree(workspace_dir)
        shutil.copytree(template_dir, workspace_dir)
        yield workspace_dir
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)
