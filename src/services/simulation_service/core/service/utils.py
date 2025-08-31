import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from logger import get_logger

logger = get_logger(__name__)


@contextmanager
def core_directory() -> Generator[None, None, None]:
    """
    A context manager to temporarily switch to the core directory and restore the original directory upon exit.

    Yields:
        None
    """
    current_file = Path(__file__)
    core_dir = current_file.parent.parent
    logger.debug("Attempting to switch to core directory: %s", core_dir)

    if not core_dir.exists():
        logger.error("Core directory not found at %s", core_dir)
        raise FileNotFoundError(f"Core directory not found at {core_dir}")

    original_dir = os.getcwd()
    logger.debug("Original working directory: %s", original_dir)

    try:
        os.chdir(core_dir)
        logger.debug("Switched to core directory: %s", os.getcwd())
        yield
    finally:
        os.chdir(original_dir)
        logger.debug("Reverted to original working directory: %s", original_dir)
