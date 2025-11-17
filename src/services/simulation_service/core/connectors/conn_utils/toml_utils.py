from pathlib import Path

import tomli

from logger import get_logger

logger = get_logger(__name__)


def _find_pyproject_toml(start: Path | None = None) -> Path | None:
    """
    #TODO: refactor
    Locate pyproject.toml by checking:
    1) Current working directory
    2) Walking up from the provided start path (or this file) to filesystem root

    Returns
    -------
    Path | None
        Returns the resolved Path if found, else None.
    """
    try:
        cwd_candidate = Path.cwd() / "pyproject.toml"
        if cwd_candidate.exists():
            return cwd_candidate.resolve()
    except Exception:
        pass

    base = start or Path(__file__).resolve()
    for candidate in [base] + list(base.parents):
        pp = candidate / "pyproject.toml"
        if pp.exists():
            return pp.resolve()

    return None


def get_timeout_value() -> int:
    """
    #TODO: refactor
    Get the timeout value in seconds."""
    pyproject_path = _find_pyproject_toml()
    if pyproject_path is None:
        logger.warning("pyproject.toml not found; using default timeout of 15 minutes")
        return 15 * 60
    try:
        with pyproject_path.open("rb") as f:
            pyproject_data = tomli.load(f)
        timeout_seconds = pyproject_data.get("toolbox-config", {}).get(
            "timeout_seconds"
        )
        logger.debug("timeout_seconds read from pyproject.toml: %s", timeout_seconds)
        if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be a positive integer in pyproject.toml"
            )
        return timeout_seconds
    except Exception:
        logger.warning(
            "Error reading timeout_seconds from pyproject.toml; using default of 15 minutes"
        )
        return 15 * 60
