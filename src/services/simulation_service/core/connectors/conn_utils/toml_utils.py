from pathlib import Path


from logger import get_logger

logger = get_logger(__name__)


def _find_pyproject_toml(start: Path | None = None) -> Path | None:
    """
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
