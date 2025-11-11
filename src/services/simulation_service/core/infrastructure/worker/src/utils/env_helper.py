import asyncio
import os
import shutil
import subprocess
import threading
import time
from logging import Logger
from pathlib import Path

from logger import get_logger

setup_logger = get_logger()  # Logger for setup phase, i.e. venv preparation


def detect_pixi_runtime() -> bool:
    if os.environ.get("PIXI_ENVIRONMENT_NAME"):
        return True
    return False


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


def _get_shared_venv_dir() -> Path:
    """Return the shared venv directory under orchestration_files."""
    repo_root = _find_repo_root()
    return repo_root / "orchestration_files/.venv_darts"


def _release_venv_lock(venv_dir: Path) -> None:
    lock_dir = venv_dir / ".venv_lock"
    try:
        if lock_dir.exists():
            shutil.rmtree(lock_dir, ignore_errors=True)
    except Exception:
        pass


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


def _venv_python_path(venv_dir: Path) -> Path:
    # Linux/macOS layout
    return venv_dir / "bin/python3.10"


def _acquire_venv_lock(venv_dir: Path, timeout: float = 600.0) -> bool:
    """Best-effort file lock to avoid concurrent venv creation/installation.

    Creates a lock directory atomically. If already exists, wait for ready marker or
    the lock to be released up to timeout seconds.
    """
    lock_dir = venv_dir / ".venv_lock"
    ready_marker = venv_dir / ".venv_ready"
    try:
        lock_dir.mkdir(parents=True, exist_ok=False)
        return True
    except FileExistsError:
        # Someone else is preparing the venv; wait for completion
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            if ready_marker.exists() and _venv_python_path(venv_dir).exists():
                return False
            if not lock_dir.exists():
                return False
            time.sleep(0.5)
        return False


def prepare_shared_venv(worker_id: str | int, logger: Logger) -> Path | None:
    """Create or reuse a shared Python3.10 venv and install model requirements.
            Creation strategy:
                    1) Use system `python3.10 -m venv` if available.
                    2) If not possible, abort and return None.
            After creation, install requirements from
            `src/services/simulation_service/core/infrastructure/worker/worker_requirements.txt`
    Returns the path to the venv python if successful, else None.
    """
    venv_dir = _get_shared_venv_dir()
    venv_dir.mkdir(parents=True, exist_ok=True)
    venv_python = _venv_python_path(venv_dir)
    ready_marker = venv_dir / ".venv_ready"

    # If already prepared, return venv path
    if venv_python.exists() and ready_marker.exists():
        logger.info("Worker %s: Using existing shared venv at %s", worker_id, venv_dir)
        return venv_python

    got_lock = _acquire_venv_lock(venv_dir)
    try:
        # Re-check after acquiring/contending for the lock
        if venv_python.exists() and ready_marker.exists():
            return venv_python

        py310 = shutil.which("python3.10")

        if not venv_python.exists():
            setup_logger.info(
                "Venv not found. Worker %s: Creating venv at %s (Python 3.10 required)",
                worker_id,
                venv_dir,
            )
            try:
                if py310:
                    setup_logger.info(
                        "Worker %s: Using system python3.10:%s to create venv.",
                        worker_id,
                        py310,
                    )
                    p = subprocess.run(
                        [py310, "-m", "venv", str(venv_dir)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                else:
                    setup_logger.error(
                        "Worker %s: python3.10 not found!",
                        worker_id,
                    )
                    return None

                setup_logger.debug(
                    "Worker %s: venv creation stdout:\n%s", worker_id, p.stdout
                )
                if p.stderr:
                    setup_logger.debug(
                        "Worker %s: venv creation stderr:\n%s", worker_id, p.stderr
                    )
            except subprocess.CalledProcessError as e:
                # Log captured output if available and return failure
                out = getattr(e, "stdout", None)
                err = getattr(e, "stderr", None)
                if out:
                    setup_logger.error(
                        "Worker %s: venv creation failed, stdout:\n%s", worker_id, out
                    )
                if err:
                    setup_logger.error(
                        "Worker %s: venv creation failed, stderr:\n%s", worker_id, err
                    )
                setup_logger.error("Worker %s: Failed to create venv: %s", worker_id, e)
                return None
            except Exception as e:
                setup_logger.error("Worker %s: Failed to create venv: %s", worker_id, e)
                return None

        # Sanity-check the interpreter version inside the venv is 3.10
        try:
            ver = subprocess.run(
                [
                    str(venv_python),
                    "-c",
                    "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            if ver.stdout.strip() != "3.10":
                setup_logger.error(
                    "Worker %s: venv interpreter is not Python 3.10 (got %s). Aborting.",
                    worker_id,
                    ver.stdout.strip(),
                )
                return None

        except subprocess.CalledProcessError as e:
            setup_logger.error(
                "Worker %s: Failed to verify venv interpreter version: %s", worker_id, e
            )
            return None

        try:
            p = subprocess.run(
                [
                    str(venv_python),
                    "-m",
                    "pip",
                    "install",
                    "-U",
                    "pip",
                    "setuptools",
                    "wheel",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            setup_logger.debug(
                "Worker %s: pip upgrade stdout:\n%s", worker_id, p.stdout
            )
            if p.stderr:
                setup_logger.debug(
                    "Worker %s: pip upgrade stderr:\n%s", worker_id, p.stderr
                )
        except subprocess.CalledProcessError as e:
            setup_logger.warning(
                "Worker %s: Failed to upgrade pip tooling in venv (%s); proceeding.",
                worker_id,
                e,
            )
            out = getattr(e, "stdout", None)
            err = getattr(e, "stderr", None)
            if out:
                setup_logger.debug(
                    "Worker %s: pip upgrade failed stdout:\n%s", worker_id, out
                )
            if err:
                setup_logger.debug(
                    "Worker %s: pip upgrade failed stderr:\n%s", worker_id, err
                )
        except Exception as e:
            setup_logger.warning(
                "Worker %s: Failed to upgrade pip tooling in venv (%s); proceeding.",
                worker_id,
                e,
            )

        req_path = Path(__file__).resolve().parents[1] / "worker_requirements.txt"
        setup_logger.info(
            "Worker %s: Installing requirements from %s", worker_id, req_path
        )
        install_cwd = str(req_path.parent)

        setup_logger.debug(
            "Worker %s: Running pip install with cwd=%s", worker_id, install_cwd
        )
        try:
            p = subprocess.run(
                [str(venv_python), "-m", "pip", "install", "-r", str(req_path)],
                check=True,
                cwd=install_cwd,
                capture_output=True,
                text=True,
            )
            setup_logger.debug(
                "Worker %s: pip install -r stdout:\n%s", worker_id, p.stdout
            )
            if p.stderr:
                setup_logger.warning(
                    "Worker %s: pip install -r stderr:\n%s", worker_id, p.stderr
                )
        except subprocess.CalledProcessError as e:
            out = getattr(e, "stdout", None)
            err = getattr(e, "stderr", None)
            if out:
                setup_logger.error(
                    "Worker %s: pip install -r %s failed stdout:\n%s",
                    worker_id,
                    req_path,
                    out,
                )
            if err:
                setup_logger.error(
                    "Worker %s: pip install -r %s failed stderr:\n%s",
                    worker_id,
                    req_path,
                    err,
                )
            setup_logger.error(
                "Worker %s: pip install -r %s failed: %s", worker_id, req_path, e
            )
            return None
        except Exception as e:
            setup_logger.error(
                "Worker %s: pip install -r %s failed: %s", worker_id, req_path, e
            )
            return None

        # Quick validation step of darts importability
        try:
            test = subprocess.run(
                [str(venv_python), "-c", "import darts, sys; print(sys.version)"],
                capture_output=True,
                text=True,
            )
            if test.returncode != 0:
                logger.error(
                    "Worker %s: DARTS not importable in venv; stderr: %s",
                    worker_id,
                    test.stderr[-500:],
                )
                return None
        except Exception as e:
            logger.error("Worker %s: Failed to validate venv import: %s", worker_id, e)
            return None

        try:
            ready_marker.touch(exist_ok=True)
        except Exception:
            pass

        setup_logger.info("Worker %s: Shared venv ready", worker_id)
        return venv_python
    finally:
        if got_lock:
            _release_venv_lock(venv_dir)
