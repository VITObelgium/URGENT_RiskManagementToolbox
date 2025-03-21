_hard_dependencies = ["numpy", "pydantic"]
_missing_dependencies = []

for _dependency in _hard_dependencies:
    try:
        __import__(_dependency)
    except ImportError as _e:
        _missing_dependencies.append(f"{_dependency}: {_e}")

if _missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(_missing_dependencies)
    )
del _hard_dependencies, _dependency, _missing_dependencies

from services.solution_updater_service.core.api import (  # noqa: F401, E402
    ControlVector,
    OptimizationEngine,
    SolutionUpdaterService,
    ensure_not_none,
)

__all__ = [
    "SolutionUpdaterService",
    "OptimizationEngine",
    "ControlVector",
    "ensure_not_none",
]
