_hard_dependencies = ["numpy", "grpc", "pydantic"]
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

from services.simulation_service.core.api import (  # noqa: F401, E402
    SimulationService,
    simulation_cluster_contex_manager,
)

__all__ = ["SimulationService", "simulation_cluster_contex_manager"]
