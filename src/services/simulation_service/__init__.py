import os

_is_worker_subprocess = (
    os.environ.get("URGENT_WORKER_SUBPROCESS") == "1"
    or os.environ.get("RUNNER_MODE", "").lower() == "docker"
)

if not _is_worker_subprocess:
    from services.simulation_service.core.api import (  # noqa: F401, E402
        SimulationService,
        simulation_cluster_context_manager,
        simulation_process_context_manager,
    )

    __all__ = [
        "SimulationService",
        "simulation_cluster_context_manager",
        "simulation_process_context_manager",
    ]
