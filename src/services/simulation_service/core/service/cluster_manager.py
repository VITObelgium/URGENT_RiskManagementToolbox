from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ClusterManager(Protocol):
    """Protocol describing the ClusterManager"""

    async def start(
        self, worker_count: int, model_archive: bytes | None
    ) -> None:  # pragma: no cover
        """Start the cluster and prepare workers.

        Args:
            worker_count: requested number of workers (may be advisory).
            model_archive: optional bytes of a zipped simulation model to distribute
                to workers (implementations may ignore or accept a path instead).
        """

    async def stop(self) -> None:  # pragma: no cover
        """Stop the cluster and clean up resources."""

    async def submit_cases(
        self, sim_cases: list[dict], timeout: float
    ) -> list[dict]:  # pragma: no cover
        """Submit simulation cases to the cluster and wait for results.

        Args:
            sim_cases: list of simulation case dicts (shape defined by the service).
            timeout: maximum seconds to wait for all submitted cases.

        Returns:
            List of result dictionaries, one per completed case. Implementations
            should aim to match the semantics currently provided by
            the in-process manager (status, results, original case).
        """
