from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ClusterManager(Protocol):
    """Protocol describing the ClusterManager"""

    def start(self, worker_count: int) -> None:  # pragma: no cover
        """Start the cluster and prepare workers.

        Args:
            worker_count: requested number of workers (may be advisory).
        """

    def stop(self) -> None:  # pragma: no cover
        """Stop the cluster and clean up resources."""
