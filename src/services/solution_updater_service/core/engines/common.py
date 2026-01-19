from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from common import OptimizationStrategy
from services.solution_updater_service.core.utils import ensure_not_none


@dataclass(frozen=True)
class SolutionMetrics:
    global_best: float

    last_population_min: float
    last_population_max: float
    last_population_avg: float
    last_population_std: float


class OptimizationEngineInterface(ABC):
    def __init__(self) -> None:
        self._metrics: SolutionMetrics | None = None
        self._strategy: OptimizationStrategy | None = None

    @abstractmethod
    def update_solution_to_next_iter(
        self,
        parameters: npt.NDArray[np.float64],
        results: npt.NDArray[np.float64],
        lb: npt.NDArray[np.float64],
        ub: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64] | None = None,
        b: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]: ...

    @property
    def metrics(self) -> SolutionMetrics:
        return ensure_not_none(self._metrics)

    @property
    def global_best_result(self) -> float:
        return self.metrics.global_best

    @property
    @abstractmethod
    def global_best_control_vector(self) -> npt.NDArray[np.float64]: ...

    def _update_metrics(self, new_results: npt.NDArray[np.float64]) -> None:
        """Updates metrics based on optimization strategy."""
        # Filter out infinite values for statistics calculation
        finite_mask = np.isfinite(new_results)

        if np.any(finite_mask):
            # Use only finite results for statistics
            finite_results = new_results[finite_mask]
            population_min = float(finite_results.min())
            population_max = float(finite_results.max())
            population_avg = float(np.average(finite_results))
            population_std = float(np.std(finite_results))
        else:
            # All values are inf/-inf (all particles failed this iteration)
            # Use sentinel values that indicate failure
            if self._strategy == OptimizationStrategy.MINIMIZE:
                population_min = np.inf
                population_max = np.inf
            else:
                population_min = -np.inf
                population_max = -np.inf
            population_avg = np.nan  # No meaningful average
            population_std = np.nan  # No meaningful std deviation

        if self._metrics is None:  # first run
            if self._strategy == OptimizationStrategy.MINIMIZE:
                global_best = population_min
            else:
                global_best = population_max
        else:
            if self._strategy == OptimizationStrategy.MINIMIZE:
                global_best = min(population_min, self._metrics.global_best)
            else:
                global_best = max(population_max, self._metrics.global_best)

        self._metrics = SolutionMetrics(
            global_best=global_best,
            last_population_min=population_min,
            last_population_max=population_max,
            last_population_avg=population_avg,
            last_population_std=population_std,
        )
