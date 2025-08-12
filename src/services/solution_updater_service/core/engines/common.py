from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from services.solution_updater_service.core.utils.type_checks import ensure_not_none


@dataclass(frozen=True)
class SolutionMetrics:
    global_min: float

    last_population_min: float
    last_population_max: float
    last_population_avg: float
    last_population_std: float


class OptimizationEngineInterface(ABC):
    def __init__(self) -> None:
        self._metrics: SolutionMetrics | None = None

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
        return self.metrics.global_min

    @property
    @abstractmethod
    def global_best_controll_vector(self) -> npt.NDArray[np.float64]: ...

    def _update_metrics(self, new_results: npt.NDArray[np.float64]) -> None:
        population_min = float(new_results.min())
        population_max = float(new_results.max())
        population_avg = float(np.average(new_results))
        population_std = float(np.std(new_results))

        if self._metrics is None:  # first run
            global_min = population_min
        else:
            global_min = min(population_min, self._metrics.global_min)

        self._metrics = SolutionMetrics(
            global_min=global_min,
            last_population_min=population_min,
            last_population_max=population_max,
            last_population_avg=population_avg,
            last_population_std=population_std,
        )
