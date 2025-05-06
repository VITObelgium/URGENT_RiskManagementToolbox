from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from services.solution_updater_service.core.utils.type_checks import ensure_not_none


@dataclass(frozen=True)
class SolutionMetrics:
    global_min: float

    last_batch_min: float
    last_batch_max: float
    last_batch_avg: float
    last_batch_std: float


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
        batch_min = float(new_results.min())
        batch_max = float(new_results.max())
        batch_avg = float(np.average(new_results))
        batch_std = float(np.std(new_results))

        if self._metrics is None:  # first run
            global_min = batch_min
        else:
            global_min = min(batch_min, self._metrics.global_min)

        self._metrics = SolutionMetrics(
            global_min=global_min,
            last_batch_min=batch_min,
            last_batch_max=batch_max,
            last_batch_avg=batch_avg,
            last_batch_std=batch_std,
        )
