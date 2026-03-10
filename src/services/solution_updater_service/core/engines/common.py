from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from common import OptimizationStrategy
from services.solution_updater_service.core.utils import ensure_not_none


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    global_best: (
        float | npt.NDArray[np.float64]
    )  # Can be scalar or multi-objective array

    min: float | npt.NDArray[np.float64]
    max: float | npt.NDArray[np.float64]
    avg: float | npt.NDArray[np.float64]
    std: float | npt.NDArray[np.float64]

    population: list[float] | list[npt.NDArray[np.float64]]


class OptimizationEngineInterface(ABC):
    def __init__(self) -> None:
        self._generation_summary: GenerationSummary | None = None
        self._indexed_objectives_strategy: dict[int, OptimizationStrategy] | None = None

    @abstractmethod
    def update_solution_to_next_iter(
        self,
        parameters: npt.NDArray[np.float64],
        results: npt.NDArray[np.float64],
        lb: npt.NDArray[np.float64],
        ub: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
        A: npt.NDArray[np.float64] | None = None,
        b: npt.NDArray[np.float64] | None = None,
        iteration_ratio: float | None = None,
    ) -> npt.NDArray[np.float64]: ...

    @property
    def generation_summary(self) -> GenerationSummary:
        return ensure_not_none(self._generation_summary)

    @property
    def global_best_result(self) -> float | npt.NDArray[np.float64]:
        return self.generation_summary.global_best

    @property
    @abstractmethod
    def global_best_control_vector(self) -> npt.NDArray[np.float64]: ...

    def _update_generation_summary(
        self,
        new_results: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
    ) -> None:
        """Updates metrics based on optimization strategy."""
        # Check if multi-objective
        is_multi_objective = len(indexed_objectives_strategy) > 1

        if is_multi_objective:
            # Multi-objective: track statistics per objective
            finite_mask = np.isfinite(new_results).all(axis=1)

            if np.any(finite_mask):
                finite_results = new_results[finite_mask]
                population_min = finite_results.min(axis=0)
                population_max = finite_results.max(axis=0)
                population_avg = np.average(finite_results, axis=0)
                population_std = np.std(finite_results, axis=0)
            else:
                # All particles failed - use sentinels per objective
                n_objectives = len(indexed_objectives_strategy)
                population_min = np.full(n_objectives, np.nan)
                population_max = np.full(n_objectives, np.nan)
                population_avg = np.full(n_objectives, np.nan)
                population_std = np.full(n_objectives, np.nan)

            # For multi-objective, global_best is the first non-dominated solution
            # (tracked separately in PSO state)
            if self._generation_summary is None:
                global_best = population_min.copy()  # Placeholder
            else:
                global_best = self._generation_summary.global_best

            self._generation_summary = GenerationSummary(
                global_best=global_best,
                min=population_min,
                max=population_max,
                avg=population_avg,
                std=population_std,
                population=[row for row in new_results],
            )
        else:
            # Single objective (original implementation)
            finite_mask = np.isfinite(new_results)

            if np.any(finite_mask):
                finite_results = new_results[finite_mask]
                population_min = float(finite_results.min())
                population_max = float(finite_results.max())
                population_avg = float(np.average(finite_results))
                population_std = float(np.std(finite_results))
            else:
                strategy = next(iter(indexed_objectives_strategy.values()))
                if strategy == OptimizationStrategy.MINIMIZE:
                    population_min = np.inf
                    population_max = np.inf
                else:
                    population_min = -np.inf
                    population_max = -np.inf
                population_avg = np.nan
                population_std = np.nan

            if self._generation_summary is None:
                strategy = next(iter(indexed_objectives_strategy.values()))
                if strategy == OptimizationStrategy.MINIMIZE:
                    global_best = population_min
                else:
                    global_best = population_max
            else:
                strategy = next(iter(indexed_objectives_strategy.values()))
                if strategy == OptimizationStrategy.MINIMIZE:
                    global_best = min(
                        population_min, self._generation_summary.global_best
                    )
                else:
                    global_best = max(
                        population_max, self._generation_summary.global_best
                    )

            self._generation_summary = GenerationSummary(
                global_best=global_best,
                min=population_min,
                max=population_max,
                avg=population_avg,
                std=population_std,
                population=new_results.flatten().tolist(),
            )
