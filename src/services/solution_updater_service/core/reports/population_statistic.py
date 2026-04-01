from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from common import OptimizationStrategy


@dataclass(frozen=True, slots=True)
class PopulationStatistic:
    generation: int
    min: float | npt.NDArray[np.float64]
    max: float | npt.NDArray[np.float64]
    avg: float | npt.NDArray[np.float64]
    std: float | npt.NDArray[np.float64]
    n_feasible: int
    results: npt.NDArray[np.float64]


class PopulationStatisticGenerator:

    @staticmethod
    def generate(
        generation: int,
        results: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
    ) -> PopulationStatistic:

        raw = np.array(results, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]

        if len(indexed_objectives_strategy) > 1:
            return PopulationStatisticGenerator._generate_multi(
                generation, raw, indexed_objectives_strategy
            )
        return PopulationStatisticGenerator._generate_single(
            generation, raw, indexed_objectives_strategy
        )

    @staticmethod
    def _generate_single(
        generation: int,
        results: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
    ) -> PopulationStatistic:

        col_idx, strategy = next(iter(indexed_objectives_strategy.items()))
        col = results[:, col_idx]

        finite_mask = np.isfinite(col)
        n_feasible = int(finite_mask.sum())

        if n_feasible > 0:
            finite = col[finite_mask]
            population_min = float(finite.min())
            population_max = float(finite.max())
            population_avg = float(finite.mean())
            population_std = float(finite.std())
        else:
            sentinel = np.inf if strategy == OptimizationStrategy.MINIMIZE else -np.inf
            population_min = sentinel
            population_max = sentinel
            population_avg = np.nan
            population_std = np.nan

        return PopulationStatistic(
            generation=generation,
            min=population_min,
            max=population_max,
            avg=population_avg,
            std=population_std,
            n_feasible=n_feasible,
            results=results.ravel(),
        )

    @staticmethod
    def _generate_multi(
        generation: int,
        results: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
    ) -> PopulationStatistic:

        n_objectives = len(indexed_objectives_strategy)
        finite_mask = np.isfinite(results).all(axis=1)
        n_feasible = int(finite_mask.sum())

        if n_feasible > 0:
            finite = results[finite_mask]
            population_min = finite.min(axis=0)
            population_max = finite.max(axis=0)
            population_avg = finite.mean(axis=0)
            population_std = finite.std(axis=0)
        else:
            population_min = np.full(n_objectives, np.nan)
            population_max = np.full(n_objectives, np.nan)
            population_avg = np.full(n_objectives, np.nan)
            population_std = np.full(n_objectives, np.nan)

        return PopulationStatistic(
            generation=generation,
            min=population_min,
            max=population_max,
            avg=population_avg,
            std=population_std,
            n_feasible=n_feasible,
            results=results,
        )