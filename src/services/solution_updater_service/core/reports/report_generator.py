from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from logger import get_csv_logger, get_logger
from services.solution_updater_service.core.reports import PopulationStatistic
from services.solution_updater_service.core.utils import ensure_not_none

LOG_PRECISION = 5


class ReportGenerator:
    def __init__(self) -> None:
        self._logger = get_logger(__name__)
        self._control_vector_logger: logging.Logger | None = None
        self._population_statistic_logger: logging.Logger | None = None
        self._best_result_logger: logging.Logger | None = None

    def log_control_vector_and_values(
        self,
        *,
        generation: int,
        control_vector: npt.NDArray[np.float64],
        population_statistic: PopulationStatistic,
        parameters_name: list[str],
        results_name: list[str],
    ) -> None:
        if self._control_vector_logger is None:
            self._control_vector_logger = get_csv_logger(
                "control_vector_logger.csv",
                logger_name="control_vector_logger",
                columns=["generation", "individual", *parameters_name, *results_name],
            )

        for idx, val in enumerate(
            np.hstack((control_vector, population_statistic.results))
        ):
            v_str = ",".join(f"{v:.{LOG_PRECISION}f}" for v in np.array(val))
            ensure_not_none(
                self._control_vector_logger, "Control vector logger not initialized"
            ).info(f"{generation},{idx},{v_str}")

    def log_population_statistic(
        self,
        *,
        generation: int,
        population_statistic: PopulationStatistic,
        results_name: list[str],
    ) -> None:

        population_size, n_objectives = population_statistic.results.shape

        if self._population_statistic_logger is None:
            self._population_statistic_logger = get_csv_logger(
                "population_statistic_logger.csv",
                logger_name="population_statistic_logger",
                columns=_generation_csv_columns(results_name, population_size),
            )

        mn = _to_obj_list(
            population_statistic.min, name="min", n_objectives=n_objectives
        )
        mx = _to_obj_list(
            population_statistic.max, name="max", n_objectives=n_objectives
        )
        av = _to_obj_list(
            population_statistic.avg, name="avg", n_objectives=n_objectives
        )
        sd = _to_obj_list(
            population_statistic.std, name="std", n_objectives=n_objectives
        )

        self._logger.info(
            f"Generation {generation} statistics: min={mn:.{LOG_PRECISION}f}, max={mx:.{LOG_PRECISION}f}, avg={av:.{LOG_PRECISION}f}, std={sd:.{LOG_PRECISION}f}",
        )

        pop_values: list[float] = []
        for idx, item in enumerate(population_statistic.results):
            item_vals = _to_obj_list(
                item, name=f"population[{idx}]", n_objectives=n_objectives
            )
            pop_values.extend(item_vals)

        row_values: list[float] = mn + mx + av + sd + pop_values
        row_str = ",".join(f"{v:.{LOG_PRECISION}f}" for v in row_values)
        ensure_not_none(
            self._population_statistic_logger,
            "Population statistic logger not initialized",
        ).info(f"{generation},{row_str}")

    def log_best_result(
        self,
        *,
        generation: int,
        best_result: float | npt.NDArray[np.float64],
        results_name: list[str],
    ) -> None:
        if self._best_result_logger is None:
            self._best_result_logger = get_csv_logger(
                "best_result_logger.csv",
                logger_name="best_result_logger",
                columns=["generation", *results_name],
            )

        if isinstance(best_result, float):
            best_result_v = [best_result]
        else:
            best_result_v = list(best_result)

        best_results_dsc = {k: float(v) for k, v in zip(results_name, best_result_v)}
        self._logger.info(f"Best result so far: {best_results_dsc}")

        arr = np.asarray(best_result, dtype=float).ravel()

        if arr.size != len(results_name):
            raise ValueError(
                f"Expected {len(results_name)} values, got {arr.size}: {arr!r}"
            )

        row_str = ",".join(f"{v:.{LOG_PRECISION}f}" for v in arr.tolist())

        ensure_not_none(
            self._best_result_logger, "Best result logger not initialized"
        ).info(f"{generation},{row_str}")


def _to_obj_list(
    x: float | npt.NDArray[np.float64], *, name: str, n_objectives: int
) -> list[float]:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 1 and n_objectives == 1:
        return [float(arr[0])]
    if arr.size != n_objectives:
        raise ValueError(
            f"Expected {name} to have {n_objectives} objective values, got {arr.size}: {arr!r}"
        )
    return [float(v) for v in arr.tolist()]


def _sanitize_col(name: str) -> str:
    return (
        name.strip()
        .replace(",", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _generation_csv_columns(
    expected_optimization_function_names: list[str],
    pop_size: int,
) -> list[str]:
    metrics = ["min", "max", "avg", "std"]
    obj_cols = [_sanitize_col(o) for o in expected_optimization_function_names]

    cols = ["generation"]
    for metric in metrics:
        cols += [f"{metric}_{obj}" for obj in obj_cols]
    cols += [f"ind_{i}_{obj}" for i in range(pop_size) for obj in obj_cols]
    return cols
