from __future__ import annotations

import inspect
import logging
from typing import Any
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from common import OptimizationStrategy
from logger import get_logger
from services.shared import Boundaries, ensure_not_none
from services.solution_updater_service.core.engines import (
    OptimizationEngineFactory,
    OptimizationEngineInterface,
)
from services.solution_updater_service.core.models import (
    ControlVector,
    OptimizationEngine,
    SolutionCandidate,
    SolutionUpdaterServiceRequest,
    SolutionUpdaterServiceResponse,
)
from services.solution_updater_service.core.reports import (
    PopulationStatisticGenerator,
    ReportGenerator,
)
from services.solution_updater_service.core.utils import (
    Idx,
    Param,
    ensure_not_none,
    get_mapping,
    get_numpy_values,
    numpy_to_dict,
)


class _MapperState:
    def __init__(
        self,
        control_vector_mapping: Mapping[Param, Idx],
        results_mapping: Mapping[Param, Idx],
        population_size: int,
    ):
        self.control_vector_mapping: Mapping[Param, Idx] = control_vector_mapping
        self.results_mapping: Mapping[Param, Idx] = results_mapping
        self.control_vector_length: int = len(control_vector_mapping)
        self.results_length: int = len(results_mapping)
        self.population_size: int = population_size


class _Mapper:
    def __init__(self) -> None:
        self._state: _MapperState | None = None

    def __init__(self) -> None:
        self._state: _MapperState | None = None

    def _state_keys(self, attr: str) -> list[str]:
        m = ensure_not_none(self._state, "Mapper state is not initialized.")
        return list(getattr(m, attr).keys())

    @property
    def is_initialized(self) -> bool:
        return self._state is not None

    def initialize(self, candidates: Sequence[SolutionCandidate]) -> None:
        if not self._state:
            self._state = self._initiate_mapper_on_first_call(candidates)

    @property
    def parameters_name(self) -> list[str]:
        return self._state_keys("control_vector_mapping")

    @property
    def results_name(self) -> list[str]:
        m = ensure_not_none(self._state, "Mapper state is not initialized.")
        return list(m.results_mapping.keys())

    @property
    def control_vector_mapping(self) -> Mapping[str, int]:
        m = ensure_not_none(self._state, "Mapper state is not initialized.")
        return m.control_vector_mapping

    @property
    def results_mapping(self) -> Mapping[str, int]:
        m = ensure_not_none(self._state, "Mapper state is not initialized.")
        return m.results_mapping

    @property
    def population_size(self) -> int:
        m = ensure_not_none(self._state, "Mapper state is not initialized.")
        return m.population_size

    def to_numpy(
        self, candidates: Sequence[SolutionCandidate]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if not self.is_initialized:
            self.initialize(candidates)

        m = ensure_not_none(self._state, "Mapper state is not initialized.")

        candidates_control_vector_array: npt.NDArray[np.float64] = np.empty(
            shape=(m.population_size, m.control_vector_length)
        )

        candidates_results_array: npt.NDArray[np.float64] = np.empty(
            shape=(m.population_size, m.results_length)
        )

        for idx, c in enumerate(candidates):
            control_vector_array = get_numpy_values(c.control_vector.items)
            results_array = get_numpy_values(c.cost_function_results.values)

            candidates_control_vector_array[idx][:] = control_vector_array
            candidates_results_array[idx][:] = results_array

        return candidates_control_vector_array, candidates_results_array

    def to_control_vectors(
        self, parameters2d: npt.NDArray[np.float64]
    ) -> list[ControlVector]:
        if not self._state:
            raise RuntimeError(
                "Mapper state is not initialized, call 'initialize' or 'to_numpy' first."
            )

        return [
            ControlVector(
                items=dict(numpy_to_dict(row, dict(self._state.control_vector_mapping)))
            )
            for row in parameters2d
        ]

    def get_variables_lb_and_ub_boundary(
        self, boundaries: dict[str, Boundaries]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if not self._state:
            raise RuntimeError(
                "Mapper state is not initialized, call 'initialize' or 'to_numpy' first."
            )

        lb = np.full(self._state.control_vector_length, -np.inf)
        ub = np.full(self._state.control_vector_length, np.inf)

        for k, b in boundaries.items():
            lb[self._state.control_vector_mapping[k]] = b.lb
            ub[self._state.control_vector_mapping[k]] = b.ub

        return lb, ub

    @staticmethod
    def _initiate_mapper_on_first_call(
        candidates: Sequence[SolutionCandidate],
    ) -> _MapperState:
        first_solution_candidate = candidates[0]
        population_size = len(candidates)
        control_vector_mapping = get_mapping(
            first_solution_candidate.control_vector.items
        )
        results_mapping = get_mapping(
            first_solution_candidate.cost_function_results.values
        )

        return _MapperState(control_vector_mapping, results_mapping, population_size)


class _SolutionUpdaterServiceLoopController:
    def __init__(
        self,
        max_generations: int,
        max_stall_generations: int,
        solution_updater_service: SolutionUpdaterService,
        is_multi_objective: bool,
    ) -> None:
        self._info = "Loop controller running"
        self._solution_updater_service = solution_updater_service

        self._max_generations = max_generations
        self._current_generation = 0

        self._base_stall, self._stall_left = (
            max_stall_generations,
            max_stall_generations,
        )

        self._last_run_global_best_result: float | None = None
        self._is_multi_objective: bool = is_multi_objective

        # Pareto convergence: store per-objective min and mean vectors in a rolling window.
        self._pareto_history_window: int = 5
        self._pareto_min_history: deque[npt.NDArray[np.float64]] = deque(
            maxlen=self._pareto_history_window
        )
        self._pareto_mean_history: deque[npt.NDArray[np.float64]] = deque(
            maxlen=self._pareto_history_window
        )

    @property
    def current_generation(self) -> int:
        return self._current_generation

    @property
    def iteration_progress(self) -> float:
        return self._current_generation / self._max_generations

    @property
    def info(self) -> str:
        return self._info

    def running(self) -> bool:
        """Checks if the loop controller should run."""
        running = True

        if self._is_max_generation_reached():
            self._info = (
                f"Max Generation {self._max_generations} reached, "
                "stopping optimization loop."
            )
            running = False
        elif self._is_max_stall_reached():
            self._info = (
                f"Maximum stall generations {self._base_stall} reached, "
                "stopping optimization loop."
            )
            running = False

        return running

    def increment_generation(self) -> None:
        self._current_generation += 1
        self._update_stall()

    def _update_stall(self) -> None:
        if self._current_generation < 1:
            return

        current_best = self._solution_updater_service.global_best_result

        if self._is_multi_objective:
            has_improved = self._pareto_has_improved(current_best)
        else:
            has_improved = self._single_objective_has_improved(current_best)

        if has_improved:
            self._stall_left = self._base_stall
        else:
            self._stall_left -= 1

        if not self._is_multi_objective:
            self._last_run_global_best_result = (
                float(current_best)
                if not isinstance(current_best, np.ndarray)
                else float(current_best.ravel()[0])
            )

    def _single_objective_has_improved(
        self, current_best: float | npt.NDArray[np.float64]
    ) -> bool:
        """Return True if the scalar best value changed since the last generation."""
        last = self._last_run_global_best_result
        if last is None:
            return True

        current_scalar = (
            float(current_best)
            if not isinstance(current_best, np.ndarray)
            else float(np.ravel(current_best)[0])
        )

        threshold = 1e-8
        denom = abs(last) if abs(last) > threshold else 1.0
        return abs(current_scalar - last) / denom > threshold

    def _pareto_has_improved(self, current_front: npt.NDArray[np.float64]) -> bool:
        """Return True if the Pareto front changed meaningfully since the last generation."""
        front = np.atleast_2d(np.asarray(current_front, dtype=float))

        per_obj_min = front.min(axis=0)  # shape (n_obj,)
        per_obj_mean = front.mean(axis=0)  # shape (n_obj,)

        self._pareto_min_history.append(per_obj_min)
        self._pareto_mean_history.append(per_obj_mean)

        if len(self._pareto_min_history) < 2:
            return True

        # Deque automatically handles maxlen, so [-2] safely accesses previous entry
        prev_min = self._pareto_min_history[-2]
        prev_mean = self._pareto_mean_history[-2]

        threshold = 1e-4

        denom_min = np.where(np.abs(prev_min) > threshold, np.abs(prev_min), 1.0)
        denom_mean = np.where(np.abs(prev_mean) > threshold, np.abs(prev_mean), 1.0)

        rel_change = max(
            float(np.max(np.abs(per_obj_min - prev_min) / denom_min)),
            float(np.max(np.abs(per_obj_mean - prev_mean) / denom_mean)),
        )

        return rel_change > threshold

    def _is_max_generation_reached(self) -> bool:
        return self._current_generation >= self._max_generations

    def _is_max_stall_reached(self) -> bool:
        return self._stall_left <= 0


class SolutionUpdaterService:
    def __init__(
        self,
        optimization_engine: OptimizationEngine,
        max_generations: int,
        max_stall_generations: int,
        objectives: dict[str, OptimizationStrategy],
        seed: int | None = None,
    ) -> None:
        """
        Initializes the SolutionUpdaterService with specified optimization engine and parameters.
        """
        self._logger = get_logger(__name__)
        self._mapper: _Mapper = _Mapper()
        self._engine: OptimizationEngineInterface = (
            OptimizationEngineFactory.get_engine(optimization_engine, seed=seed)
        )
        self._objectives = objectives

        is_multi_objective = len(objectives) > 1
        self._report_generator = ReportGenerator(is_multi_objective=is_multi_objective)

        self._control_vector_logger: logging.Logger | None = None
        self._population_logger: logging.Logger | None = None

        self.loop_controller = _SolutionUpdaterServiceLoopController(
            max_generations=max_generations,
            max_stall_generations=max_stall_generations,
            solution_updater_service=self,
            is_multi_objective=is_multi_objective,
        )

    @property
    def global_best_result(self) -> npt.NDArray[np.float64]:
        """Return best result(s) found by the engine."""
        res = ensure_not_none(self._engine, "Engine not initialized").global_best_result
        return np.atleast_1d(np.asarray(res, dtype=np.float64))

    @property
    def global_best_result_descriptive(
        self,
    ) -> dict[str, float] | list[dict[str, float]]:
        """Return best result(s) keyed by objective name."""
        global_best = np.atleast_1d(np.asarray(self.global_best_result, dtype=float))

        names = self._mapper.results_name

        if global_best.ndim == 2:
            n_obj = global_best.shape[1]
            if len(names) != n_obj:
                raise ValueError(
                    f"results_name has {len(names)} entries but Pareto front "
                    f"has {n_obj} objectives: {names!r}"
                )
            return [{k: float(v) for k, v in zip(names, row)} for row in global_best]

        if len(names) != len(global_best):
            raise ValueError(
                f"results_name has {len(names)} entries but global best "
                f"has {len(global_best)} values: {names!r}"
            )
        return {k: float(v) for k, v in zip(names, global_best)}

    @property
    def global_best_control_vector(self) -> ControlVector | list[ControlVector]:
        """Return best control vector(s) found by the engine."""
        raw = ensure_not_none(
            self._engine, "Engine not initialized"
        ).global_best_control_vector

        if raw.ndim == 2:
            return self._mapper.to_control_vectors(raw)

        return self._mapper.to_control_vectors(raw[np.newaxis, :])[0]

    def process_request(
        self, request_dict: dict[str, Any]
    ) -> SolutionUpdaterServiceResponse:
        self._logger.info("Processing control vectors update request...")

        config = SolutionUpdaterServiceRequest(**request_dict)

        if not config.solution_candidates:
            raise RuntimeError("Nothing to optimize")

        if not self._mapper.is_initialized:
            self._mapper.initialize(config.solution_candidates)

        control_vector, cost_function_values = self._mapper.to_numpy(
            config.solution_candidates
        )

        lb, ub = self._mapper.get_variables_lb_and_ub_boundary(
            config.optimization_constrains.boundaries
        )

        engine = ensure_not_none(self._engine, "Engine not initialized.")
        iteration_ratio = self.loop_controller.iteration_progress
        indexed_objectives_strategy = self._get_indexed_strategy()
        A_np, b_np = self._get_linear_inequalities_matrices(config, control_vector)

        if (
            A_np is not None
            and b_np is not None
            and self._engine_supports_linear_constraints()
        ):
            updated_params = engine.update_solution_to_next_iter(
                control_vector,
                cost_function_values,
                lb,
                ub,
                indexed_objectives_strategy,
                A=A_np,
                b=b_np,
                iteration_ratio=iteration_ratio,
            )
        else:
            updated_params = engine.update_solution_to_next_iter(
                control_vector,
                cost_function_values,
                lb,
                ub,
                indexed_objectives_strategy,
                iteration_ratio=iteration_ratio,
            )

        global_best_result = ensure_not_none(
            self._engine, "Engine not initialized"
        ).global_best_result

        population_statistic = PopulationStatisticGenerator.generate(
            self.loop_controller.current_generation,
            cost_function_values,
            indexed_objectives_strategy,
        )

        self._report_generator.log_control_vector_and_values(
            generation=self.loop_controller.current_generation,
            control_vector=control_vector,
            population_statistic=population_statistic,
            parameters_name=self._mapper.parameters_name,
            results_name=self._mapper.results_name,
        )

        self._report_generator.log_best_result(
            generation=self.loop_controller.current_generation,
            best_result=global_best_result,
            results_name=self._mapper.results_name,
        )

        self._report_generator.log_population_statistic(
            generation=self.loop_controller.current_generation,
            population_statistic=population_statistic,
            results_name=self._mapper.results_name,
        )

        next_iter_solutions = self._mapper.to_control_vectors(updated_params)

        self._logger.info("Control vectors update request processed successfully.")

        return SolutionUpdaterServiceResponse(next_iter_solutions=next_iter_solutions)

    def _engine_supports_linear_constraints(self) -> bool:
        """Return True if the engine's update method accepts A and b keyword args."""
        sig = inspect.signature(self._engine.update_solution_to_next_iter)
        return "A" in sig.parameters and "b" in sig.parameters

    def _get_indexed_strategy(self) -> dict[int, OptimizationStrategy]:
        indexed_objectives_strategy: dict[int, OptimizationStrategy] = {}

        results_mapping = self._mapper.results_mapping

        for obj_name, strategy in self._objectives.items():
            if obj_name in results_mapping:
                idx = results_mapping[obj_name]
                indexed_objectives_strategy[idx] = strategy
            else:
                raise ValueError(
                    f"Objective '{obj_name}' not found in results {self._mapper.results_name}"
                )
        return indexed_objectives_strategy

    def _get_linear_inequalities_matrices(
        self,
        config: SolutionUpdaterServiceRequest,
        control_vector: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | tuple[None, None]:
        if not config.optimization_constrains.linear_inequalities:
            return None, None

        if not self._mapper.is_initialized:
            raise ValueError("Control vector mapping state is not initialized.")

        self._logger.info("Linear inequalities provided, processing...")
        simple_to_idx = {
            fk: idx for fk, idx in self._mapper.control_vector_mapping.items()
        }

        A_rows = []
        for row in config.optimization_constrains.linear_inequalities.A:
            dense = np.zeros(control_vector.shape[1], dtype=np.float64)
            for var, coef in row.items():
                if var not in simple_to_idx:
                    raise ValueError(
                        f"Linear inequality variable '{var}' not found in "
                        f"control vector mapping {simple_to_idx}"
                    )
                dense[simple_to_idx[var]] = float(coef)
            A_rows.append(dense)

        A_np = np.vstack(A_rows)
        b_np = np.array(
            config.optimization_constrains.linear_inequalities.b,
            dtype=np.float64,
        )

        senses = config.optimization_constrains.linear_inequalities.sense
        if senses is not None:
            for i, s in enumerate(senses):
                if s in (">", ">="):
                    A_np[i, :] *= -1.0
                    b_np[i] *= -1.0

        return A_np, b_np
