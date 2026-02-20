from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from common import OptimizationStrategy
from logger import get_csv_logger, get_logger
from services.shared import Boundaries
from services.solution_updater_service.core.engines import (
    GenerationSummary,
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
from services.solution_updater_service.core.utils import (
    ensure_not_none,
    get_mapping,
    get_numpy_values,
    numpy_to_dict,
)

type Param = str
type Idx = int


class _MapperState:
    """
    Represents the internal state used for mapping between structured dictionary-based
    data and their corresponding representations in NumPy arrays, enabling efficient
    optimization-related operations.

    The class provides key elements for handling and transforming data for population-level
    candidates in optimization workflows. `_MapperState` acts as the backbone of internal
    data representation for `_Mapper`.

    Attributes:
        control_vector_mapping (Mapping[Param, Idx]):
            A dictionary-like mapping where keys represent parameter names
            (strings) from the control vector, and values represent their corresponding
            indices (`int`) in the NumPy array representation.

        results_mapping (Mapping[Param, Union[Idx, Sequence[Idx]]]):
            A dictionary-like mapping where keys represent parameter names (strings)
            from the result set, and values represent either a single index (`int`) or
            a sequence of indices (`Sequence[int]`) in the NumPy results array. This allows
            handling both scalar results and structured data that spans multiple result indices.

        control_vector_length (int):
            The total number of control vector parameters. Derived from the length of
            the `control_vector_mapping`.

        results_length (int):
            The total number of result parameters. Derived from the length of the
            `results_mapping`.

        population_size (int):
            The number of individuals or candidates in the optimization problem's
            population. Used to define the size of data structures for storing
            population-wide data.
    """

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
    """
    A utility class for mapping between structured data used in optimization,
    enabling transformations between dictionary-based and NumPy array-based
    representations of control vectors and results.

    The `Mapper` class is designed to facilitate efficient handling of optimization
    parameters and results for a population of candidates, particularly when dealing
    with large datasets or computational optimizations.
    """

    @property
    def is_initialized(self) -> bool:
        if not self._state:
            return False
        return True

    @property
    def parameters_name(self) -> list[str]:
        m = ensure_not_none(self._state, "Mapper state is not initialized.")
        return list(m.control_vector_mapping.keys())

    @property
    def control_vector_mapping(self) -> Mapping[str, int]:
        m = ensure_not_none(self._state, "Mapper state is not initialized.")
        return m.control_vector_mapping

    @property
    def results_name(self) -> list[str]:
        m = ensure_not_none(self._state, "Mapper state is not initialized.")
        return list(m.results_mapping.keys())

    def __init__(self) -> None:
        self._state: _MapperState | None = None

    def to_numpy(
        self, candidates: Sequence[SolutionCandidate]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Converts a sequence of `SolutionCandidate` objects into NumPy arrays for
        control vectors and cost function results.

        This method processes a list of `SolutionCandidate` objects (one for each
        candidate in the optimization process) and returns two NumPy arrays:
          1. A 2D NumPy array where each row represents the values of the control
             vector for a candidate.
          2. A 2D NumPy array where each row contains the cost function results
             associated with a candidate.

        Args:
            candidates (Sequence[SolutionCandidate]):
                A list or other sequence of `SolutionCandidate` objects, each representing
                an individual or candidate in the optimization population. Each candidate
                contains:
                - A control vector (`control_vector`) with parameter values.
                - Cost function results (`cost_function_results`) providing the optimization metrics.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
                A tuple of two NumPy arrays:
                - The first array (2D) contains the control vector values for each candidate.
                  Shape: `(population_size, control_vector_length)`.
                - The second array (2D) contains the cost function results for each candidate.
                  Shape: `(population_size, results_length)`.

        Raises:
            RuntimeError:
                If no `candidates` are provided or if the `_Mapper` state cannot be
                initialized. The `_state` must be initialized before utilizing this method.

        Notes:
            - The method assumes the `SolutionCandidate` objects to have structured
              and valid `control_vector` and `cost_function_results`.
            - To map the data efficiently into the `Mapper`'s internal representation,
              the `control_vector` and `cost_function_results` of the first candidate
              are used to determine the field mappings (`control_vector_mapping` and
              `results_mapping`).

        Example:
            ```python
            # Sample candidates
            candidates = [
                SolutionCandidate(
                    control_vector={"param1": 1.0, "param2": 2.0},
                    cost_function_results={"cost1": 5.0, "cost2": 10.0},
                ),
                SolutionCandidate(
                    control_vector={"param1": 1.5, "param2": 2.5},
                    cost_function_results={"cost1": 6.0, "cost2": 8.0},
                ),
            ]

            # Convert candidates to NumPy arrays
            control_vector_array, results_array = mapper.to_numpy(candidates)

            # `control_vector_array`:
            # [[1.0, 2.0],
            #  [1.5, 2.5]]

            # `results_array`:
            # [[5.0, 10.0],
            #  [6.0, 8.0]]
            ```
        """
        if not self._state:
            self._state = _Mapper._initiate_mapper_on_first_call(candidates)

        candidates_control_vector_array: npt.NDArray[np.float64] = np.empty(
            shape=(self._state.population_size, self._state.control_vector_length)
        )

        candidates_results_array: npt.NDArray[np.float64] = np.empty(
            shape=(self._state.population_size, self._state.results_length)
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
        """
        Converts a 2D NumPy array of parameter values into a list of `ControlVector` objects.

        This method maps each row of a 2D NumPy array, where each row corresponds to a
        candidate's parameter values, back into a structured `ControlVector` representation.
        The mapping leverages the internal state of the `Mapper`, specifically the
        `control_vector_mapping`, which provides the correspondence between control
        vector indices and their parameter names.

        Args:
            parameters2d (npt.NDArray[np.float64]):
                A 2D NumPy array where each row holds the parameter values for an individual candidate
                in the population. The shape of the array should be
                `(population_size, control_vector_length)`.

        Returns:
            list[ControlVector]:
                A list of `ControlVector` objects, with each object representing the set of
                parameter values for an individual candidate. Each `ControlVector` is restored
                to its structured, dictionary-based representation.

        Raises:
            RuntimeError:
                If the internal `Mapper` state (`_state`) is not initialized, which is required
                to provide the `control_vector_mapping`. The `to_numpy` method must be called
                before this to initialize the state.

        Notes:
            - The `Mapper` must maintain a valid internal state, including the
              `control_vector_mapping`, which defines the relationship between
              parameter names and their array indices.
            - The number of columns in the `parameters2d` array must match the
              `control_vector_length` defined in the internal state.

        Example:
            ```python
            # Example of a 2D NumPy array with two candidates
            parameters2d = np.array([
                [1.0, 2.0],  # Candidate 1 parameter values
                [1.5, 2.5],  # Candidate 2 parameter values
            ])

            # Assume control vector mapping: {'param1': 0, 'param2': 1}
            control_vectors = mapper.to_control_vectors(parameters2d)

            # `control_vectors` will be:
            # [
            #     ControlVector(items={'param1': 1.0, 'param2': 2.0}),
            #     ControlVector(items={'param1': 1.5, 'param2': 2.5}),
            # ]
            ```
        """

        if not self._state:
            raise RuntimeError(
                "Mapper state is not initialized, call 'to_numpy' first."
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
        """
        Retrieves the lower and upper boundary values for control vector parameters
        in the optimization process.

        This function returns NumPy arrays representing the per-parameter lower and
        upper bounds for the control vector used in the optimization process. If boundary
        values are not specified or missing, it defaults to `-np.inf` (negative infinity)
        for the lower bound and `np.inf` (positive infinity) for the upper bound.

        Args:
            boundaries (OptimizationConstrains | None):
                An `OptimizationBoundaries` object containing boundary constraints for the
                control vector parameters. The boundaries include per-parameter lower
                and upper bounds structured as key-value pairs, where the key is the
                parameter name and the value is a tuple of the form `(lower_bound, upper_bound)`.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
                A tuple containing two 1D NumPy arrays:
                - The first array represents the lower bounds (`lb`) for each parameter
                  in the control vector.
                - The second array represents the upper bounds (`ub`) for each parameter
                  in the control vector.

        Raises:
            RuntimeError:
                If the internal `Mapper` state is not initialized. This occurs if the
                `to_numpy` method has not been called before this function.

        Notes:
            - If the `boundaries` argument is None or does not contain any items,
              the function defaults to `-np.inf` for all lower bounds and `np.inf` for
              all upper bounds.
            - The function matches boundary constraints with control vector indices
              using the `control_vector_mapping` attribute from the internal state.
        """

        if not self._state:
            raise RuntimeError(
                "Mapper state is not initialized, call 'to_numpy' first."
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
    ) -> None:
        """
        Helper class to control the loop of the solution updater service.
        """
        self._info = "Loop controller running"
        self._solution_updater_service = solution_updater_service

        self._max_generations = max_generations
        self._current_generation = 0

        self._base_stall, self._stall_left = (
            max_stall_generations,
            max_stall_generations,
        )

        # Track history for multi-objective
        self._last_run_global_best_result: float | npt.NDArray[np.float64] | None = None
        self._is_multi_objective: bool | None = None
        self._pareto_front_history: list[npt.NDArray[np.float64]] = []

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
        """Checks if the loop controller should run"""
        running = True

        if self._is_max_generation_reached():
            self._info = f"Max Generation {self._max_generations} reached, stopping optimization loop."
            running = False
        elif self._is_max_stall_reached():
            self._info = f"Maximum stall generations {self._base_stall} reached, stopping optimization loop."
            running = False

        return running

    def increment_generation(self) -> None:
        self._current_generation += 1
        self._update_stall()

    def _update_stall(self) -> None:
        if self._current_generation < 1:
            return

        current_best = self._solution_updater_service.global_best_result

        # Detect if multi-objective on first run
        if self._is_multi_objective is None:
            self._is_multi_objective = isinstance(current_best, np.ndarray)

        if self._is_multi_objective:
            # Multi-objective: use Pareto front convergence
            has_improved = self._has_pareto_converged(current_best)
        else:
            # Single-objective: direct comparison
            last_best = self._last_run_global_best_result
            has_improved = last_best != current_best if last_best is not None else True

        if has_improved:
            self._stall_left = self._base_stall
        else:
            self._stall_left -= 1

        self._last_run_global_best_result = (
            current_best.copy()
            if isinstance(current_best, np.ndarray)
            else current_best
        )

    def _has_pareto_converged(
        self, current_best: float | npt.NDArray[np.float64]
    ) -> bool:
        """
        Check if Pareto front has converged using a moving window approach.

        Returns True if improvement detected (NOT converged yet).
        """
        # Store current front
        if isinstance(current_best, np.ndarray):
            self._pareto_front_history.append(current_best.copy())

        # Need at least 2 iterations to compare
        if len(self._pareto_front_history) < 2:
            return True  # Still improving (not converged)

        # Keep only recent history (e.g., last 5 iterations)
        max_history = 5
        if len(self._pareto_front_history) > max_history:
            self._pareto_front_history.pop(0)

        # Compare last two fronts
        previous = self._pareto_front_history[-2]
        current = self._pareto_front_history[-1]

        # Simple metric: check if objectives changed significantly
        threshold = 1e-4

        for i in range(len(current)):
            if previous[i] != 0:
                relative_change = abs((current[i] - previous[i]) / previous[i])
            else:
                relative_change = abs(current[i] - previous[i])

            if relative_change > threshold:
                return True  # Significant change = still improving

        return False  # No significant change = converged

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

        Args:
            optimization_engine (OptimizationEngine): The optimization algorithm to use.
            max_generations (int): Maximum number of optimization iterations to perform.
            max_stall_generations (int, optional): Number of consecutive iterations without improvement
                before early stopping is triggered. Defaults to 10.
            seed (int, optional): Random seed for the optimization engine.
        """
        self._mapper: _Mapper = _Mapper()
        self._engine: OptimizationEngineInterface = (
            OptimizationEngineFactory.get_engine(optimization_engine, seed=seed)
        )
        self._objectives = objectives
        self._logger = get_logger(__name__)
        self.loop_controller = _SolutionUpdaterServiceLoopController(
            max_generations=max_generations,
            max_stall_generations=max_stall_generations,
            solution_updater_service=self,
        )

        self._control_vector_logger: logging.Logger | None = None

    @property
    def global_best_result(self) -> float | npt.NDArray[np.float64]:
        return self._engine.global_best_result

    @property
    def global_best_control_vector(self) -> ControlVector:
        # cast to array
        control_vector_array = np.array([self._engine.global_best_control_vector])
        first_index = 0
        return self._mapper.to_control_vectors(control_vector_array)[first_index]

    @property
    def parameters_name(self) -> list[str]:
        return self._mapper.parameters_name

    @property
    def results_name(self) -> list[str]:
        return self._mapper.results_name

    def get_generation_summary(self) -> GenerationSummary:
        return self._engine.generation_summary

    def process_request(
        self, request_dict: dict[str, Any]
    ) -> SolutionUpdaterServiceResponse:
        self._logger.info("Processing control vectors update request...")
        config = SolutionUpdaterServiceRequest(**request_dict)

        if not config.solution_candidates:
            raise RuntimeError("Nothing to optimize")

        control_vector, cost_function_values = self._mapper.to_numpy(
            config.solution_candidates
        )

        self._log_control_vector_and_values(control_vector, cost_function_values)

        indexed_objectives_strategy = self._get_indexed_strategy()

        lb, ub = self._mapper.get_variables_lb_and_ub_boundary(
            config.optimization_constrains.boundaries
        )

        A_np, b_np = self._get_linear_inequalities_matrices(config, control_vector)

        engine = ensure_not_none(self._engine)
        iteration_ratio = self.loop_controller.iteration_progress
        if A_np is not None and b_np is not None:
            try:
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
            except TypeError:
                # Fallback if engine does not support A,b
                self._logger.warning(
                    "Engine does not support A,b, using default update_solution_to_next_iter"
                )
                updated_params = engine.update_solution_to_next_iter(
                    control_vector,
                    cost_function_values,
                    lb,
                    ub,
                    indexed_objectives_strategy,
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

        next_iter_solutions = self._mapper.to_control_vectors(updated_params)

        self._logger.info("Control vectors update request processed successfully.")

        return SolutionUpdaterServiceResponse(next_iter_solutions=next_iter_solutions)

    def _get_indexed_strategy(self) -> dict[int, OptimizationStrategy]:
        indexed_objectives_strategy: dict[int, OptimizationStrategy] = {}
        for obj_name, strategy in self._objectives.items():
            # Find the index of this objective in the results
            if obj_name in self._mapper.results_name:
                idx = self._mapper.results_name.index(obj_name)
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
                        f"Linear inequality variable '{var}' not found in control vector mapping {simple_to_idx}"
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

    def _log_control_vector_and_values(
        self, control_vector, cost_function_values
    ) -> None:
        if self._control_vector_logger is None:
            self._control_vector_logger = get_csv_logger(
                "control_vector_logger.csv",
                logger_name="control_vector_logger",
                columns=["generation", "individual"]
                + [p for p in self._mapper.parameters_name]
                + [r for r in self._mapper.results_name],
            )

        for idx, val in enumerate(np.hstack((control_vector, cost_function_values))):
            v_str = ",".join(f"{v:.9f}" for v in val)
            self._control_vector_logger.info(
                f"{self.loop_controller.current_generation},{idx},{v_str}"
            )
