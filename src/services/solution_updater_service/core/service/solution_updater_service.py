from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from logger.u_logger import get_logger
from services.solution_updater_service.core.engines import (
    OptimizationEngineFactory,
    OptimizationEngineInterface,
)
from services.solution_updater_service.core.models import (
    ControlVector,
    OptimizationConstrains,
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
        self, constrains: OptimizationConstrains | None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Retrieves the lower and upper boundary values for control vector parameters
        in the optimization process.

        This function returns NumPy arrays representing the per-parameter lower and
        upper bounds for the control vector used in the optimization process. If boundary
        values are not specified or missing, it defaults to `-np.inf` (negative infinity)
        for the lower bound and `np.inf` (positive infinity) for the upper bound.

        Args:
            constrains (OptimizationConstrains | None):
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
        if not constrains or not constrains.boundaries:
            return np.full(self._state.control_vector_length, -np.inf), np.full(
                self._state.control_vector_length, np.inf
            )

        lb = np.full(self._state.control_vector_length, -np.inf)
        ub = np.full(self._state.control_vector_length, np.inf)

        for k, (vlb, vub) in constrains.boundaries.items():
            lb[self._state.control_vector_mapping[k]] = vlb
            ub[self._state.control_vector_mapping[k]] = vub

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
    def __init__(self, max_generations: int) -> None:
        """
        Helper class to control the loop of the solution updater service.
        This class manages the number of iterations for the optimization process, and raises StopIteration exception when convergence fails.
        """
        self._max_generations = max_generations
        self.current_generation = 0
        self._is_running = True

    def running(self) -> bool:
        """
        Checks if the loop controller should run

        Returns:
            bool: True if the loop controller is running, False otherwise.
        """
        if self.current_generation >= self._max_generations:
            self._is_running = False
        self.current_generation += 1
        return self._is_running


class SolutionUpdaterService:
    def __init__(
        self, optimization_engine: OptimizationEngine, max_generations: int
    ) -> None:
        self._mapper: _Mapper = _Mapper()
        self._engine: OptimizationEngineInterface = (
            OptimizationEngineFactory.get_engine(optimization_engine)
        )
        self._logger = get_logger(__name__)
        self.loop_controller = _SolutionUpdaterServiceLoopController(max_generations)

    @property
    def global_best_result(self) -> float:
        return self._engine.global_best_result

    @property
    def global_best_controll_vector(self) -> ControlVector:
        # cast to array
        control_vector_array = np.array([self._engine.global_best_controll_vector])
        first_index = 0
        return self._mapper.to_control_vectors(control_vector_array)[first_index]

    def process_request(
        self, request_dict: dict[str, Any]
    ) -> SolutionUpdaterServiceResponse:
        """
        Updates the solution space for the next optimization iteration.

        This method processes a configuration dictionary that contains candidate solutions,
        optimization boundaries, and other necessary parameters, then interacts with the
        optimization engine to compute the updated control vectors for the next iteration
        in the optimization process.

        Steps:
        1. Converts the candidate solutions' control vectors and cost function values into
           NumPy array representations.
        2. Retrieves lower and upper boundaries for the control vector parameters.
        3. Delegates the computation of the updated parameters to the optimization engine.
        4. Maps the updated parameters from their NumPy array representations back to
           structured control vector objects.
        5. Packs the updated control vectors into an `OptimizationServiceResult` object.

        Args:
            request_dict (dict[str, Any]):
                A dictionary containing optimization configuration parameters. It must
                include the following:
                - A sequence of `SolutionCandidate` objects (`solution_candidates`),
                  representing the population of control vectors and their associated
                  cost function results.
                - `optimization_boundaries` (optional): An `OptimizationBoundaries`
                  object defining the per-parameter lower and upper limits.
                - An `optimization_engine` specification, which determines the optimization
                  algorithm used.

        Returns:
            SolutionUpdaterServiceResponse:
                An object containing the updated solution control vectors for the next
                optimization iteration. It includes:
                - `next_iter_solutions`: A list of updated `ControlVector` instances.

        Raises:
            RuntimeError:
                - If no solution candidates are provided in the configuration (`config_dict`).
                - If an initialization issue occurs with the underlying optimization engine.

        Notes:
            - The `config_dict` is deserialized into an `OptimizationServiceConfig` object,
              which provides structured access to all required optimization parameters.
            - Internally, the NumPy-based representation of control vectors enables
              efficient manipulation of large optimization data before it is re-structured
              into higher-order `ControlVector` entities.
        """
        self._logger.info("Processing control vectors update request...")
        config = SolutionUpdaterServiceRequest(**request_dict)

        if not config.solution_candidates:
            raise RuntimeError("Nothing to optimize")

        self._check_convergence(config.solution_candidates)

        control_vector, cost_function_values = self._mapper.to_numpy(
            config.solution_candidates
        )

        lb, ub = self._mapper.get_variables_lb_and_ub_boundary(
            config.optimization_constraints
        )

        updated_params = ensure_not_none(self._engine).update_solution_to_next_iter(
            control_vector, cost_function_values, lb, ub
        )

        next_iter_solutions = self._mapper.to_control_vectors(updated_params)
        self._logger.info("Control vectors update request processed successfully.")

        return SolutionUpdaterServiceResponse(next_iter_solutions=next_iter_solutions)

    def _check_convergence(
        self, solution: list[SolutionCandidate], tol: float = 1e-4
    ) -> None:
        """
        Should raise StopIteration exception when convergence reach desired value.
        Args:
            tol: function convergence tolerance
            solution: list of SolutionCandidate

        Returns:

        """
        pass
