# dispatcher.py
import random
from typing import Any, Callable

from common import OptimizationStrategy
from logger import get_logger
from services.problem_dispatcher_service.core.builder import TaskBuilder
from services.problem_dispatcher_service.core.models import (
    ControlVector,
    ProblemDispatcherDefinition,
    ProblemDispatcherServiceResponse,
    ServiceType,
)
from services.problem_dispatcher_service.core.service.handlers import (
    ProblemTypeHandler,
    WellPlacementHandler,
)
from services.problem_dispatcher_service.core.utils import CandidateGenerator

PROBLEM_TYPE_HANDLERS: dict[str, ProblemTypeHandler] = {
    "well_placement": WellPlacementHandler(),
}

PROBLEM_TYPE_TO_SERVICE_TYPE: dict[str, ServiceType] = {
    "well_placement": ServiceType.WellManagementService,
}


class ProblemDispatcherService:
    def __init__(self, problem_definition: ProblemDispatcherDefinition):
        """
        Initialize the ProblemDispatcherService.

        Args:
            problem_definition (ProblemDispatcherDefinition): The problem definition to process.
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing ProblemDispatcherService")

        try:
            self._problem_definition = problem_definition
            self._n_size = (
                self._problem_definition.optimization_parameters.population_size
            )
            self._handlers = PROBLEM_TYPE_HANDLERS
            self._service_type_map = PROBLEM_TYPE_TO_SERVICE_TYPE
            self._initial_state = self._build_initial_state()
            self._boundaries = self._build_boundaries()
            opt_params = self._problem_definition.optimization_parameters
            self._linear_inequalities: dict[str, list] | None = getattr(
                opt_params, "linear_inequalities", None
            )
            self._task_builder = TaskBuilder(
                self._initial_state, self._handlers, self._service_type_map
            )
            self.logger.info("ProblemDispatcherService initialized successfully.")
        except Exception as e:
            self.logger.error(
                "Failed to initialize ProblemDispatcherService: %s", str(e)
            )
            raise

    @property
    def optimization_strategy(self) -> OptimizationStrategy:
        return self._problem_definition.optimization_parameters.optimization_strategy

    @property
    def max_generation(self) -> int:
        return self._problem_definition.optimization_parameters.max_generations

    @property
    def patience(self) -> int:
        return self._problem_definition.optimization_parameters.patience

    @property
    def boundaries(self) -> dict[str, tuple[float, float]]:
        return self._boundaries

    @property
    def linear_inequalities(self) -> dict[str, list] | None:
        return self._linear_inequalities

    def process_iteration(
        self, next_iter_solutions: list[ControlVector] | None = None
    ) -> ProblemDispatcherServiceResponse:
        """
        Process one iteration of generating or processing solutions.

        Args:
            next_iter_solutions: Existing solutions from the previous iteration, if any.

        Returns:
            ProblemDispatcherServiceResponse: Response containing solution candidates.
        """
        self.logger.info("Processing iteration.")
        self.logger.debug(
            "Processing iteration. next_iter_solutions: %s",
            next_iter_solutions if next_iter_solutions else "None",
        )

        try:
            if next_iter_solutions is None:
                control_vectors = CandidateGenerator.generate(
                    self._boundaries,
                    self._n_size,
                    random.uniform,
                    self._linear_inequalities,
                )
                self.logger.debug("Generated control vectors: %s", control_vectors)
            else:
                control_vectors = [cv.items for cv in next_iter_solutions]
                self.logger.debug("Using provided control vectors: %s", control_vectors)

            solution_candidates = self._task_builder.build(control_vectors)
            self.logger.info(
                "Iteration processed successfully. Generated %d solution candidates.",
                len(solution_candidates),
            )
            return ProblemDispatcherServiceResponse(
                solution_candidates=solution_candidates
            )
        except Exception as e:
            self.logger.error("Error during process_iteration: %s", str(e))
            raise

    def _process_problem_items(
        self,
        process_func: Callable[..., dict[str, Any] | Any],
        log_message: str,
        merge_results: bool = False,
    ) -> dict[str, Any]:
        """
        Shared logic to process problem items for state building or constraints.

        Args:
            process_func (Callable): Function to process the problem items.
            log_message (str): Log message indicating the operation.
            merge_results (bool): If True, merge the processed results into one dictionary.
        """
        self.logger.info(f"processing problem items: {log_message}")
        try:
            # Add type annotation for the result variable
            result: dict[str, Any] | None = {} if merge_results else None
            for problem_type, handler in self._handlers.items():
                items = getattr(self._problem_definition, problem_type, None)
                if items:
                    processed_result = process_func(handler, items)

                    if merge_results:
                        # Ensure result is always a dictionary when merge_results is True
                        if result is None:
                            result = {}  # Safeguard for type consistency
                        result.update(processed_result)
                    else:
                        # Initialize result as a dictionary if it is None
                        if result is None:
                            result = {}
                        result[problem_type] = processed_result
                    self.logger.debug(
                        "%s for %s: %s", log_message, problem_type, processed_result
                    )
            # If result is still None (no items processed), return an empty dictionary
            return result if result is not None else {}
        except Exception as e:
            self.logger.error("Error during %s: %s", log_message, str(e))
            raise

    def _build_initial_state(self) -> dict[str, Any]:
        """
        Build the initial state for the problem handler.

        Returns:
            dict[str, Any]: Initial state based on the problem definition.
        """
        return self._process_problem_items(
            process_func=lambda handler, items: handler.build_initial_state(items),
            log_message="Building initial state",
            merge_results=False,
        )

    def _build_boundaries(self) -> dict[str, tuple[float, float]]:
        """
        Build problem constraints.

        Returns:
            dict[str, tuple[float, float]]: Boundaries of the problem.
        """
        return self._process_problem_items(
            process_func=lambda handler, items: handler.build_boundaries(items),
            log_message="Building boundaries",
            merge_results=True,
        )
