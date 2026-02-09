import random
from typing import Any, Callable

from common import OptimizationStrategy
from logger import get_logger
from services.problem_dispatcher_service.core.builder import TaskBuilder
from services.problem_dispatcher_service.core.models import (
    LinearInequalities,
    ProblemDispatcherDefinition,
    ProblemDispatcherServiceResponse,
    ServiceType,
)
from services.problem_dispatcher_service.core.service.handlers import (
    ProblemTypeHandler,
    WellDesignHandler,
)
from services.problem_dispatcher_service.core.utils import CandidateGenerator
from services.shared import Boundaries
from services.solution_updater_service import ControlVector

PROBLEM_TYPE_HANDLERS: dict[ServiceType, ProblemTypeHandler] = {
    ServiceType.WellDesignService: WellDesignHandler(),
}


class ProblemDispatcherService:
    def __init__(self, problem_definition: ProblemDispatcherDefinition):
        """
        Initialize the ProblemDispatcherService.

        Args:
            problem_definition (ProblemDispatcherDefinition): The problem definition to process.
        """
        self.logger = get_logger(__name__)
        self.logger.debug("Initializing ProblemDispatcherService")

        try:
            self._problem_definition = problem_definition
            self._n_size = (
                self._problem_definition.optimization_parameters.population_size
            )
            self._handlers = PROBLEM_TYPE_HANDLERS
            self._initial_state = self._build_initial_state()

            self._linear_inequalities = (
                self._problem_definition.optimization_parameters.linear_inequalities
            )
            self._task_builder = TaskBuilder(self._initial_state, self._handlers)
            self._full_key_boundaries = self._build_full_key_boundaries()
            self._full_key_linear_inequalities = (
                self._build_full_key_linear_inequalities()
            )

            self.logger.debug("ProblemDispatcherService initialized successfully.")
        except Exception as e:
            self.logger.error(
                "Failed to initialize ProblemDispatcherService: %s", str(e)
            )
            raise

    @property
    def optimization_objectives(self) -> dict[str, OptimizationStrategy]:
        return self._problem_definition.optimization_parameters.objectives

    @property
    def max_generation(self) -> int:
        return self._problem_definition.optimization_parameters.max_generations

    @property
    def population_size(self) -> int:
        return self._n_size

    @property
    def max_stall_generations(self) -> int:
        return self._problem_definition.optimization_parameters.max_stall_generations

    @property
    def full_key_boundaries(self) -> dict[str, Boundaries]:
        return self._full_key_boundaries

    @property
    def full_key_linear_inequalities(self) -> LinearInequalities | None:
        return self._full_key_linear_inequalities

    def process_iteration(
        self, next_iter_solutions: list[ControlVector] | None = None
    ) -> ProblemDispatcherServiceResponse:
        self.logger.debug("Processing iteration.")
        self.logger.debug(
            "Processing iteration. next_iter_solutions: %s",
            next_iter_solutions if next_iter_solutions else "None",
        )

        try:
            if next_iter_solutions is None:
                control_vectors = CandidateGenerator.generate(
                    self._full_key_boundaries,
                    self._n_size,
                    random.uniform,
                    self._initial_state,
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
        self.logger.debug(f"Processing problem items: {log_message}")
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
        return self._process_problem_items(
            process_func=lambda handler, items: handler.build_initial_state(items),
            log_message="Building initial state",
            merge_results=False,
        )

    def _build_full_key_boundaries(self) -> dict[str, Boundaries]:
        return self._process_problem_items(
            process_func=lambda handler, items: handler.build_full_key_boundaries(
                items
            ),
            log_message="Building boundaries",
            merge_results=True,
        )

    def _build_full_key_linear_inequalities(self) -> LinearInequalities | None:
        separator = "#"
        if self._linear_inequalities is None:
            return None
        return LinearInequalities(
            **{
                "A": [
                    {k.replace(".", separator): v for (k, v) in row.items()}
                    for row in self._linear_inequalities.A
                ],
                "b": self._linear_inequalities.b,
                "sense": self._linear_inequalities.sense,
            }
        )
