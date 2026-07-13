from typing import Any

import numpy as np

from common import OptimizationStrategy
from common.models import RunMode
from logger import get_logger
from services.problem_dispatcher_service.core.builder import TaskBuilder
from services.problem_dispatcher_service.core.models import (
    LinearInequalities,
    ProblemDispatcherDefinition,
    ProblemDispatcherServiceResponse,
)
from services.problem_dispatcher_service.core.utils import (
    DEFAULT_SEPARATOR,
    CandidateGenerator,
    build_full_key_boundaries,
    build_initial_state,
    convert_key_separator,
)
from services.shared import Boundaries
from services.solution_updater_service import ControlVector

logger = get_logger(__name__)


class ProblemDispatcherService:
    """Turns a problem definition into per-iteration solution-candidate tasks.

    Everything derivable from the config (initial state, optimizer boundary
    map, linear inequality constraints) is built once in ``__init__`` so
    ``process_iteration`` only has to pick control vectors and build tasks.
    """

    def __init__(self, problem_definition: ProblemDispatcherDefinition):
        self._definition = problem_definition
        self._params = problem_definition.optimization_parameters
        self._is_evaluation = problem_definition.run_mode == RunMode.Evaluation

        self._initial_state: dict[str, Any] = build_initial_state(
            problem_definition.domain_services
        )
        self._task_builder = TaskBuilder(self._initial_state)

        self._full_key_boundaries: dict[str, Boundaries] = (
            {}
            if self._is_evaluation
            else build_full_key_boundaries(problem_definition.domain_services)
        )
        self._full_key_linear_inequalities = (
            None if self._is_evaluation else self._convert_linear_inequalities()
        )

        logger.debug(
            "ProblemDispatcherService initialized. Initial state: %s; "
            "boundaries: %s; linear inequalities: %s",
            self._initial_state,
            self._full_key_boundaries,
            self._full_key_linear_inequalities,
        )

    @property
    def optimization_objectives(self) -> dict[str, OptimizationStrategy]:
        return self._params.objectives or {}

    @property
    def expected_optimization_function_names(self) -> list[str]:
        return list(self.optimization_objectives.keys())

    @property
    def max_generation(self) -> int:
        return self._params.max_generations

    @property
    def population_size(self) -> int:
        return self._params.population_size

    @property
    def max_stall_generations(self) -> int:
        return self._params.max_stall_generations

    @property
    def full_key_boundaries(self) -> dict[str, Boundaries]:
        return self._full_key_boundaries

    @property
    def full_key_linear_inequalities(self) -> LinearInequalities | None:
        return self._full_key_linear_inequalities

    def process_iteration(
        self, next_iter_solutions: list[ControlVector] | None = None
    ) -> ProblemDispatcherServiceResponse:
        """Generate or forward control vectors for one optimization iteration.

        Args:
            next_iter_solutions: Optimizer-updated vectors from the previous
                iteration, or ``None`` to generate the initial population.

        Returns:
            Response containing one ``SolutionCandidateServicesTasks`` per candidate.
        """
        control_vectors = self._control_vectors(next_iter_solutions)
        solution_candidates = self._task_builder.build(control_vectors)
        logger.info(
            "Iteration processed: %d solution candidates.", len(solution_candidates)
        )
        return ProblemDispatcherServiceResponse(solution_candidates=solution_candidates)

    def _control_vectors(
        self, next_iter_solutions: list[ControlVector] | None
    ) -> list[dict[str, Any]]:
        if self._is_evaluation:
            if next_iter_solutions and len(next_iter_solutions) > 1:
                logger.warning(
                    "Evaluation run-mode received %d control vectors; "
                    "only the first will be used.",
                    len(next_iter_solutions),
                )
            return [next_iter_solutions[0].items] if next_iter_solutions else [{}]

        if next_iter_solutions is None:
            rng = np.random.default_rng(self._params.seed)
            control_vectors = CandidateGenerator.generate(
                self._full_key_boundaries,
                self.population_size,
                rng.uniform,
                self._initial_state,
                self._params.linear_inequalities,
            )
            logger.debug("Generated %d initial control vectors.", len(control_vectors))
            return control_vectors

        logger.debug("Using %d provided control vectors.", len(next_iter_solutions))
        return [cv.items for cv in next_iter_solutions]

    def _convert_linear_inequalities(self) -> LinearInequalities | None:
        source = self._params.linear_inequalities
        if source is None:
            return None
        return LinearInequalities(
            A=[
                {
                    convert_key_separator(key, output_separator=DEFAULT_SEPARATOR): v
                    for key, v in row.items()
                }
                for row in source.A
            ],
            b=source.b,
            sense=source.sense,
        )
