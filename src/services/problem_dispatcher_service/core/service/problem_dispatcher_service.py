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
    ServiceType,
)
from services.problem_dispatcher_service.core.utils import (
    DEFAULT_SEPARATOR,
    CandidateGenerator,
    build_full_key_boundaries_from_well_design,
    build_initial_state_from_well_design,
    convert_key_separator,
)
from services.shared import Boundaries
from services.solution_updater_service import ControlVector


class ProblemDispatcherService:
    def __init__(self, problem_definition: ProblemDispatcherDefinition):
        """Initialize ProblemDispatcherService.

        Builds the initial well state, optimizer boundary map, and linear
        inequality constraints from the problem definition so that
        ``process_iteration`` can generate or forward control vectors without
        re-parsing the config on every iteration.

        Args:
            problem_definition: Validated configuration for the optimization run.
        """
        self.logger = get_logger(__name__)
        self.logger.debug("Initializing ProblemDispatcherService")

        try:
            self._problem_definition = problem_definition
            self._population_size = (
                self._problem_definition.optimization_parameters.population_size
            )
            self._linear_inequalities = (
                self._problem_definition.optimization_parameters.linear_inequalities
            )

            well_items = self._problem_definition.well_design
            self._initial_state: dict[str, Any] = {
                ServiceType.WellDesignService: build_initial_state_from_well_design(
                    well_items
                )
            }
            self.logger.debug("Initial state built: %s", self._initial_state)

            self._task_builder = TaskBuilder(self._initial_state)
            self._full_key_boundaries = self._build_full_key_boundaries()
            self.logger.debug("Full-key boundaries: %s", self._full_key_boundaries)

            self._full_key_linear_inequalities = (
                self._build_full_key_linear_inequalities()
            )
            self.logger.debug(
                "Full-key linear inequalities: %s", self._full_key_linear_inequalities
            )

            self.logger.debug("ProblemDispatcherService initialized successfully.")
        except Exception as e:
            self.logger.error(
                "Failed to initialize ProblemDispatcherService: %s", str(e)
            )
            raise

    @property
    def optimization_objectives(self) -> dict[str, OptimizationStrategy]:
        return self._problem_definition.optimization_parameters.objectives or {}

    @property
    def expected_optimization_function_names(self) -> list[str]:
        return list(self.optimization_objectives.keys())

    @property
    def max_generation(self) -> int:
        return self._problem_definition.optimization_parameters.max_generations

    @property
    def population_size(self) -> int:
        return self._population_size

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
        """Generate or forward control vectors for one optimization iteration.

        Args:
            next_iter_solutions: Optimizer-updated vectors from the previous
                iteration, or ``None`` to generate the initial population.

        Returns:
            Response containing one ``SolutionCandidateServicesTasks`` per candidate.
        """
        self.logger.debug(
            "Processing iteration. next_iter_solutions: %s",
            next_iter_solutions if next_iter_solutions else "None",
        )

        try:
            if self._problem_definition.run_mode == RunMode.Evaluation:
                if next_iter_solutions and len(next_iter_solutions) > 1:
                    self.logger.warning(
                        "Evaluation run-mode received %d control vectors; only the first will be used.",
                        len(next_iter_solutions),
                    )
                control_vectors = (
                    [next_iter_solutions[0].items] if next_iter_solutions else [{}]
                )
                self.logger.debug(
                    "Evaluation mode control vectors: %s", control_vectors
                )
            elif next_iter_solutions is None:
                rng = np.random.default_rng(
                    self._problem_definition.optimization_parameters.seed
                )
                control_vectors = CandidateGenerator.generate(
                    self._full_key_boundaries,
                    self.population_size,
                    rng.uniform,
                    self._initial_state,
                    self._linear_inequalities,
                )
                self.logger.debug(
                    "Generated %d initial control vectors.", len(control_vectors)
                )
            else:
                control_vectors = [cv.items for cv in next_iter_solutions]
                self.logger.debug(
                    "Using %d provided control vectors.", len(control_vectors)
                )

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

    def _build_full_key_boundaries(self) -> dict[str, Boundaries]:
        if self._problem_definition.run_mode == RunMode.Evaluation:
            return {}
        return build_full_key_boundaries_from_well_design(
            self._problem_definition.well_design
        )

    def _build_full_key_linear_inequalities(self) -> LinearInequalities | None:
        if (
            self._problem_definition.run_mode == RunMode.Evaluation
            or self._linear_inequalities is None
        ):
            return None
        return LinearInequalities(
            **{
                "A": [
                    {
                        convert_key_separator(k, output_separator=DEFAULT_SEPARATOR): v
                        for (k, v) in row.items()
                    }
                    for row in self._linear_inequalities.A
                ],
                "b": self._linear_inequalities.b,
                "sense": self._linear_inequalities.sense,
            }
        )
