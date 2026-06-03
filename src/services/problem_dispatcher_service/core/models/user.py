from __future__ import annotations

from typing import Self, Any

from pathlib import Path
import psutil
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)
from pydantic.functional_validators import AfterValidator
from typing import Annotated

from common import OptimizationStrategy
from common.models import RunMode
from logger import get_logger
from services.problem_dispatcher_service.core.utils import validate_bounds_recursive
from services.shared import (
    Boundaries,
    LinearInequalities,
    ServiceRequest,
    ServiceType,
)
from services.solution_updater_service import ControlVector

logger = get_logger(__name__)

type VariableName = str
type ObjectiveFnName = str
type ParameterBoundaries = dict[
    VariableName,
    Boundaries | ParameterBoundaries,
]


class WellDesignItem(BaseModel, extra="forbid"):
    well_name: str
    initial_state: dict[str, Any]
    parameter_bounds: ParameterBoundaries | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def set_initial_state_well_name(cls, values):
        values["initial_state"]["name"] = values["well_name"]
        return values

    @model_validator(mode="after")
    def validate_parameter_bounds(self) -> Self:
        """Validate that all parameter bounds have lb <= ub."""
        if self.parameter_bounds:
            validate_bounds_recursive(self.parameter_bounds)
        return self


class SimulationConfig(BaseModel, extra="forbid"):
    """Simulation config parameters.

    Attributes:
        worker_simulation_timeout_seconds (int): Worker-side timeout in seconds for the local simulation subprocess to complete (default: 15 minutes)
        server_job_timeout_seconds (int): Server-side watchdog timeout in seconds for assigned jobs (default: 1 hour)
        worker_count (int): The number of parallel workers to use for simulations.
        checkpoint_interval (int): Save checkpoint every n generation.
    """

    server_job_timeout_seconds: PositiveInt = Field(default=3600)
    worker_simulation_timeout_seconds: PositiveInt = Field(default=900)
    worker_count: PositiveInt = Field(default=4, ge=1)
    checkpoint_interval: PositiveInt = Field(default=5, ge=1)
    checkpoint_path: DirectoryPath = Field(
        default=Path.joinpath(Path(__file__).resolve().parents[5], "checkpoints"),
        validate_default=True,
    )

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def create_log_folder_if_missing(cls, value: Any) -> Any:
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("worker_count", mode="before")
    @classmethod
    def validate_worker_count(cls, value: int) -> int:
        physical_cores = psutil.cpu_count(logical=False)
        worker_count = max(1, physical_cores)
        if value > worker_count:
            raise ValueError(
                f"worker_count {value} exceeds available physical cores {physical_cores}"
            )
        return value

    @model_validator(mode="after")
    def validate_timeouts(self) -> SimulationConfig:
        if self.server_job_timeout_seconds < self.worker_simulation_timeout_seconds:
            raise ValueError(
                "server_job_timeout_seconds must be greater than or equal to "
                "worker_simulation_timeout_seconds"
            )
        return self


class PluginConfig(BaseModel, extra="forbid"):
    """Selects which plugin implementations to use for a run.

    Available built-in names: connector="opendarts", optimizer="pso", domain_service="builtin".
    """

    connector: str
    optimizer: str
    domain_service: str


class OptimizationParameters(BaseModel, extra="forbid"):
    """
    Represents the optimization parameters for the problem dispatcher service.

    Attributes:
        objectives (dict[ObjectiveFnName, OptimizationStrategy]): The objective function(s) with their respective optimization strategy. If multiple objective functions are provided, the optimization algorithm uses pareto front approximation.
        max_generations (int): The maximum number of generations for the optimization algorithm.
        population_size (int): The size of the population in each generation.
        max_stall_generations (int): The number of generations to wait for improvement before stopping.
        linear_inequalities (LinearInequalities | None): The linear inequality constraints
            for the optimization problem. An example structure is:
                "A": [
                    {"INJ.md": 1.0, "PRO.md": 1.0}, {"INJ.md": 1.0, "PRO.md": 1.0}
                ],
                "b": [30.0, 3000.0],
                "sense": [">=", "<="]
    """

    objectives: dict[ObjectiveFnName, OptimizationStrategy] | None = Field(default=None)
    max_generations: PositiveInt = Field(default=10, ge=1)
    population_size: PositiveInt = Field(default=10, ge=2, le=200)
    max_stall_generations: PositiveInt = Field(default=10, ge=1)
    linear_inequalities: LinearInequalities | None = Field(default=None)
    seed: int | None = Field(default=None)


def _unique_items(seq: list[WellDesignItem]) -> list[WellDesignItem]:
    """Ensures that all well names in the sequence are unique."""
    names = [w.well_name for w in seq]
    if len(names) != len(set(names)):
        dup = next(n for n in names if names.count(n) > 1)
        raise ValueError(f"Duplicate well_name detected: {dup}")
    return seq


UniqueWellList = Annotated[list[WellDesignItem], AfterValidator(_unique_items)]


class ProblemDispatcherDefinition(BaseModel, extra="forbid"):
    """
    Services
    Optimization parameters
    """

    run_mode: RunMode = Field(default=RunMode.Optimization)
    well_design: UniqueWellList
    optimization_parameters: OptimizationParameters = Field(
        default_factory=OptimizationParameters
    )
    worker_simulation_timeout_seconds: int = Field(
        default=900,
        description=(
            "Worker-side timeout in seconds for the local simulation subprocess "
            "to complete (default: 15 minutes)"
        ),
    )
    simulation_config: SimulationConfig = Field(default_factory=SimulationConfig)
    plugins: PluginConfig

    @model_validator(mode="after")
    def apply_task_overrides(self) -> Self:
        """When run-mode is 'evaluation', collapse all loop-control parameters to 1."""
        if self.run_mode == RunMode.Evaluation:
            self.optimization_parameters.population_size = 1
            self.optimization_parameters.max_generations = 1
            self.optimization_parameters.max_stall_generations = 1
            self.simulation_config.worker_count = 1
        return self

    @model_validator(mode="after")
    def validate_worker_count_not_greater_than_population_size(
        self,
    ) -> Self:
        if (
            self.simulation_config.worker_count
            > self.optimization_parameters.population_size
        ):
            logger.warning(
                f"Worker_count {self.simulation_config.worker_count} exceeds population_size {self.optimization_parameters.population_size}. Setting worker_count to population_size."
            )
            self.simulation_config.worker_count = (
                self.optimization_parameters.population_size
            )
        return self

    @model_validator(mode="after")
    def validate_objectives_for_task(self) -> Self:
        if (
            self.run_mode == RunMode.Optimization
            and not self.optimization_parameters.objectives
        ):
            raise ValueError("objectives are required during optimization run.")
        return self

    @model_validator(mode="after")
    def validate_parameter_bounds_for_task(self) -> Self:
        if self.run_mode == RunMode.Optimization:
            missing = [w.well_name for w in self.well_design if not w.parameter_bounds]
            if missing:
                raise ValueError(
                    "parameter_bounds are required when run-model is 'optimization' "
                    f"(missing for wells: {', '.join(missing)})"
                )
        return self

    @model_validator(mode="after")
    def check_linear_inequalities_constraints_compliance(self) -> Self:
        if (
            self.run_mode == RunMode.Evaluation
            or not self.optimization_parameters.linear_inequalities
        ):
            return self

        def _has_nested_path(d: dict, path: list[str]) -> bool:
            node = d
            for p in path:
                if not isinstance(node, dict) or p not in node:
                    return False
                node = node[p]
            return True

        constraints_by_well = {
            w.well_name: (w.parameter_bounds or {}) for w in self.well_design
        }

        for A_row in self.optimization_parameters.linear_inequalities.A:
            for key in A_row.keys():
                service, var = key.split(".", 1)
                if service not in ServiceType:
                    raise ValueError(
                        f"Unknown service type: {service}. Make sure service type is used as prefix in variable name."
                    )
                well_name, attr_path = var.split(".", 1)
                top_attr = attr_path.split(".", 1)[0]
                if well_name not in constraints_by_well:
                    raise ValueError(
                        f"Linear inequalities reference unknown well: {well_name}"
                    )
                well_constraints = constraints_by_well[well_name]
                if top_attr not in well_constraints:
                    raise ValueError(
                        f"Well '{well_name}' missing optimization constraint for '{top_attr}'"
                    )
                if "." in attr_path:
                    parts = attr_path.split(".")
                    if not _has_nested_path(well_constraints, parts):
                        raise ValueError(
                            f"Constraint path missing for variable '{var}' in well '{well_name}'"
                        )

        return self


class RequestPayload(BaseModel, extra="forbid"):
    request: ServiceRequest
    control_vector: ControlVector


class SolutionCandidateServicesTasks(BaseModel, extra="forbid"):
    tasks: dict[ServiceType, RequestPayload]


class ProblemDispatcherServiceResponse(BaseModel, extra="forbid"):
    solution_candidates: list[SolutionCandidateServicesTasks]
