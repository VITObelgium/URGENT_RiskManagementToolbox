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
    DomainServiceName,
    LinearInequalities,
    ServiceRequest,
)
from services.solution_updater_service import ControlVector

logger = get_logger(__name__)

type VariableName = str
type ObjectiveFnName = str
type ParameterBoundaries = dict[
    VariableName,
    Boundaries | ParameterBoundaries,
]


class DomainServiceItem(BaseModel, extra="forbid"):
    """One optimizable item owned by a domain service (e.g. a well)."""

    name: str
    initial_state: dict[str, Any]
    parameter_bounds: ParameterBoundaries | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def set_initial_state_name(cls, values: dict[str, Any]) -> dict[str, Any]:
        values["initial_state"]["name"] = values["name"]
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

    Available built-in names: connector="opendarts", optimizer="pso",
    domain_services=["well_design"] (demo second service: "well_counter").
    """

    connector: str
    optimizer: str
    domain_services: list[DomainServiceName] = Field(min_length=1)

    @field_validator("domain_services")
    @classmethod
    def validate_unique_domain_services(
        cls, value: list[DomainServiceName]
    ) -> list[DomainServiceName]:
        if duplicates := _duplicate_names(value):
            raise ValueError(
                f"plugins.domain_services must be unique. Duplicate: {duplicates}"
            )
        return value


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


def _duplicate_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    return sorted({n for n in names if n in seen or seen.add(n)})  # type: ignore[func-returns-value]


def _unique_items(seq: list[DomainServiceItem]) -> list[DomainServiceItem]:
    """Ensures that all item names in the sequence are unique."""
    if duplicates := _duplicate_names([item.name for item in seq]):
        raise ValueError(f"Duplicate item name detected: {duplicates}")
    return seq


UniqueItemList = Annotated[list[DomainServiceItem], AfterValidator(_unique_items)]


class ProblemDispatcherDefinition(BaseModel, extra="forbid"):
    """Validated run configuration.

    ``domain_services`` maps each configured domain service plugin name to the
    items that service owns; its keys must match ``plugins.domain_services``
    exactly. Each service contributes one column to the simulation payload.
    """

    run_mode: RunMode = Field(default=RunMode.Optimization)
    domain_services: dict[DomainServiceName, UniqueItemList]
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
    def validate_domain_services_match_plugins(self) -> Self:
        configured = set(self.plugins.domain_services)
        defined = set(self.domain_services)
        if configured != defined:
            missing = sorted(configured - defined)
            extra = sorted(defined - configured)
            raise ValueError(
                "domain_services sections must match plugins.domain_services "
                f"exactly (missing sections: {missing}, unconfigured sections: {extra})"
            )
        return self

    @model_validator(mode="after")
    def validate_parameter_bounds_for_task(self) -> Self:
        if self.run_mode == RunMode.Optimization:
            missing = [
                f"{service}.{item.name}"
                for service, items in self.domain_services.items()
                for item in items
                if not item.parameter_bounds
            ]
            if missing:
                raise ValueError(
                    "parameter_bounds are required when run-mode is 'optimization' "
                    f"(missing for items: {', '.join(missing)})"
                )
        return self

    @model_validator(mode="after")
    def check_linear_inequalities_constraints_compliance(self) -> Self:
        """Validate that every inequality variable resolves to a declared bound.

        Variables are named ``<domain_service>.<item>.<attribute path>``; rows
        may mix variables from different domain services.
        """
        if (
            self.run_mode == RunMode.Evaluation
            or not self.optimization_parameters.linear_inequalities
        ):
            return self

        def _has_nested_path(d: dict[str, Any], path: list[str]) -> bool:
            node: Any = d
            for p in path:
                if not isinstance(node, dict) or p not in node:
                    return False
                node = node[p]
            return True

        bounds_by_item = {
            service: {item.name: (item.parameter_bounds or {}) for item in items}
            for service, items in self.domain_services.items()
        }

        for A_row in self.optimization_parameters.linear_inequalities.A:
            for key in A_row.keys():
                service, _, var = key.partition(".")
                if service not in bounds_by_item or not var:
                    raise ValueError(
                        f"Unknown domain service prefix in inequality variable {key!r}. "
                        f"Configured domain services: {sorted(bounds_by_item)}"
                    )
                item_name, _, attr_path = var.partition(".")
                if not attr_path:
                    raise ValueError(
                        f"Inequality variable {key!r} must be of the form "
                        "'<domain_service>.<item>.<attribute path>'"
                    )
                if item_name not in bounds_by_item[service]:
                    raise ValueError(
                        f"Linear inequalities reference unknown item: "
                        f"'{item_name}' in domain service '{service}'"
                    )
                item_bounds = bounds_by_item[service][item_name]
                if not _has_nested_path(item_bounds, attr_path.split(".")):
                    raise ValueError(
                        f"Constraint path missing for variable '{var}' in domain "
                        f"service '{service}'"
                    )

        return self


class RequestPayload(BaseModel, extra="forbid"):
    request: ServiceRequest
    control_vector: ControlVector


class SolutionCandidateServicesTasks(BaseModel, extra="forbid"):
    tasks: dict[DomainServiceName, RequestPayload]


class ProblemDispatcherServiceResponse(BaseModel, extra="forbid"):
    solution_candidates: list[SolutionCandidateServicesTasks]
