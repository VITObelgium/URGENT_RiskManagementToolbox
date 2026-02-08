from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Self

import psutil
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from common import OptimizationStrategy
from common.models.linear_inequalities import LinearInequalities
from logger import get_logger
from services.solution_updater_service import ControlVector
from services.well_management_service import WellModel

logger = get_logger(__name__)


class VariableBnd(BaseModel, extra="forbid"):
    lb: float = Field(default=float("-inf"))
    ub: float = Field(default=float("inf"))

    @model_validator(mode="after")
    def check_bounds(self) -> Self:
        if self.lb > self.ub:
            raise ValueError("Variable bounds invalid: lb must be <= ub")
        return self


type VariableName = str
type OptimizationConstraintsMap = dict[
    VariableName, VariableBnd | OptimizationConstraintsMap
]


class WellPlacementItem(BaseModel, extra="forbid"):
    well_name: str
    initial_state: WellModel
    optimization_constraints: OptimizationConstraintsMap

    @model_validator(mode="before")
    @classmethod
    def set_initial_state_well_name(cls, values: Any) -> Any:
        if (
            isinstance(values, dict)
            and "well_name" in values
            and "initial_state" in values
        ):
            values["initial_state"]["name"] = values["well_name"]
        return values


class OptimizationParameters(BaseModel, extra="forbid"):
    """
    Represents the optimization parameters for the problem dispatcher service.

    Attributes:
        max_generations (int): The maximum number of generations for the optimization algorithm.
        population_size (int): The size of the population in each generation.
        patience (int): The number of generations to wait for improvement before stopping.
        worker_count (int): The number of parallel workers to use for simulations.
        optimization_strategy (str): The direction of the optimization objective,
            either 'maximize' or 'minimize'.
        linear_inequalities (LinearInequalities | None): The linear inequality constraints
            for the optimization problem.
    """

    max_generations: int = Field(default=10, gt=0)
    population_size: int = Field(default=10, gt=0)
    patience: int = Field(default=10, gt=0)
    worker_count: int = Field(default=4, gt=0)
    optimization_strategy: OptimizationStrategy = Field(
        default=OptimizationStrategy.MAXIMIZE,
    )
    linear_inequalities: LinearInequalities | None = Field(default=None)

    @field_validator("worker_count", mode="before")
    @classmethod
    def validate_worker_count(cls, value: int) -> int:
        physical_cores = psutil.cpu_count(logical=False)
        worker_count = max(1, math.floor(physical_cores / 2))
        if value > worker_count:
            raise ValueError(
                f"worker_count {value} exceeds available physical cores {physical_cores}"
            )
        return value

    @model_validator(mode="after")
    def validate_worker_count_not_greater_than_population_size(
        self,
    ) -> Self:
        if self.worker_count > self.population_size:
            self.worker_count = self.population_size
            logger.warning(
                f"Worker_count {self.worker_count} exceeds population_size {self.population_size}. Setting worker_count to population_size."
            )
        return self


class ServiceType(StrEnum):
    # mapping between service and optimization problem
    WellManagementService = "well_placement"


def _unique_items(seq: list[WellPlacementItem]) -> list[WellPlacementItem]:
    """Ensures that all well names in the sequence are unique."""
    names = [w.well_name for w in seq]
    if len(names) != len(set(names)):
        dup = next(n for n in names if names.count(n) > 1)
        raise ValueError(f"Duplicate well_name detected: {dup}")
    return seq


UniqueWellList = Annotated[list[WellPlacementItem], AfterValidator(_unique_items)]


class ProblemDispatcherDefinition(BaseModel, extra="forbid"):
    well_placement: UniqueWellList
    optimization_parameters: OptimizationParameters

    @model_validator(mode="after")
    def check_linear_inequalities_constraints_compliance(self) -> Self:
        lin = self.optimization_parameters.linear_inequalities
        if not lin:
            return self

        def _has_nested_path(d: dict, path: list[str]) -> bool:
            node = d
            for p in path:
                if not isinstance(node, dict) or p not in node:
                    return False
                node = node[p]
            return True

        constraints_by_well = {
            w.well_name: (w.optimization_constraints or {}) for w in self.well_placement
        }

        for A_row in lin.A:
            for key in A_row.keys():
                var = key
                well_name, attr_path = var.split(".", 1)
                top_attr = attr_path.split(".", 1)[0]
                if well_name not in constraints_by_well:
                    raise ValueError(
                        f"linear_inequalities reference unknown well: {well_name}"
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


type ServiceRequest = list[
    dict[str, Any]
]  # generic type which should be compiled with service(s) request


class RequestPayload(BaseModel, extra="forbid"):
    request: ServiceRequest
    control_vector: ControlVector


class SolutionCandidateServicesTasks(BaseModel, extra="forbid"):
    """
    Represents a collection of tasks mapped to their respective service types.

    Attributes:
        tasks (dict[ServiceType, RequestPayload]): A dictionary mapping each service type
        to its corresponding payload containing service requests and control vectors.
    """

    tasks: dict[ServiceType, RequestPayload]


class ProblemDispatcherServiceResponse(BaseModel, extra="forbid"):
    solution_candidates: list[SolutionCandidateServicesTasks]
