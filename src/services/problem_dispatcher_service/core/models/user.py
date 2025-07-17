from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from common import OptimizationStrategy
from services.problem_dispatcher_service.core.models import ControlVector, WellModel


class VariableBnd(BaseModel, extra="forbid"):
    lb: float = Field(default=float("-inf"))
    ub: float = Field(default=float("inf"))


type VariableName = str
type OptimizationConstrains = dict[VariableName, VariableBnd | OptimizationConstrains]


class WellPlacementItem(BaseModel, extra="forbid"):
    well_name: str
    initial_state: WellModel
    optimization_constrains: OptimizationConstrains

    @model_validator(mode="before")
    @classmethod
    def set_initial_state_well_name(cls, values):
        values["initial_state"]["name"] = values["well_name"]
        return values


class OptimizationParameters(BaseModel, extra="forbid"):
    """
    Represents the optimization parameters for the problem dispatcher service.

    Attributes:
        optimization_strategy (str): The direction of the optimization objective,
            either 'maximize' or 'minimize'.
    """

    optimization_strategy: OptimizationStrategy = Field(
        default=OptimizationStrategy.MAXIMIZE,
    )


class ServiceType(StrEnum):
    # mapping between service and optimization problem
    WellManagementService = "well_placement"


class ProblemDispatcherDefinition(BaseModel, extra="forbid"):
    well_placement: list[WellPlacementItem]
    optimization_parameters: OptimizationParameters


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
