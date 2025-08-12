from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from common import OptimizationStrategy
from services.problem_dispatcher_service.core.models import ControlVector, WellModel


class VariableBnd(BaseModel, extra="forbid"):
    lb: float = Field(default=float("-inf"))
    ub: float = Field(default=float("inf"))


class MDBnd(BaseModel, extra="forbid"):
    lb: float = Field(default=0.0)
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
        linear_inequalities (dict[str, list] | None): The linear inequality constraints
            for the optimization problem. An example structure is:
            {
                "A": [
                    {"INJ.md": 1.0, "PRO.md": 1.0}, {"INJ.md": 1.0, "PRO.md": 1.0}
                ],
                "b": [30.0, 3000.0],
                "sense": [">=", "<="]  # optional list, defaults to "<=" if omitted
            }

    """

    optimization_strategy: OptimizationStrategy = Field(
        default=OptimizationStrategy.MAXIMIZE,
    )
    total_md_len: MDBnd | None = Field(default=None)
    linear_inequalities: dict[str, list] | None = Field(default=None)

    @model_validator(mode="after")
    def validate_linear_inequalities(self) -> OptimizationParameters:
        if self.linear_inequalities is None:
            return self
        try:
            A = self.linear_inequalities["A"]
            b = self.linear_inequalities["b"]
            senses = self.linear_inequalities.get("sense")
        except KeyError:
            raise KeyError("linear_inequalities must contain both 'A' and 'b'")

        if not isinstance(A, list) or not isinstance(b, list):
            raise TypeError("'A' and 'b' in linear_inequalities must be lists")
        if len(A) != len(b):
            raise ValueError("Number of rows in A must match length of b")

        if senses is not None:
            if not isinstance(senses, list):
                raise TypeError("'sense' must be a list when provided")
            if len(senses) != len(A):
                raise ValueError("Length of 'sense' must match number of rows in A")
            allowed = {"<=", ">=", "<", ">"}
            if not all(s in allowed for s in senses):
                raise ValueError(
                    f"Invalid inequality direction(s) in 'sense'. Allowed: {sorted(allowed)}"
                )
        else:
            # As default when not specified, we set all senses to "<="
            self.linear_inequalities["sense"] = ["<="] * len(A)

        # Collect attribute suffixes to ensure same attribute referenced (e.g., all '.md')
        attr_suffix: str | None = None
        for row_idx, row in enumerate(A):
            if not isinstance(row, dict):
                raise TypeError(
                    f"Row {row_idx} in A must be a dict mapping variable to coefficient"
                )
            if len(row) == 0:
                raise ValueError(f"Row {row_idx} in A is empty")
            for var, coef in row.items():
                if not isinstance(coef, (int, float)):
                    raise TypeError(
                        f"Coefficient for {var} in row {row_idx} must be numeric"
                    )
                if "." not in var:
                    raise ValueError(
                        f"Variable '{var}' in linear inequalities must contain a '.' separating well and attribute (e.g., 'INJ.md')"
                    )
                suffix = var.split(".", 1)[1]
                if attr_suffix is None:
                    attr_suffix = suffix
                elif suffix != attr_suffix:
                    raise ValueError(
                        "All variables in linear_inequalities must refer to the same attribute (e.g., all '.md'). "
                        f"Found both '{attr_suffix}' and '{suffix}'."
                    )
            if not isinstance(b[row_idx], (int, float)):
                raise TypeError(f"b[{row_idx}] must be numeric")
        return self


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
