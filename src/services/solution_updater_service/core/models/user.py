from __future__ import annotations

from collections import OrderedDict
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field, model_validator

from services.shared import validate_boundaries, validate_linear_inequalities

type OptimizationVariableName = str
type CostFunctionName = str
type LowerBound = float
type UpperBound = float


class OptimizationEngine(StrEnum):
    PSO = "PSO"


class OptimizationConstrains(BaseModel, extra="forbid"):
    boundaries: dict[OptimizationVariableName, tuple[LowerBound, UpperBound]]
    A: list[dict[OptimizationVariableName, float]] | None = Field(default=None)
    b: list[float] | None = Field(default=None)
    sense: list[str] | None = Field(default=None)

    @model_validator(mode="after")
    def validate_constraints(self) -> OptimizationConstrains:
        for variable, (lower_bound, upper_bound) in self.boundaries.items():
            validate_boundaries(lower_bound, upper_bound)
        validate_linear_inequalities(self.A, self.b, self.sense)
        return self


class ControlVector(BaseModel, extra="forbid"):
    items: dict[OptimizationVariableName, float]


class CostFunctionResults(BaseModel, extra="forbid"):
    values: dict[CostFunctionName, float]


class SolutionCandidate(BaseModel, extra="forbid"):
    control_vector: ControlVector
    cost_function_results: CostFunctionResults


class SolutionUpdaterServiceRequest(BaseModel, extra="forbid"):
    solution_candidates: list[SolutionCandidate]
    parameter_bounds: OptimizationConstrains | None = Field(default=None)

    @model_validator(mode="after")
    def validate_solution_candidates_contain_the_same_cost_functions(
        self,
    ) -> SolutionUpdaterServiceRequest:
        cost_functions = self.solution_candidates[0].cost_function_results.values.keys()
        for candidate in self.solution_candidates:
            if candidate.cost_function_results.values.keys() != cost_functions:
                raise ValueError(
                    "All solution candidates must have the same cost functions."
                )
        return self

    @model_validator(mode="after")
    def validate_solution_candidates_contain_the_same_optimization_variables(
        self,
    ) -> SolutionUpdaterServiceRequest:
        optimization_variables = self.solution_candidates[0].control_vector.items.keys()
        for candidate in self.solution_candidates:
            if candidate.control_vector.items.keys() != optimization_variables:
                raise ValueError(
                    "All solution candidates must have the same parameters."
                )
        return self

    @model_validator(mode="after")
    def validate_optimization_boundaries_contain_the_same_optimization_variables(
        self,
    ) -> SolutionUpdaterServiceRequest:
        if self.parameter_bounds is None:
            return self
        bounded_optimization_variables = self.parameter_bounds.boundaries.keys()
        for candidate in self.solution_candidates:
            if candidate.control_vector.items.keys() != bounded_optimization_variables:
                raise ValueError(
                    "Optimization boundaries must contain the same keys as the optimization variables."
                )
        return self

    @model_validator(mode="after")
    def reorder_solution_candidates_optimization_variables_and_cost_functions(
        self,
    ) -> SolutionUpdaterServiceRequest:
        """
        Reorders the control vector parameters and metrics alphabetically so that all solution
        candidates have the same consistent order.
        """
        # Get the keys for control vector and cost function metrics from the first candidate
        optimization_variables = sorted(
            self.solution_candidates[0].control_vector.items.keys()
        )
        cost_functions = sorted(
            self.solution_candidates[0].cost_function_results.values.keys()
        )

        for candidate in self.solution_candidates:
            # Reorder control vector items
            candidate.control_vector.items = OrderedDict(
                (key, candidate.control_vector.items.get(key, np.nan))
                for key in optimization_variables
            )

            # Reorder cost function metric values
            candidate.cost_function_results.values = OrderedDict(
                (key, candidate.cost_function_results.values.get(key, np.nan))
                for key in cost_functions
            )

        return self


class SolutionUpdaterServiceResponse(BaseModel, extra="forbid"):
    next_iter_solutions: list[ControlVector]
