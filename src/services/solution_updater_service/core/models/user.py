from __future__ import annotations

from collections import OrderedDict
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field, model_validator

type OptimizationVariable = str
type CostFunction = str
type LowerBound = float
type UpperBound = float


class OptimizationEngine(StrEnum):
    PSO = "PSO"


class OptimizationConstrains(BaseModel, extra="forbid"):
    boundaries: dict[OptimizationVariable, tuple[LowerBound, UpperBound]]
    A: list[dict[str, float]] | None = Field(default=None)
    b: list[float] | None = Field(default=None)
    sense: list[str] | None = Field(default=None)

    @model_validator(mode="after")
    def validate_lower_bound_is_less_than_upper_bound(self) -> OptimizationConstrains:
        for variable, (lower_bound, upper_bound) in self.boundaries.items():
            if lower_bound >= upper_bound:
                raise ValueError(
                    f"Lower bound for {variable} must be less than upper bound."
                )
        if (self.A is None) ^ (self.b is None):
            raise ValueError("Both A and b must be provided together or both None")
        if self.A is not None:
            if len(self.A) != len(self.b):  # type: ignore[arg-type]
                raise ValueError("A row count must equal length of b")
            if self.sense is not None and len(self.sense) != len(self.A):
                raise ValueError("sense length must match number of A rows")
            if self.sense is None and self.A is not None:
                self.sense = ["<="] * len(self.A)
            allowed = {"<=", ">=", "<", ">"}
            for s in self.sense or []:
                if s not in allowed:
                    raise ValueError(
                        f"Invalid inequality direction '{s}'. Allowed: {sorted(allowed)}"
                    )
            suffix_ref: str | None = None
            for idx, row in enumerate(self.A):
                if not isinstance(row, dict) or not row:
                    raise ValueError(f"Row {idx} in A must be a non-empty dict")
                for var, coef in row.items():
                    if not isinstance(coef, (int, float)):
                        raise TypeError(
                            f"Coefficient for variable '{var}' in row {idx} must be numeric"
                        )
                    if "." not in var:
                        raise ValueError(
                            f"Variable '{var}' must contain '.' separating well and attribute (e.g., 'INJ.md')"
                        )
                    suffix = var.split(".", 1)[1]
                    if suffix_ref is None:
                        suffix_ref = suffix
                    elif suffix != suffix_ref:
                        raise ValueError(
                            "All variables in linear inequalities must refer to the same attribute; "
                            f"found '{suffix_ref}' and '{suffix}'."
                        )
        return self


class ControlVector(BaseModel, extra="forbid"):
    """
    Represents a control vector used in optimization processes.

    Attributes:
        items (dict[str, float]): A dictionary where keys are strings
            representing the parameter names and values are floats
            defining the parameter values.

    Notes:
        The `extra="forbid"` option ensures that only the defined fields
        are allowed when creating an instance of this model.
    """

    items: dict[OptimizationVariable, float]


class CostFunctionResults(BaseModel, extra="forbid"):
    """
    Represents the results of a cost function evaluation in the context of optimization.

    Attributes:
        values (dict[str, float]): A dictionary where the keys are identifiers
            (e.g., cost function) and the values is single float values,
            representing the evaluation results for each metric.

    Notes:
        The `extra="forbid"` option ensures that no additional fields are allowed beyond
        those explicitly defined in the model.
    """

    values: dict[CostFunction, float]


class SolutionCandidate(BaseModel, extra="forbid"):
    """
    Represents a candidate solution in the optimization process.

    Attributes:
        control_vector (ControlVector): The control vector associated with this candidate solution,
            defining the parameters and their values.
        cost_function_results (CostFunctionResults): The results of evaluating the cost function
            for this candidate solution.

    Notes:
        The `extra="forbid"` option ensures that no additional fields are allowed
        beyond those explicitly defined in the model.
    """

    control_vector: ControlVector
    cost_function_results: CostFunctionResults


class SolutionUpdaterServiceRequest(BaseModel, extra="forbid"):
    """
    Configuration for the optimization service, defining the setup for the optimization process.

    Attributes:
        solution_candidates (list[SolutionCandidate]): A list of initial solution candidates
            to be evaluated during the optimization process.

    Notes:
        The `extra="forbid"` option ensures that no additional fields are allowed
        beyond those explicitly defined in the model.
    """

    solution_candidates: list[SolutionCandidate]
    optimization_constraints: OptimizationConstrains | None = Field(default=None)

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
        if self.optimization_constraints is None:
            return self
        bounded_optimization_variables = self.optimization_constraints.boundaries.keys()
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
    """
    Represents the result of the optimization service.

    Attributes:
        next_iter_solutions (list[ControlVector]): A list of `ControlVector` instances representing
            the next iteration's optimal solutions determined by the optimization process. Each
            control vector contains the parameters and their corresponding optimized values.
        patience_exceeded (PatienceExceeded): A boolean flag indicating whether the optimization
            process has reached its patience limit. When True, it signals that the solution has not
            improved for the specified number of consecutive iterations, suggesting convergence
            or a local minimum.

    Notes:
        The `extra="forbid"` option ensures that no additional fields are allowed beyond the explicitly
        defined attributes in this model, ensuring the structure remains consistent and predictable.
    """

    next_iter_solutions: list[ControlVector]
