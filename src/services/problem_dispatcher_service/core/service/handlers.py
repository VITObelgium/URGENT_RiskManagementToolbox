from typing import Any, Protocol

from services.problem_dispatcher_service.core.models import (
    ParameterBoundaries,
    WellDesignItem,
)
from services.shared import Boundaries, ServiceType


class ProblemTypeHandler(Protocol):
    """
    Protocol for problem handlers, defining methods required to handle specific optimization problems.
    """

    def build_initial_state(self, items: list[Any]) -> dict[str, Any]:
        """
        Build the initial state for the optimization problem.

        Args:
            items (list[Any]): A list of items to build the initial state from.

        Returns:
            dict[str, Any]: A dictionary representing the initial state.
        """

        ...

    def build_full_key_boundaries(
        self, items: list[Any], separator: str = "#"
    ) -> dict[str, Boundaries]:
        """
        Build the constraints for the optimization problem.

        Args:
            items (list[Any]): A list of items to build constraints from.
            separator (str): Separator used to construct constraint keys.


        Returns:
            dict[str, tuple[float, float]]: A dictionary of constraints with keys as identifiers
            and values as the bounds (lower, upper).
        """
        ...

    def build_service_tasks(
        self, solution_items: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Build service task payloads that are required after solving the optimization problem.

        Args:
            solution_items (dict[str, Any]): The solution items provided by the optimization process.

        Returns:
            list[dict[str, Any]]: A list of service task dictionaries.
        """

        ...


class WellDesignHandler(ProblemTypeHandler):
    def build_initial_state(self, items: list[WellDesignItem]) -> dict[str, Any]:
        return {item.well_name: item.initial_state.model_dump() for item in items}

    def build_full_key_boundaries(
        self,
        items: list[WellDesignItem],
        separator: str = "#",
    ) -> dict[str, Boundaries]:
        """
        Build the constraints for the well placement optimization problem.
        Args:
            items (list[WellDesignItem]): A list of well placement items.
            separator (str): Separator used to construct constraint keys.
        Returns:
            dict[str, tuple[float, float]]: A dictionary of constraints with keys as identifiers
            and values as the bounds (lower, upper).
        """

        result = {}
        for item in items:
            # Add existing constraints (e.g., wellhead x, y)
            flattened = _flatten_optimization_parameters(item.parameter_bounds)
            for key, value in flattened.items():
                result[
                    f"{ServiceType.WellDesignService}#{item.well_name}{separator}{key}"
                ] = Boundaries(**{"lb": value[0], "ub": value[1]})
        return result

    def build_service_tasks(
        self, solution_items: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return list(solution_items.values())


def _flatten_optimization_parameters(
    optimization_parameters: ParameterBoundaries,
    parent_key: str = "",
    separator: str = "#",
) -> dict[str, tuple[float, float]]:
    flat: dict[str, tuple[float, float]] = {}
    for key, value in optimization_parameters.items():
        full_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, Boundaries):
            flat[full_key] = (value.lb, value.ub)
        elif isinstance(value, dict):
            nested_flat = _flatten_optimization_parameters(value, full_key, separator)
            flat.update(nested_flat)
        else:
            raise TypeError(
                f"Invalid type for key '{key}': expected VariableBnd or dict, got {type(value)}"
            )
    return flat
