import copy
from collections.abc import MutableMapping, Sequence
from typing import Any, cast
from collections.abc import Callable

import numpy as np

from services.problem_dispatcher_service.core.utils.keys import (
    DEFAULT_SEPARATOR,
    convert_key_separator,
    join_key,
    split_key,
)
from services.shared import Boundaries, LinearInequalities, ServiceType
from services.solution_updater_service.core.utils import (
    repair_against_linear_inequalities,
)


def is_bounds_dict(value: Any) -> bool:
    """Check if a value is a Boundaries-like dict with 'lb' and 'ub' keys."""
    return (
        isinstance(value, dict) and "lb" in value and "ub" in value and len(value) == 2
    )


def validate_bounds(lb: float, ub: float, key: str = "") -> None:
    """
    Validate that lower bound is less than or equal to upper bound.

    Args:
        lb: Lower bound value
        ub: Upper bound value
        key: Optional key/path for error message context

    Raises:
        ValueError: If lb > ub
    """
    if lb > ub:
        key_info = f" for '{key}'" if key else ""
        raise ValueError(
            f"Lower bound must be less than or equal to upper bound{key_info}. "
            f"Got lb={lb}, ub={ub}"
        )


def validate_bounds_finite(lb: float, ub: float, key: str = "") -> None:
    """
    Validate that bounds are finite values.

    Args:
        lb: Lower bound value
        ub: Upper bound value
        key: Optional key/path for error message context

    Raises:
        ValueError: If bounds are not finite
    """
    if not (np.isfinite(lb) and np.isfinite(ub)):
        key_info = f" for {key}" if key else ""
        raise ValueError(
            f"Invalid boundary{key_info}: bounds must be finite. Got lb={lb}, ub={ub}"
        )


def validate_bounds_recursive(bounds: dict[str, Any], path: str = "") -> None:
    """
    Recursively validate that all boundary dicts have lb <= ub.

    Args:
        bounds: Dictionary containing nested boundary definitions
        path: Current path in the nested structure (for error messages)

    Raises:
        ValueError: If any boundary has lb > ub
    """
    for key, value in bounds.items():
        current_path = f"{path}.{key}" if path else key
        if isinstance(value, dict):
            if is_bounds_dict(value):
                validate_bounds(value["lb"], value["ub"], current_path)
            else:
                validate_bounds_recursive(value, current_path)


def update_initial_state(
    initial_state: dict[str | ServiceType, Any], update_dict: dict[str, Any]
) -> dict[str | ServiceType, Any]:
    """
    Recursively updates a deep copy of the initial_state dictionary with values
    from update_dict. If a value in update_dict is a dictionary, the function
    merges it with the corresponding dictionary in initial_state.

    This function ensures immutability of the input initial_state by using
    deepcopy.
    """
    updated_dict = copy.deepcopy(initial_state)
    for key, value in update_dict.items():
        if isinstance(value, dict) and isinstance(updated_dict.get(key), dict):
            updated_dict[key] = update_initial_state(updated_dict[key], value)
        else:
            updated_dict[key] = copy.deepcopy(value)
    return updated_dict


def parse_flat_dict_to_nested(
    flat_dict: dict[str, float] | dict[str, None],
    separator: str = DEFAULT_SEPARATOR,
) -> dict[str, Any]:
    def _merge_nested_dict(base: MutableMapping, keys: list[str], d_value: Any):
        current = base
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = d_value

    result: dict[str, Any] = {}
    for flat_key, value in flat_dict.items():
        keys = split_key(flat_key, separator)
        _merge_nested_dict(result, keys, value)
    return result


def get_corresponding_initial_state_as_flat_dict(
    initial_state: dict[str, Any],
    variable_source: list[str],
    separator: str = DEFAULT_SEPARATOR,
):
    template = {k: None for k in variable_source}
    nestest_template = parse_flat_dict_to_nested(template, separator)
    template_with_initial_state = _fill_none(nestest_template, initial_state)
    return _flatten_dict(template_with_initial_state, separator=separator)


def _fill_none(target, source):
    """
    Recursively replace None values in `target` with values from `source`.
    Modifies `target` in place and also returns it.
    """
    for key, value in target.items():
        if key not in source:
            continue

        if value is None:
            target[key] = source[key]

        elif isinstance(value, dict) and isinstance(source[key], dict):
            _fill_none(value, source[key])

    return target


def _flatten_dict(d, parent_key="", separator: str = DEFAULT_SEPARATOR):
    flat = {}

    for key, value in d.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict):
            flat.update(_flatten_dict(value, new_key, separator))
        else:
            flat[new_key] = value

    return flat


def flatten_optimization_parameters(
    optimization_parameters: dict[str, Any],
    parent_key: str = "",
    separator: str = DEFAULT_SEPARATOR,
) -> dict[str, tuple[float, float]]:
    """Recursively flatten a nested parameter-boundaries dict into (lb, ub) tuples."""
    flat: dict[str, tuple[float, float]] = {}
    for key, value in optimization_parameters.items():
        full_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, Boundaries):
            flat[full_key] = (value.lb, value.ub)
        elif isinstance(value, dict):
            if is_bounds_dict(value):
                lb = cast(float, value["lb"])
                ub = cast(float, value["ub"])
                validate_bounds(lb, ub, full_key)
                flat[full_key] = (lb, ub)
            else:
                flat.update(flatten_optimization_parameters(value, full_key, separator))
        else:
            raise TypeError(
                f"Invalid type for key '{key}': expected Boundaries or dict, got {type(value)}"
            )
    return flat


def build_initial_state_from_well_design(items: list[Any]) -> dict[str, Any]:
    """Build the initial state dict from a list of WellDesignItem objects."""
    return {item.well_name: item.initial_state.model_dump() for item in items}


def build_full_key_boundaries_from_well_design(
    items: list[Any], separator: str = DEFAULT_SEPARATOR
) -> dict[str, Boundaries]:
    """Build the full-key boundaries dict from a list of WellDesignItem objects."""
    result: dict[str, Boundaries] = {}
    for item in items:
        if not item.parameter_bounds:
            continue
        flattened = flatten_optimization_parameters(item.parameter_bounds)
        for key, value in flattened.items():
            full_key = join_key(
                str(ServiceType.DomainService),
                item.well_name,
                key,
                separator=separator,
            )
            result[full_key] = Boundaries(lb=value[0], ub=value[1])
    return result


class CandidateGenerator:
    @staticmethod
    def generate(
        full_key_boundaries: dict[str, Boundaries],
        n_size: int,
        random_fn: Callable[[float, float], float],
        initial_state: dict[str, Any],
        linear_inequalities: LinearInequalities | None = None,
        separator: str = DEFAULT_SEPARATOR,
        tol: float = 1e-3,
        max_repair_iter: int = 20,
    ) -> list[dict[str, float]]:
        """
        Generate n_size candidates within per-variable constraints. If sparse linear_inequalities
        are provided (A, b, sense), ensures that the subset of variables they reference
        additionally satisfies A x <= b.


        Parameters:
            full_key_boundaries: A mapping of fully-qualified flat keys to (lb, ub) tuples, e.g."well_design#INJ#md": (2000, 2700)
            n_size: The number of candidate solutions to generate.
            random_fn: A function to generate random numbers within a given range.
            initial_state: The initial state of the problem.
            linear_inequalities: Optional linear inequality constraints. See README.md for declaration and usage.
            separator: The separator used in the fully-qualified keys.
            tol: Tolerance for constraint satisfaction.
            max_repair_iter: Maximum number of repair iterations for constraint satisfaction.

        Returns:
            A list of candidate solutions, each represented as a dictionary of variable assignments.
        """

        for key, bnd in full_key_boundaries.items():
            validate_bounds(bnd.lb, bnd.ub, key)
            validate_bounds_finite(bnd.lb, bnd.ub, key)

        user_initial_candidate = get_corresponding_initial_state_as_flat_dict(
            initial_state, list(full_key_boundaries.keys()), separator=separator
        )

        for key, val in user_initial_candidate.items():
            bnd = full_key_boundaries[key]
            lb, ub = bnd.lb, bnd.ub
            if not (lb - tol <= val <= ub + tol):
                raise ValueError(
                    f"Initial user state for {key} is out of bounds: {val} (Bounds: [{lb}, {ub}])"
                )

        # No linear constraints scenario
        if not linear_inequalities:
            if not linear_inequalities:
                random_candidates = [
                    {
                        key: random_fn(b.lb, b.ub)
                        for key, b in full_key_boundaries.items()
                    }
                    for _ in range(n_size - 1)
                ]
                return [user_initial_candidate] + random_candidates

        # Scenario with linear constraints provided
        keys = list(full_key_boundaries.keys())
        lbs = {k: float(full_key_boundaries[k].lb) for k in keys}
        ubs = {k: float(full_key_boundaries[k].ub) for k in keys}

        def sample_one() -> dict[str, float]:
            return {k: float(random_fn(lbs[k], ubs[k])) for k in keys}

        A_rows: list[dict[str, float]] = linear_inequalities.A
        b_vals: list[float] = linear_inequalities.b
        senses: Sequence[str] = linear_inequalities.sense

        # Extract involved sparse variable names like "INJ.md", "PRO.md"
        sparse_vars: list[str] = []
        for row in A_rows:
            for v in row.keys():
                if v not in sparse_vars:
                    sparse_vars.append(v)

        # Map "INJ.md" -> "well_design#INJ#md"
        def fk(var: str) -> str:
            return convert_key_separator(var, output_separator=separator)

        full_keys = [fk(v) for v in sparse_vars]
        # Filter out vars not present in constraints
        valid_mask = [k in full_key_boundaries for k in full_keys]
        sparse_vars = [v for v, m in zip(sparse_vars, valid_mask) if m]
        full_keys = [k for k, m in zip(full_keys, valid_mask) if m]

        if not full_keys:
            return [sample_one() for _ in range(n_size)]

        m = len(A_rows)
        k = len(full_keys)
        A = np.zeros((m, k), dtype=float)
        b = np.array(b_vals, dtype=float)

        # Normalize senses: convert >=, > to <= by multiplying by -1
        senses_norm = []
        for i, s in enumerate(senses):
            s = s.strip()
            senses_norm.append(s)
        for i, row in enumerate(A_rows):
            for j, sv in enumerate(sparse_vars):
                coef = float(row.get(sv, 0.0))
                A[i, j] = coef

        # Convert strict inequalities to non-strict with tiny slack
        slack = 1e-9
        for i, s in enumerate(senses_norm):
            if s in (">", ">="):
                # multiply row by -1 to convert to <=
                A[i, :] *= -1.0
                b[i] *= -1.0
                if s == ">":
                    b[i] -= slack
            elif s == "<":
                b[i] -= slack
            # "<=" remains as is

        lb_vec = np.array([lbs[k] for k in full_keys], dtype=float)
        ub_vec = np.array([ubs[k] for k in full_keys], dtype=float)

        x_user = np.array([user_initial_candidate[k] for k in full_keys], dtype=float)
        if np.any((A.dot(x_user) - b) > tol):
            raise ValueError(
                "User provided initial state violates linear inequality constraints."
            )

        # Generate candidate solutions
        candidates: list[dict[str, float]] = []
        for _ in range(n_size - 1):
            c = sample_one()
            x0 = np.array([c[k] for k in full_keys], dtype=float)
            xr = repair_against_linear_inequalities(
                x0, A, b, lb_vec, ub_vec, max_iter=max_repair_iter, tol=tol
            )
            if np.any((A.dot(xr) - b) > tol):
                raise ValueError(
                    "Initial population generation failed: linear inequalities appear infeasible with given bounds."
                )

            for idx, kf in enumerate(full_keys):
                c[kf] = float(xr[idx])
            candidates.append(c)

        return [user_initial_candidate] + candidates
