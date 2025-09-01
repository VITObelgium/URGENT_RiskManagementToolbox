import copy
from typing import Any, Callable, MutableMapping

import numpy as np

from services.solution_updater_service.core.utils import (
    repair_against_linear_inequalities,
)


def update_initial_state(
    initial_state: dict[str, Any], update_dict: dict[str, Any]
) -> dict[str, Any]:
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
    flat_dict: dict[str, float], separator: str = "#"
) -> dict[str, Any]:
    def _merge_nested_dict(base: MutableMapping, keys: list[str], d_value: Any):
        current = base
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = d_value

    result: dict[str, Any] = {}
    for flat_key, value in flat_dict.items():
        keys = flat_key.split(separator)
        _merge_nested_dict(result, keys, value)
    return result


class CandidateGenerator:
    @staticmethod
    def generate(
        constraints: dict[str, tuple[float, float]],
        n_size: int,
        random_fn: Callable[[float, float], float],
        linear_inequalities: dict[str, list] | None = None,
        separator: str = "#",
        tol: float = 1e-3,
        max_repair_iter: int = 20,
    ) -> list[dict[str, float]]:
        """
        Generate n_size candidates within per-variable constraints. If sparse linear_inequalities
        are provided (A, b, sense), ensures that the subset of variables they reference
        additionally satisfies A x <= b.


        Parameters:
            constraints: A mapping of fully-qualified flat keys to (lb, ub) tuples, e.g."well_placement#INJ#md": (2000, 2700)
            n_size: The number of candidate solutions to generate.
            random_fn: A function to generate random numbers within a given range.
            linear_inequalities: Optional linear inequality constraints. See README.md for declaration and usage.
            separator: The separator used in the fully-qualified keys.
            tol: Tolerance for constraint satisfaction.
            max_repair_iter: Maximum number of repair iterations for constraint satisfaction.

        Returns:
            A list of candidate solutions, each represented as a dictionary of variable assignments.
        """

        # No linear constraints scenario
        if not linear_inequalities:
            return [
                {key: random_fn(lb, ub) for key, (lb, ub) in constraints.items()}
                for _ in range(n_size)
            ]

        # Scenario with linear constraints provided
        keys = list(constraints.keys())
        lbs = {k: float(constraints[k][0]) for k in keys}
        ubs = {k: float(constraints[k][1]) for k in keys}

        def sample_one() -> dict[str, float]:
            return {k: float(random_fn(lbs[k], ubs[k])) for k in keys}

        A_rows: list[dict[str, float]] = linear_inequalities["A"]
        b_vals: list[float] = linear_inequalities["b"]
        senses: list[str] = linear_inequalities.get("sense", ["<="] * len(A_rows))

        # Extract involved sparse variable names like "INJ.md", "PRO.md"
        sparse_vars: list[str] = []
        for row in A_rows:
            for v in row.keys():
                if v not in sparse_vars:
                    sparse_vars.append(v)

        # Map "INJ.md" -> "well_placement#INJ#md"
        def fk(var: str) -> str:
            well, attr = var.split(".", 1)
            return f"well_placement{separator}{well}{separator}{attr}"

        full_keys = [fk(v) for v in sparse_vars]
        # Filter out vars not present in constraints
        valid_mask = [k in constraints for k in full_keys]
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

        # Generate candidate solutions
        candidates: list[dict[str, float]] = []
        for _ in range(n_size):
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

        return candidates
