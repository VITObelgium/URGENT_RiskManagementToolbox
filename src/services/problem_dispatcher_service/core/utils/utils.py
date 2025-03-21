import copy
from typing import Any, Callable, MutableMapping


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
    ) -> list[dict[str, float]]:
        return [
            {key: random_fn(lb, ub) for key, (lb, ub) in constraints.items()}
            for _ in range(n_size)
        ]
