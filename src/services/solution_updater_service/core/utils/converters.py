from typing import Mapping

import numpy as np
import numpy.typing as npt

type Param = str
type Idx = int


def get_numpy_values(
    data: Mapping[Param, float],
) -> npt.NDArray[np.float64]:
    """
    Extracts a NumPy array of float values from a dictionary of optimization parameters.

    Args:
        data (Mapping[Param, float]): A dictionary mapping optimization parameters (str)
                                      to their corresponding float values.

    Returns:
        npt.NDArray[np.float64]: A 1D NumPy array containing the float values from the input dictionary.
    """
    return np.fromiter(data.values(), dtype=np.float64)


def get_mapping(
    data: Mapping[Param, float],
) -> Mapping[Param, Idx]:
    """
    Creates a mapping dictionary from optimization parameters to their indices.

    Args:
        data (dict): A dictionary where each key is an optimization parameter (str),
                     and the value is a float representing its corresponding value.

    Returns:
        dict: A dictionary mapping each optimization parameter (str) to an index (int).
    """
    keys, _ = zip(*data.items())
    return {key: idx for idx, key in enumerate(keys)}


def numpy_to_dict(
    values: npt.NDArray[np.float64],
    mapping: Mapping[Param, Idx],
) -> Mapping[str, float]:
    """
    Converts a NumPy array of values and a mapping dictionary back into a dictionary
    where each key corresponds to an optimization parameter and each value is its
    associated float value.

    Args:
        values (npt.NDArray[np.float64]): A NumPy array containing float values.
        mapping (dict): A dictionary mapping optimization parameters (str) to indices (int).

    Returns:
        dict: A dictionary where each key is an optimization parameter (str), and each
              value is the corresponding float value obtained from the NumPy array.
    """
    return {key: float(values[idx]) for key, idx in mapping.items()}
