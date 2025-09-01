from __future__ import annotations

import numpy as np
import numpy.typing as npt


def reflect_and_clip(
    values: npt.NDArray[np.float64],
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Reflect values at bound violations and clip into [lb, ub].

    Supports 1D (shape: [k]) or 2D (shape: [n, k]) arrays. lb/ub must be broadcastable to values.

    Parameters:
        values: The input values to reflect and clip.
        lb: The lower bounds.
        ub: The upper bounds.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]: A tuple of (reflected_values, out_of_bounds_mask) where mask matches the shape of values.
    """
    arr = np.asarray(values, dtype=np.float64)
    lb_arr = np.asarray(lb, dtype=np.float64)
    ub_arr = np.asarray(ub, dtype=np.float64)

    below = arr < lb_arr
    above = arr > ub_arr

    reflected = np.where(below, 2.0 * lb_arr - arr, arr)
    reflected = np.where(above, 2.0 * ub_arr - reflected, reflected)

    clipped = np.clip(reflected, lb_arr, ub_arr)
    mask = below | above
    return clipped, mask


def repair_against_linear_inequalities(
    values: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
    *,
    max_iter: int = 10,
    tol: float = 1e-3,
) -> npt.NDArray[np.float64]:
    """Greedy projection/repair of values to satisfy A x <= b with box constraints.

    Works for batched 2D arrays (n, k) or single 1D vectors (k,).

    Parameters:
        values: The input values to repair.
        A: The matrix representing the linear inequalities.
        b: The vector representing the right-hand side of the inequalities.
        lb: The lower bounds for the variables.
        ub: The upper bounds for the variables.
        max_iter: The maximum number of iterations for the repair process.
        tol: The tolerance for considering a constraint satisfied.

    Returns:
        The repaired values.
    """
    arr = np.asarray(values, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]

    arr, _ = reflect_and_clip(arr, lb, ub)

    for _ in range(max_iter):
        Ax = (A @ arr.T).T
        violations = Ax - b
        violation_mask = violations > tol
        if not np.any(violation_mask):
            break

        for i in range(arr.shape[0]):
            if not np.any(violation_mask[i]):
                continue
            idx = int(np.argmax(violations[i]))
            a_row = A[idx]
            excess = violations[i, idx]
            denom = float(np.dot(a_row, a_row))
            if denom > 1e-12:
                arr[i] = arr[i] - (excess / denom) * a_row
        arr, _ = reflect_and_clip(arr, lb, ub)

    if single:
        return arr[0]
    return arr
