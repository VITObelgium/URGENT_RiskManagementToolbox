import math
from typing import Iterator, Tuple, TypeVar

from plugins.domain_services.well_design import POINT_ATOL, Point, TrajectoryPoint

T = TypeVar("T", TrajectoryPoint, Point)


def is_subsequence_points(
    output: Tuple[T, ...],
    expected: Tuple[T, ...],
) -> bool:
    it: Iterator[T] = iter(output)
    return all(any(elem == item for item in it) for elem in expected)


def is_subsequence_tuple_of_float(
    output: tuple[tuple[float, float, float], ...],
    expected: tuple[tuple[float, float, float], ...],
) -> bool:
    def are_close(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
        return all(math.isclose(a[i], b[i], abs_tol=POINT_ATOL) for i in range(3))

    it: Iterator[tuple[float, float, float]] = iter(output)
    return all(any(are_close(elem, item) for item in it) for elem in expected)
