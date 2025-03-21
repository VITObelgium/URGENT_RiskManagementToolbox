from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from services.well_management_service.core.utilities.constants import POINT_ATOL


@dataclass(frozen=True, slots=True)
class Point:
    """
    Positive Z - downwards
    """

    x: float
    y: float
    z: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            raise TypeError(f"Expected type is Point, actual is {type(other)}")
        return (
            math.isclose(self.x, other.x, abs_tol=POINT_ATOL)
            and math.isclose(self.y, other.y, abs_tol=POINT_ATOL)
            and math.isclose(self.z, other.z, abs_tol=POINT_ATOL)
        )

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y, self.z))

    def __str__(self) -> str:
        return f"Point({self.x}, {self.y}, {self.z})"
