from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from services.well_management_service.core import models
from services.well_management_service.core.utilities.constants import POINT_ATOL


@dataclass(frozen=True, slots=True)
class TrajectoryPoint(models.Point):
    """
    Positive Z - downwards
    """

    md: float

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y, self.z, self.md))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrajectoryPoint):
            raise TypeError(
                f"Expected type is TrajectoryPoint, actual is {type(other)}"
            )
        return (
            math.isclose(self.x, other.x, abs_tol=POINT_ATOL)
            and math.isclose(self.y, other.y, abs_tol=POINT_ATOL)
            and math.isclose(self.z, other.z, abs_tol=POINT_ATOL)
            and math.isclose(self.md, other.md, abs_tol=POINT_ATOL)
        )

    def is_close_to_md(self, other_md: float) -> bool:
        return math.isclose(other_md, self.md, abs_tol=POINT_ATOL)


class Trajectory(tuple[TrajectoryPoint, ...]):
    """
    Trajectory object represent the set of points defining well shape in space
    """

    def __new__(cls, iterable: tuple[TrajectoryPoint, ...]) -> Trajectory:
        if len(iterable) == 0:
            raise ValueError(
                "Can't create empty trajectory. Wellhead must be provided."
            )
        return super().__new__(cls, iterable)

    def get_last_trajectory_point(self) -> TrajectoryPoint:
        """
        Retrieve the last point in the trajectory.

        Returns:
            TrajectoryPoint: The last point in the trajectory.
        """

        return self[-1]

    def get_xz_inclination(self) -> float:
        """
        The inclination is evaluated in the XZ plane, and if the movement is clockwise, the inclination is returned as negative.
        If the trajectory contains fewer than two points, a default inclination of 0.0 degree is returned.
        Otherwise, the inclination is computed using the last two points of the trajectory.

        Returns:
            Inclination in degrees, negative if the movement is clockwise in the XZ plane.
        """

        if len(self) < 2:
            return 0.0

        last_point: TrajectoryPoint = self[-1]
        second_last_point: TrajectoryPoint = self[-2]

        dz = second_last_point.z - last_point.z
        dx = second_last_point.x - last_point.x

        d = max(
            np.sqrt(dx**2 + dz**2), 1.0e-6
        )  # ensure no ZeroDivisionError() will be raised
        inclination = np.rad2deg(np.arccos(-dz / d))

        # Determine side-awareness
        # The angle's sign will give us the clockwise or counterclockwise direction.
        if dx > 0:
            inclination = -inclination  # Clockwise direction

        return float(inclination)

    def get_points_in_md_range(
        self, md_start: float, md_end: float
    ) -> tuple[models.Point, ...]:
        """
        Returns tuple of Points in md start - end range
        Parameters
        ----------
        md_start
        md_end

        Returns
        -------

        """
        tol = 1e-6
        return tuple(
            [
                models.Point(t.x, t.y, t.z)
                for t in self
                if (md_start - tol) <= t.md <= md_end + tol
            ]
        )
