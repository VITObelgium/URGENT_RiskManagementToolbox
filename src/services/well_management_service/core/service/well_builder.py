from __future__ import annotations

from typing import MutableMapping, Sequence

import numpy as np

from logger import get_logger
from services.well_management_service.core.models import (
    Completion,
    Perforation,
    PerforationRange,
    Point,
    Trajectory,
    TrajectoryPoint,
    Well,
)
from services.well_management_service.core.sections.section_interface import (
    SectionInterface,
)
from services.well_management_service.utils._exceptions import (
    EmptySectionConfigurationException,
)


class WellBuilder:
    def __init__(self) -> None:
        self._name: str = "Unknown"
        self._wellhead: TrajectoryPoint = TrajectoryPoint(0, 0, 0, 0)
        self._azimuth: float = 0.0  # deg
        self._translation: Point = Point(0, 0, 0)
        self._step: float = 0.5  # m
        self._sections: tuple[SectionInterface, ...] | None = None
        self._perforations: MutableMapping[str, PerforationRange] | None = None

        self.__logger = get_logger(__name__)

    def well_name(self, name: str) -> WellBuilder:
        self.__logger.debug(f"Setting up well name: {name}")
        self._name = name
        return self

    def rotate(self, azimuth: float) -> WellBuilder:
        self.__logger.debug(f"Setting up well rotating azimuth: {azimuth} deg")
        self._azimuth = azimuth
        return self

    def translate(self, translation: Point) -> WellBuilder:
        self.__logger.debug(
            f"Setting up well translation vector: {str(TrajectoryPoint)}"
        )
        self._translation = translation
        return self

    def discretize(self, step: float) -> WellBuilder:
        self.__logger.debug(f"Setting up well trajectory sampling step: {step} m")
        self._step = step
        return self

    def sections(self, sections: Sequence[SectionInterface]) -> WellBuilder:
        self.__logger.debug("Setting up well sections...")
        self._sections = tuple([s for s in sections])
        return self

    def perforations(
        self, perforations: MutableMapping[str, PerforationRange] | None
    ) -> WellBuilder:
        self.__logger.debug("Setting up well perforation intervals...")
        if not perforations:
            self.__logger.debug("Perforation intervals are empty, skipping...")
            return self
        self._perforations = perforations
        return self

    def build(self) -> Well:
        self.__logger.debug(f"Well {self._name} building process is starting...")
        if not self._sections:
            raise EmptySectionConfigurationException(
                "Well builder doesn't contain valid sections configuration"
            )

        trajectory: Trajectory = Trajectory((self._wellhead,))

        for section in self._sections:
            trajectory = section.append_to_trajectory(
                trajectory, self._step, self._perforations
            )

        trajectory = _rotate(self._azimuth, trajectory)
        trajectory = _move(self._translation, trajectory)

        completion = _build_completion(trajectory, self._perforations)

        well = Well(
            name=self._name,
            trajectory=trajectory,
            completion=completion,
        )
        self.__logger.debug(f"Well {self._name} building process completed")
        return well


def _rotate(angle: float, trajectory: Trajectory) -> Trajectory:
    """
    Rotates the well model over the vertical axis (Z-axis) by the given angle.
    All points in the trajectory must have a y-coordinate equal to zero.
    """
    first_y = trajectory[0].y

    # cast the trajectory to x-z plane
    xz_trajectory = _move(
        translation=Point(x=0, y=-first_y, z=0), trajectory=trajectory
    )

    rotated_xz_trajectory = []
    angle = np.deg2rad(angle)

    for p in xz_trajectory:
        # Apply 2D rotation matrix to the x and y coordinates
        x_rot = np.cos(angle) * p.x - np.sin(angle) * p.y
        y_rot = np.sin(angle) * p.x + np.cos(angle) * p.y

        rotated_xz_trajectory.append(TrajectoryPoint(x=x_rot, y=y_rot, z=p.z, md=p.md))

    # cast the rotated trajectory to the initial plane
    rotated_trajectory = _move(
        translation=Point(x=0, y=first_y, z=0),
        trajectory=Trajectory(tuple(rotated_xz_trajectory)),
    )

    return Trajectory(tuple(rotated_trajectory))


def _move(translation: Point, trajectory: Trajectory) -> Trajectory:
    return Trajectory(
        tuple(
            TrajectoryPoint(
                x=p.x + translation.x,
                y=p.y + translation.y,
                z=p.z + translation.z,
                md=p.md,
            )
            for p in trajectory
        )
    )


def _build_completion(
    trajectory: Trajectory, perforations: MutableMapping[str, PerforationRange] | None
) -> Completion | None:
    if not perforations:
        return None

    completion_perforations = []
    for p_name, p_range in perforations.items():
        perforation_points = trajectory.get_points_in_md_range(
            p_range.start_md, p_range.end_md
        )
        if perforation_points:
            completion_perforations.append(
                Perforation(p_name, p_range, perforation_points)
            )

    if not completion_perforations:
        return None

    return Completion(tuple(completion_perforations))
