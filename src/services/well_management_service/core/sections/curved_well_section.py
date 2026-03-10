from typing import MutableMapping

import numpy as np

from logger import get_logger
from services.well_management_service.core.models import (
    PerforationRange,
    Point,
    Trajectory,
    TrajectoryPoint,
)
from services.well_management_service.core.sections.perforation_md_provider import (
    PerforationMdProvider,
)
from services.well_management_service.core.sections.section_interface import (
    SectionInterface,
)
from services.well_management_service.core.utilities.constants import DLS_RATE


class CurvedWellSection(SectionInterface):
    def __init__(self, md: float, dls: float, user_inclination: float | None = None):
        super().__init__(md, user_inclination)
        self._dls: float = dls  # deg/30 m
        self.__logger = get_logger(__name__)
        if user_inclination is None:
            self.__logger.debug(
                "Curved well section inclination is not provided. Calculating inclination from the trajectory"
            )

    def append_to_trajectory(
        self,
        trajectory: Trajectory,
        md_step: float,
        perforations: MutableMapping[str, PerforationRange] | None = None,
    ) -> Trajectory:
        last_point = trajectory.get_last_trajectory_point()
        x, y, z, point_md = last_point
        section_trajectory: list[TrajectoryPoint] = list(trajectory)

        num_steps = int(self._md // md_step)
        remaining_md = self._md % md_step
        inclination_deg = (
            self._user_inclination
            if self._user_inclination is not None
            else trajectory.get_xz_inclination()
        )

        self.__logger.debug(
            f"Appending curved well section to the trajectory with step: {self._md} [m], inclination: {inclination_deg} [deg] and DLS: {self._dls} [deg/30 m]"
        )

        rotation_center = _rotation_center(Point(x, y, z), self._dls, inclination_deg)
        rotation_radius = _circle_radius(self._dls)

        rx = x - rotation_center.x
        rz = z - rotation_center.z

        theta_start_rad = _calculate_theta_start_rad(rx, rz, inclination_deg)

        sign_z = 1 if self._dls <= 0 else -1
        sign_x = 1 if inclination_deg >= 0 else -1

        section_start_md = point_md
        section_end_md = section_start_md + self._md
        perforation_md_provider = PerforationMdProvider(
            perforations, section_start_md, section_end_md
        )

        perforation_md = perforation_md_provider.get_next_md()
        step_arc_length = 0.0
        for _ in range(num_steps):
            step_arc_length += md_step
            theta_rad = theta_start_rad + _angle_increase_rad(
                step_arc_length, rotation_radius
            )
            x = rotation_center.x + rotation_radius * np.cos(theta_rad) * sign_x
            z = rotation_center.z + rotation_radius * np.sin(theta_rad) * sign_z
            point_md += md_step

            last_added_point = section_trajectory[-1]

            if last_added_point.is_close_to_md(perforation_md):
                perforation_md = perforation_md_provider.get_next_md()

            while point_md > perforation_md and not last_added_point.is_close_to_md(
                perforation_md
            ):
                theta_perf = theta_start_rad + _angle_increase_rad(
                    perforation_md - last_point.z, rotation_radius
                )
                x_perf = (
                    rotation_center.x + rotation_radius * np.cos(theta_perf) * sign_x
                )
                y_perf = y
                z_perf = (
                    rotation_center.z + rotation_radius * np.sin(theta_perf) * sign_z
                )
                perf_point = TrajectoryPoint(x_perf, y_perf, z_perf, perforation_md)
                self.__logger.debug(
                    f"Appending perforation boundary point: {perf_point} to the trajectory"
                )
                section_trajectory.append(perf_point)
                perforation_md = perforation_md_provider.get_next_md()

            self.__logger.debug(
                f"Appending point: {TrajectoryPoint(x, y, z, point_md)} to the trajectory"
            )

            section_trajectory.append(TrajectoryPoint(x, y, z, point_md))

        if remaining_md > 0.0:
            theta_rad = theta_start_rad + _angle_increase_rad(self._md, rotation_radius)
            x = rotation_center.x + rotation_radius * np.cos(theta_rad) * sign_x
            z = rotation_center.z + rotation_radius * np.sin(theta_rad) * sign_z
            point_md += remaining_md

            last_added_point = section_trajectory[-1]

            if last_added_point.is_close_to_md(perforation_md):
                perforation_md = perforation_md_provider.get_next_md()

            while point_md > perforation_md and not last_added_point.is_close_to_md(
                perforation_md
            ):
                theta_perf = theta_start_rad + _angle_increase_rad(
                    perforation_md - last_point.z, rotation_radius
                )
                x_perf = (
                    rotation_center.x + rotation_radius * np.cos(theta_perf) * sign_x
                )
                y_perf = y
                z_perf = (
                    rotation_center.z + rotation_radius * np.sin(theta_perf) * sign_z
                )
                perf_point = TrajectoryPoint(x_perf, y_perf, z_perf, perforation_md)
                self.__logger.debug(
                    f"Appending perforation boundary point: {perf_point} to the trajectory"
                )
                section_trajectory.append(perf_point)
                perforation_md = perforation_md_provider.get_next_md()

            self.__logger.debug(
                f"Appending point: {TrajectoryPoint(x, y, z, point_md)} to the trajectory"
            )

            section_trajectory.append(TrajectoryPoint(x, y, z, point_md))

        return Trajectory(tuple(section_trajectory))


def _rotation_center(last_point: Point, dls: float, inclination0_deg: float) -> Point:
    r = _circle_radius(dls)
    x, y, z = last_point.x, last_point.y, last_point.z

    same_sign = (dls >= 0 and inclination0_deg >= 0) or (
        dls < 0 and inclination0_deg < 0
    )
    sign_x = 1 if same_sign else -1
    sign_z = -1 if dls >= 0 else 1

    x0 = x + sign_x * r * np.cos(np.deg2rad(np.abs(inclination0_deg)))
    z0 = z + sign_z * r * np.sin(np.deg2rad(np.abs(inclination0_deg)))

    return Point(x0, y, z0)


def _circle_radius(dls: float) -> float:
    return float(np.abs(DLS_RATE / np.deg2rad(dls)))


def _angle_increase_rad(arc_length: float, circle_radius: float) -> float:
    return arc_length / circle_radius


def _calculate_theta_start_rad(rx: float, rz: float, inclination: float) -> float:
    if rx == 0 and rz == 0:
        raise ValueError(
            "rx and rz cannot be both 0 - rotation center cannot be in the well bore"
        )
    if rx == 0:
        if rz > 0:
            return float(-np.pi / 2 if inclination >= 0 else 3 * np.pi / 2)
        else:
            return float(np.pi / 2)

    if rx > 0 and rz > 0:
        return float(
            -np.atan2(rz, rx) if inclination >= 0 else np.pi + np.atan2(rz, rx)
        )
    elif rx <= 0 and rz < 0:
        return float(np.atan2(rz, rx) if inclination >= 0 else np.pi - np.atan2(rz, rx))
    elif rx < 0 <= rz:
        return float(
            2 * np.pi - np.atan2(rz, rx)
            if inclination >= 0
            else np.pi + np.atan2(rz, rx)
        )
    elif rx > 0 >= rz:
        return float(np.atan2(rz, rx) if inclination >= 0 else np.pi - np.atan2(rz, rx))

    raise ValueError("Invalid combination of rx and rz")
