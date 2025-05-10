from typing import Sequence

import numpy as np

from logger import get_logger
from services.well_management_service.core.models import (
    PerforationRange,
    Trajectory,
    TrajectoryPoint,
)
from services.well_management_service.core.sections.perforation_md_provider import (
    PerforationMdProvider,
)
from services.well_management_service.core.sections.section_interface import (
    SectionInterface,
)


class LinearWellSection(SectionInterface):
    def __init__(self, md: float, user_inclination: float | None = None):
        super().__init__(md, user_inclination)
        self.__logger = get_logger(__name__)
        if user_inclination is None:
            self.__logger.debug(
                "Linear well section inclination is not provided. Calculating inclination from the trajectory"
            )

    def append_to_trajectory(
        self,
        trajectory: Trajectory,
        md_step: float,
        perforations: Sequence[PerforationRange] | None = None,
    ) -> Trajectory:
        last_point = trajectory.get_last_trajectory_point()
        x, y, z, point_md = last_point
        section_trajectory: list[TrajectoryPoint] = list(trajectory)

        num_steps = int(self._md // md_step)
        remaining_md = self._md % md_step
        inclination = (
            self._user_inclination
            if self._user_inclination is not None
            else trajectory.get_xz_inclination()
        )
        inclination_rad = np.deg2rad(inclination)
        cos_incl = np.cos(inclination_rad)
        sin_incl = np.sin(inclination_rad)

        self.__logger.debug(
            f"Appending linear well section to the trajectory with step: {md_step} [m] and inclination: {inclination} [deg]"
        )

        section_start_md = point_md
        section_end_md = section_start_md + self._md
        perforation_md_provider = PerforationMdProvider(
            perforations, section_start_md, section_end_md
        )

        perforation_md = perforation_md_provider.get_next_md()

        for _ in range(num_steps):
            z += md_step * cos_incl
            x += md_step * sin_incl
            point_md += md_step

            last_added_point = section_trajectory[-1]

            if last_added_point.is_close_to_md(perforation_md):
                perforation_md = perforation_md_provider.get_next_md()

            while point_md > perforation_md and not last_added_point.is_close_to_md(
                perforation_md
            ):
                x_perf = x - (point_md - perforation_md) * sin_incl
                y_perf = y
                z_perf = z - (point_md - perforation_md) * cos_incl
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
            z += remaining_md * cos_incl
            x += remaining_md * sin_incl
            point_md += remaining_md

            last_added_point = section_trajectory[-1]

            if last_added_point.is_close_to_md(perforation_md):
                perforation_md = perforation_md_provider.get_next_md()

            while point_md > perforation_md and not last_added_point.is_close_to_md(
                perforation_md
            ):
                x_perf = x - (point_md - perforation_md) * sin_incl
                y_perf = y
                z_perf = z - (point_md - perforation_md) * cos_incl
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
