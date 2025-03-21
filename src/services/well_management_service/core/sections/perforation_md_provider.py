from typing import Sequence

import numpy as np

from services.well_management_service.core.models import PerforationRange


class PerforationMdProvider:
    def __init__(
        self,
        perforations: Sequence[PerforationRange] | None,
        section_start_md: float,
        section_end_md: float,
    ) -> None:
        self.__perforation_md_boundary_point_for_range: list[float] = (
            self._get_perforations_boundary_points_inside_md_range(
                perforations, section_start_md, section_end_md
            )
        )
        self.__perforation_md_iter = iter(
            self.__perforation_md_boundary_point_for_range
        )

    @staticmethod
    def _get_perforations_boundary_points_inside_md_range(
        perforations: Sequence[PerforationRange] | None,
        section_start_md: float,
        section_end_md: float,
    ) -> list[float]:
        if not perforations:
            return []

        return [
            p_md
            for perforation in perforations
            for p_md in perforation
            if section_start_md <= p_md <= section_end_md
        ]

    def get_next_md(self) -> float:
        return next(self.__perforation_md_iter, np.inf)
