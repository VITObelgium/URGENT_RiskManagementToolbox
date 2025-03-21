from __future__ import annotations

from typing import Sequence

import numpy as np

from services.well_management_service.core.models import (
    HWellModel,
    PerforationRange,
    TrajectoryPoint,
)
from services.well_management_service.core.sections import (
    CurvedWellSection,
    LinearWellSection,
)
from services.well_management_service.core.well_templates.well_template_interface import (
    WellTemplateInterface,
)


class HWellTemplate(WellTemplateInterface):
    """
    J-shaped well with 90 degrees angle and fixed DLS.
    """

    FIXED_ANGLE = 90.0  # degrees
    FIXED_DLS = 4.0  # degrees per 30 meters

    _CURVATURE_MD = (FIXED_ANGLE * 30) / FIXED_DLS  # md of the CurvedWellSection
    _CURVATURE_RADIUS = 2 * _CURVATURE_MD / np.pi

    def __init__(
        self,
        name: str,
        TVD: float,
        md_lateral: float,
        wellhead: TrajectoryPoint,
        azimuth: float,
        md_step: float,
        perforations: Sequence[PerforationRange] | None = None,
    ):
        super().__init__(name, md_step, wellhead, azimuth, perforations)
        md_linear1, md_curved, md_linear2 = self.get_sections_mds(TVD, md_lateral)

        self._well_builder.sections(
            [
                LinearWellSection(md=md_linear1),
                CurvedWellSection(md=md_curved, dls=self.FIXED_DLS),
                LinearWellSection(md=md_linear2),
            ]
        )

    @classmethod
    def get_sections_mds(cls, depth: float, width: float) -> tuple[float, float, float]:
        if (vertical_LWS_md := depth - cls._CURVATURE_RADIUS) < 0:
            raise ValueError(
                "Horizontal well true total depth is less than curved well section radius. Increase depth."
            )

        elif (horizontal_LWS_md := width - cls._CURVATURE_RADIUS) < 0:
            raise ValueError(
                "Horizontal well width is less than curved well section radius. Increase width."
            )

        return vertical_LWS_md, cls._CURVATURE_MD, horizontal_LWS_md

    @classmethod
    def from_model(cls, model: HWellModel) -> HWellTemplate:
        return cls(
            name=model.name,
            TVD=model.TVD,
            md_lateral=model.md_lateral,
            wellhead=TrajectoryPoint(
                x=model.wellhead.x, y=model.wellhead.y, z=model.wellhead.z, md=0
            ),
            azimuth=model.azimuth,
            md_step=model.md_step,
            perforations=(
                [PerforationRange(p.start_md, p.end_md) for p in model.perforations]
                if model.perforations
                else None
            ),
        )
