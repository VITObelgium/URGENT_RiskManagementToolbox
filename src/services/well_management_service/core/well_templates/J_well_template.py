from __future__ import annotations

from typing import MutableMapping

from logger import get_logger
from services.well_management_service.core.models import (
    JWellModel,
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


class JWellTemplate(WellTemplateInterface):
    def __init__(
        self,
        name: str,
        md_linear1: float,
        md_curved: float,
        dls: float,
        md_linear2: float,
        wellhead: TrajectoryPoint,
        azimuth: float,
        md_step: float,
        perforations: MutableMapping[str, PerforationRange] | None = None,
    ):
        super().__init__(name, md_step, wellhead, azimuth, perforations)
        self.__logger = get_logger(__name__)
        self.__logger.debug(
            f"Creating JWellTemplate model with md_linear1: {md_linear1} m, md_curved: {md_curved} m,dls: {dls} deg/30 m , md_linear2: {md_linear2} m"
        )

        self._well_builder.sections(
            [
                LinearWellSection(md=md_linear1),
                CurvedWellSection(md=md_curved, dls=dls),
                LinearWellSection(md=md_linear2),
            ]
        )

    @classmethod
    def from_model(cls, model: JWellModel) -> JWellTemplate:
        return cls(
            name=model.name,
            md_linear1=model.md_linear1,
            md_curved=model.md_curved,
            dls=model.dls,
            md_linear2=model.md_linear2,
            wellhead=TrajectoryPoint(
                x=model.wellhead.x, y=model.wellhead.y, z=model.wellhead.z, md=0
            ),
            azimuth=model.azimuth,
            md_step=model.md_step,
            perforations=(
                {
                    n: PerforationRange(p.start_md, p.end_md)
                    for n, p in model.perforations.items()
                }
                if model.perforations
                else None
            ),
        )
