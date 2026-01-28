from __future__ import annotations

from typing import MutableMapping

from logger import get_logger
from services.well_management_service.core.models import (
    PerforationRange,
    SWellModel,
    TrajectoryPoint,
)
from services.well_management_service.core.sections import (
    CurvedWellSection,
    LinearWellSection,
)
from services.well_management_service.core.well_templates.well_template_interface import (
    WellTemplateInterface,
)


class SWellTemplate(WellTemplateInterface):
    def __init__(
        self,
        name: str,
        md_linear1: float,
        md_curved1: float,
        dls1: float,
        md_linear2: float,
        md_curved2: float,
        dls2: float,
        md_linear3: float,
        wellhead: TrajectoryPoint,
        azimuth: float,
        md_step: float,
        perforations: MutableMapping[str, PerforationRange] | None = None,
    ):
        super().__init__(name, md_step, wellhead, azimuth, perforations)
        self.__logger = get_logger(__name__)
        self.__logger.debug(
            f"Creating SWellTemplate model with md_linear1: {md_linear1} m, md_curved1: {md_curved1} m, dls1: {dls1} deg/30 m , \n md_linear2: {md_linear2} m, md_curved2: {md_curved2} m, dls2: {dls2} deg/30 m, md_linear3: {md_linear3} m"
        )

        self._well_builder.sections(
            [
                LinearWellSection(md=md_linear1),
                CurvedWellSection(md=md_curved1, dls=dls1),
                LinearWellSection(md=md_linear2),
                CurvedWellSection(md=md_curved2, dls=dls2),
                LinearWellSection(md=md_linear3),
            ]
        )

    @classmethod
    def from_model(cls, model: SWellModel) -> SWellTemplate:
        return cls(
            name=model.name,
            md_linear1=model.md_linear1,
            md_curved1=model.md_curved1,
            dls1=model.dls1,
            md_linear2=model.md_linear2,
            md_curved2=model.md_curved2,
            dls2=model.dls2,
            md_linear3=model.md_linear3,
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
