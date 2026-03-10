from __future__ import annotations

from typing import MutableMapping

from logger import get_logger
from services.well_management_service.core.models import (
    IWellModel,
    PerforationRange,
    TrajectoryPoint,
)
from services.well_management_service.core.sections import (
    LinearWellSection,
)
from services.well_management_service.core.well_templates.well_template_interface import (
    WellTemplateInterface,
)


class IWellTemplate(WellTemplateInterface):
    def __init__(
        self,
        name: str,
        md: float,
        wellhead: TrajectoryPoint,
        md_step: float,
        perforations: MutableMapping[str, PerforationRange] | None = None,
    ):
        zero_azimuth = 0  # no azimuth in vertical well
        super().__init__(name, md_step, wellhead, zero_azimuth, perforations)
        self.__logger = get_logger(__name__)
        self.__logger.debug(f"Creating IWellTemplate model with MD: {md} m")

        self._well_builder.sections([LinearWellSection(md=md)])

    @classmethod
    def from_model(cls, model: IWellModel) -> IWellTemplate:
        return cls(
            name=model.name,
            md=model.md,
            wellhead=TrajectoryPoint(
                x=model.wellhead.x, y=model.wellhead.y, z=model.wellhead.z, md=0
            ),
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
