from abc import ABC
from typing import Sequence

from services.well_management_service.core.models import (
    PerforationRange,
    TrajectoryPoint,
    Well,
)
from services.well_management_service.core.service import WellBuilder


class WellTemplateInterface(ABC):
    def __init__(
        self,
        well_name: str,
        md_step: float,
        wellhead: TrajectoryPoint,
        azimuth: float,
        perforations: Sequence[PerforationRange] | None,
    ):
        self._well_builder = (
            WellBuilder()
            .well_name(well_name)
            .discretize(md_step)
            .translate(wellhead)
            .rotate(azimuth)
            .perforations(perforations)
        )

    def build(self) -> Well:
        return self._well_builder.build()
