from abc import ABC
from typing import MutableMapping

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
        perforations: MutableMapping[str, PerforationRange] | None = None,
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
