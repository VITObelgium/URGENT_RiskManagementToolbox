from __future__ import annotations

import itertools
from typing import Literal, Sequence, Union

import numpy as np
from pydantic import BaseModel, Field, FiniteFloat, model_validator
from typing_extensions import Annotated

from services.well_management_service.core import models


class PerforationRangeModel(BaseModel, extra="forbid"):
    start_md: float = Field(ge=0.0)
    end_md: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_perforation_start_and_end(self) -> PerforationRangeModel:
        if self.start_md >= self.end_md:
            raise ValueError(
                f"Invalid range: start ({self.start_md}) must be less than end ({self.end_md})"
            )
        return self


class PositionModel(BaseModel, extra="forbid"):
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat


class IWellModel(BaseModel, extra="forbid", str_strip_whitespace=True):
    well_type: Literal["IWell"] = Field(default="IWell")
    name: str
    md: float = Field(gt=0.0)
    wellhead: PositionModel
    md_step: float = Field(ge=0.1)
    perforations: Sequence[PerforationRangeModel] | None = Field(default=None)

    @model_validator(mode="after")
    def sort_perforations(self) -> IWellModel:
        if self.perforations:
            self.perforations = sorted(self.perforations, key=lambda p: p.start_md)
        return self

    @model_validator(mode="after")
    def validate_perforations_not_overlap(self) -> IWellModel:
        if not _are_perforations_non_overlapping(self.perforations):
            raise ValueError("Perforations can't overlap")
        return self

    @model_validator(mode="after")
    def validate_perforation_within_well_md(self) -> IWellModel:
        if not _is_perforation_within_total_well_md(
            (self.md,),
            self.perforations,
        ):
            raise ValueError(
                "One or more perforation intervals extend beyond the total measured depth of the well."
            )
        return self


class JWellModel(BaseModel, extra="forbid", str_strip_whitespace=True):
    well_type: Literal["JWell"] = Field(default="JWell")
    name: str
    md_linear1: float = Field(gt=0.0)
    md_curved: float = Field(gt=0.0)
    dls: float = Field(gt=-45.00, le=45.00)
    md_linear2: float = Field(gt=0.0)
    wellhead: PositionModel
    azimuth: float = Field(ge=0.0, lt=360.0)
    md_step: float = Field(ge=0.1)
    perforations: Sequence[PerforationRangeModel] | None = Field(default=None)

    @model_validator(mode="after")
    def sort_perforations(self) -> JWellModel:
        if self.perforations:
            self.perforations = sorted(self.perforations, key=lambda p: p.start_md)
        return self

    @model_validator(mode="after")
    def validate_perforations_not_overlap(self) -> JWellModel:
        if not _are_perforations_non_overlapping(self.perforations):
            raise ValueError("Perforations can't overlap")
        return self

    @model_validator(mode="after")
    def validate_perforation_within_well_md(self) -> JWellModel:
        if not _is_perforation_within_total_well_md(
            (self.md_linear1, self.md_curved, self.md_linear2),
            self.perforations,
        ):
            raise ValueError(
                "One or more perforation intervals extend beyond the total measured depth of the well."
            )
        return self


class SWellModel(BaseModel, extra="forbid", str_strip_whitespace=True):
    well_type: Literal["SWell"] = Field(default="SWell")
    name: str
    md_linear1: float = Field(gt=0.0)
    md_curved1: float = Field(gt=0.0)
    dls1: float = Field(gt=-45.00, le=45.00)
    md_linear2: float = Field(gt=0.0)
    md_curved2: float = Field(gt=0.0)
    dls2: float = Field(gt=-45.00, le=45.00)
    md_linear3: float = Field(gt=0.0)
    wellhead: PositionModel
    azimuth: float = Field(ge=0.0, lt=360.0)
    md_step: float = Field(ge=0.1)
    perforations: Sequence[PerforationRangeModel] | None = Field(default=None)

    @model_validator(mode="after")
    def sort_perforations(self) -> SWellModel:
        if self.perforations:
            self.perforations = sorted(self.perforations, key=lambda p: p.start_md)
        return self

    @model_validator(mode="after")
    def check_perforations_not_overlap(self) -> SWellModel:
        if not _are_perforations_non_overlapping(self.perforations):
            raise ValueError("Perforations can't overlap")
        return self

    @model_validator(mode="after")
    def validate_perforation_within_well_md(self) -> SWellModel:
        if not _is_perforation_within_total_well_md(
            (
                self.md_linear1,
                self.md_curved1,
                self.md_linear2,
                self.md_curved2,
                self.md_linear3,
            ),
            self.perforations,
        ):
            raise ValueError(
                "One or more perforation intervals extend beyond the total measured depth of the well."
            )
        return self


class HWellModel(BaseModel, extra="forbid", str_strip_whitespace=True):
    well_type: Literal["HWell"] = Field(default="HWell")
    name: str
    TVD: float = Field(gt=0.0)
    md_lateral: float = Field(gt=0.0)
    wellhead: PositionModel
    azimuth: float = Field(ge=0.0, lt=360.0)
    md_step: float = Field(ge=0.1)
    perforations: Sequence[PerforationRangeModel] | None = Field(default=None)

    @model_validator(mode="after")
    def sort_perforations(self) -> HWellModel:
        if self.perforations:
            self.perforations = sorted(self.perforations, key=lambda p: p.start_md)
        return self

    @model_validator(mode="after")
    def check_perforations_not_overlap(self) -> HWellModel:
        if not _are_perforations_non_overlapping(self.perforations):
            raise ValueError("Perforations can't overlap")
        return self

    @model_validator(mode="after")
    def validate_perforation_within_well_md(self) -> HWellModel:
        from services.well_management_service.core.well_templates import HWellTemplate

        md_linear1, md_curved, md_linear2 = HWellTemplate.get_sections_mds(
            depth=self.TVD, width=self.md_lateral
        )
        if not _is_perforation_within_total_well_md(
            (
                md_linear1,
                md_curved,
                md_linear2,
            ),
            self.perforations,
        ):
            raise ValueError(
                "One or more perforation intervals extend beyond the total measured depth of the well."
            )
        return self


WellModel = Annotated[
    Union[IWellModel, JWellModel, SWellModel, HWellModel],
    Field(discriminator="well_type"),
]


class WellManagementServiceRequest(BaseModel, extra="forbid"):
    models: list[WellModel]

    @model_validator(mode="after")
    def ensure_unique_names(self) -> WellManagementServiceRequest:
        seen_names = set()
        for model in self.models:
            if model.name in seen_names:
                raise ValueError("Wells names must be unique.")
            seen_names.add(model.name)
        return self


class WellManagementServiceResponse(BaseModel, extra="forbid"):
    wells: list[SimulationWellModel]


class SimulationWellPerforationModel(BaseModel, extra="forbid"):
    range: tuple[float, float]
    points: tuple[tuple[float, float, float], ...]


class SimulationWellCompletionModel(BaseModel, extra="forbid"):
    perforations: tuple[SimulationWellPerforationModel, ...]


class SimulationWellModel(BaseModel, extra="forbid"):
    name: str
    trajectory: tuple[tuple[float, float, float], ...]
    completion: SimulationWellCompletionModel | None = Field(default=None)

    @classmethod
    def from_well(cls, well: models.Well) -> SimulationWellModel:
        return cls(
            name=well.name,
            trajectory=tuple([(t.x, t.y, t.z) for t in well.trajectory]),
            completion=(
                SimulationWellCompletionModel(
                    perforations=tuple(
                        [
                            SimulationWellPerforationModel(
                                points=tuple([(p.x, p.y, p.z) for p in p.points]),
                                range=(p.range.start_md, p.range.end_md),
                            )
                            for p in well.completion.perforations
                        ]
                    )
                )
                if well.completion
                else None
            ),
        )


def _is_perforation_within_total_well_md(
    well_mds: Sequence[float], perforations: Sequence[PerforationRangeModel] | None
) -> bool:
    if not perforations:
        return True

    total_measured_depth = np.sum(well_mds)
    return all(
        perf.start_md <= total_measured_depth and perf.end_md <= total_measured_depth
        for perf in perforations
    )


def _are_perforations_non_overlapping(
    perforations: Sequence[PerforationRangeModel] | None,
) -> bool:
    if not perforations or len(perforations) < 2:
        return True

    return all(
        current_p.end_md <= next_p.start_md
        for current_p, next_p in itertools.pairwise(perforations)
    )
