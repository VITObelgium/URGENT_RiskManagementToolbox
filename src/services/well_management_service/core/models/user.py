from __future__ import annotations

import itertools
from typing import Literal, Union

from pydantic import BaseModel, Field, FiniteFloat, model_validator
from typing_extensions import Annotated

from services.well_management_service.core import models

type PerforationName = str


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
    perforations: dict[PerforationName, PerforationRangeModel] | None = Field(
        default=None
    )

    @model_validator(mode="after")
    def sort_perforations(self) -> IWellModel:
        if self.perforations:
            self.perforations = _sort_perforations(self.perforations)
        return self

    @model_validator(mode="after")
    def validate_perforations_not_overlap(self) -> IWellModel:
        if not _are_perforations_non_overlapping(self.perforations):
            raise ValueError("Perforations can't overlap")
        return self

    @model_validator(mode="after")
    def adjust_perforations_to_well_md(self) -> IWellModel:
        total_md = self.md
        self.perforations = _adjust_perforations(total_md, self.perforations)
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
    perforations: dict[PerforationName, PerforationRangeModel] | None = Field(
        default=None
    )

    @model_validator(mode="after")
    def sort_perforations(self) -> JWellModel:
        if self.perforations:
            self.perforations = _sort_perforations(self.perforations)
        return self

    @model_validator(mode="after")
    def validate_perforations_not_overlap(self) -> JWellModel:
        if not _are_perforations_non_overlapping(self.perforations):
            raise ValueError("Perforations can't overlap")
        return self

    @model_validator(mode="after")
    def adjust_perforations_to_well_md(self) -> JWellModel:
        total_md = self.md_linear1 + self.md_curved + self.md_linear2
        self.perforations = _adjust_perforations(total_md, self.perforations)
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
    perforations: dict[PerforationName, PerforationRangeModel] | None = Field(
        default=None
    )

    @model_validator(mode="after")
    def sort_perforations(self) -> SWellModel:
        if self.perforations:
            self.perforations = _sort_perforations(self.perforations)
        return self

    @model_validator(mode="after")
    def check_perforations_not_overlap(self) -> SWellModel:
        if not _are_perforations_non_overlapping(self.perforations):
            raise ValueError("Perforations can't overlap")
        return self

    @model_validator(mode="after")
    def adjust_perforations_to_well_md(self) -> SWellModel:
        total_md = (
                self.md_linear1
                + self.md_curved1
                + self.md_linear2
                + self.md_curved2
                + self.md_linear3
        )
        self.perforations = _adjust_perforations(total_md, self.perforations)
        return self


class HWellModel(BaseModel, extra="forbid", str_strip_whitespace=True):
    well_type: Literal["HWell"] = Field(default="HWell")
    name: str
    TVD: float = Field(gt=0.0)
    md_lateral: float = Field(gt=0.0)
    wellhead: PositionModel
    azimuth: float = Field(ge=0.0, lt=360.0)
    md_step: float = Field(ge=0.1)
    perforations: dict[PerforationName, PerforationRangeModel] | None = Field(
        default=None
    )

    @model_validator(mode="after")
    def sort_perforations(self) -> HWellModel:
        if self.perforations:
            self.perforations = _sort_perforations(self.perforations)
        return self

    #
    @model_validator(mode="after")
    def check_perforations_not_overlap(self) -> HWellModel:
        if not _are_perforations_non_overlapping(self.perforations):
            raise ValueError("Perforations can't overlap")
        return self

    @model_validator(mode="after")
    def adjust_perforations_to_well_md(self) -> HWellModel:
        from services.well_management_service.core.well_templates import HWellTemplate

        md_linear1, md_curved, md_linear2 = HWellTemplate.get_sections_mds(
            depth=self.TVD, width=self.md_lateral
        )
        total_md = md_linear1 + md_curved + md_linear2
        self.perforations = _adjust_perforations(total_md, self.perforations)
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


def _adjust_perforations(
        total_md: float, perforations: dict[PerforationName, PerforationRangeModel] | None
) -> dict[PerforationName, PerforationRangeModel] | None:
    """
    Adjusts perforations to ensure they are within the well's total MD.
    If a perforation's start is beyond the MD, it's removed.
    If a perforation's end is beyond the MD, it's truncated to the MD.
    """
    if not perforations:
        return None

    adjusted_perforations: dict[str, PerforationRangeModel] = {}

    for name, perf in perforations.items():
        # If the perforation starts after the well ends, discard it.
        if perf.start_md >= total_md:
            continue

        new_end_md = min(perf.end_md, total_md)

        if perf.start_md < new_end_md:
            adjusted_perforations[name] = PerforationRangeModel(
                start_md=perf.start_md,
                end_md=new_end_md,
            )

    return adjusted_perforations if adjusted_perforations else None


def _are_perforations_non_overlapping(
        perforations: dict[PerforationName, PerforationRangeModel] | None,
) -> bool:
    if not perforations or len(perforations) < 2:
        return True

    ordered = sorted(perforations.values(), key=lambda p: p.start_md)

    return all(
        current_p.end_md <= next_p.start_md
        for current_p, next_p in itertools.pairwise(ordered)
    )


def _sort_perforations(
        perforations: dict[PerforationName, PerforationRangeModel],
) -> dict[PerforationName, PerforationRangeModel]:
    return dict(sorted(perforations.items(), key=lambda item: item[1].start_md))
