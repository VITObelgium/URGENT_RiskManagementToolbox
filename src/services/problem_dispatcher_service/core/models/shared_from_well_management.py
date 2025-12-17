from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field, FiniteFloat, model_validator


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
    md_step: float = Field(ge=0.1, default=0.5)
    perforations: list[PerforationRangeModel] | None = Field(
        default=None
    )  # simplification -> only one perforation

    @model_validator(mode="after")
    def sort_perforations(self) -> IWellModel:
        if self.perforations:
            self.perforations = sorted(self.perforations, key=lambda p: p.start_md)
        return self

    @model_validator(mode="after")
    def check_perforations_not_overlap(self) -> IWellModel:
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

    @model_validator(mode="after")
    def perforate_whole_md_if_no_perforation(self):
        if not self.perforations:
            self.perforations = [PerforationRangeModel(start_md=0, end_md=self.md)]
        return self

    @model_validator(mode="after")
    def accept_only_one_perforation(self):
        if self.perforations and len(self.perforations) > 1:
            raise ValueError("Only single perforation is supported")
        return self


WellModel = Annotated[
    IWellModel,
    Field(discriminator="well_type"),
]


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
