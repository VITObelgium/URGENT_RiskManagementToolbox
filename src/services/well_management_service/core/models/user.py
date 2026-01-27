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
        if not _has_valid_range(self.start_md, self.end_md):
            raise ValueError(
                f"Invalid range: start ({self.start_md}) must be less than end ({self.end_md})"
            )
        return self


class PositionModel(BaseModel, extra="forbid"):
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat


class _BaseWellModel(BaseModel, extra="forbid"):
    name: str
    wellhead: PositionModel
    md_step: float = Field(ge=0.1, default=0.5)
    perforations: dict[PerforationName, PerforationRangeModel] | None = Field(
        default=None
    )

    @model_validator(mode="after")
    def sort_perforations(self) -> _BaseWellModel:
        if self.perforations:
            self.perforations = _sort_perforations(self.perforations)
        return self

    @model_validator(mode="after")
    def validate_perforations_not_overlap(self) -> _BaseWellModel:
        if not _are_perforations_non_overlapping(self.perforations):
            raise ValueError("Perforations can't overlap")
        return self


class IWellModel(_BaseWellModel, extra="forbid", str_strip_whitespace=True):
    well_type: Literal["IWell"] = Field(default="IWell")
    md: float = Field(gt=0.0)

    @model_validator(mode="after")
    def adjust_perforations_to_well_md(self) -> IWellModel:
        total_md = self.md
        self.perforations = _adjust_perforations(total_md, self.perforations)
        return self


class JWellModel(_BaseWellModel, extra="forbid", str_strip_whitespace=True):
    well_type: Literal["JWell"] = Field(default="JWell")
    md_linear1: float = Field(gt=0.0)
    md_curved: float = Field(gt=0.0)
    dls: float = Field(gt=-45.00, le=45.00)
    md_linear2: float = Field(gt=0.0)
    azimuth: float = Field(ge=0.0, lt=360.0)

    @model_validator(mode="after")
    def adjust_perforations_to_well_md(self) -> JWellModel:
        total_md = self.md_linear1 + self.md_curved + self.md_linear2
        self.perforations = _adjust_perforations(total_md, self.perforations)
        return self


class SWellModel(_BaseWellModel, extra="forbid", str_strip_whitespace=True):
    well_type: Literal["SWell"] = Field(default="SWell")
    md_linear1: float = Field(gt=0.0)
    md_curved1: float = Field(gt=0.0)
    dls1: float = Field(gt=-45.00, le=45.00)
    md_linear2: float = Field(gt=0.0)
    md_curved2: float = Field(gt=0.0)
    dls2: float = Field(gt=-45.00, le=45.00)
    md_linear3: float = Field(gt=0.0)
    azimuth: float = Field(ge=0.0, lt=360.0)

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


class HWellModel(_BaseWellModel, extra="forbid", str_strip_whitespace=True):
    well_type: Literal["HWell"] = Field(default="HWell")
    TVD: float = Field(gt=0.0)
    md_lateral: float = Field(gt=0.0)
    azimuth: float = Field(ge=0.0, lt=360.0)

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
    def validate_wells_name(self) -> WellManagementServiceRequest:
        wells_name = [n.name for n in self.models]
        if duplicates := _duplicate_well_names(wells_name):
            raise ValueError(f"Well names must be unique. Duplicate:{duplicates}")
        return self


class WellManagementServiceResponse(BaseModel, extra="forbid"):
    wells: list[SimulationWellModel]

    @model_validator(mode="after")
    def validate_wells_name(self) -> WellManagementServiceResponse:
        wells_name = [n.name for n in self.wells]
        if duplicates := _duplicate_well_names(wells_name):
            raise ValueError(f"Well names must be unique. Duplicate:{duplicates}")
        return self


class SimulationWellPerforationModel(BaseModel, extra="forbid"):
    name: str
    range: tuple[float, float]
    points: tuple[tuple[float, float, float], ...]

    @model_validator(mode="after")
    def validate_perforation_start_and_end(self) -> SimulationWellPerforationModel:
        start_md, end_md = self.range
        if not _has_valid_range(start_md, end_md):
            raise ValueError(
                f"Invalid range: start ({start_md}) must be less than end ({end_md})"
            )
        return self


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
            trajectory=tuple(((t.x, t.y, t.z) for t in well.trajectory)),
            completion=(
                SimulationWellCompletionModel(
                    perforations=tuple(
                        (
                            SimulationWellPerforationModel(
                                name=p.name,
                                points=tuple([(p.x, p.y, p.z) for p in p.points]),
                                range=(p.range.start_md, p.range.end_md),
                            )
                            for p in well.completion.perforations
                        )
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
    """
    Checks if perforations are non-overlapping.

    Args:
        perforations: A dictionary of perforation names to their ranges.

    Returns:
        True if perforations are non-overlapping, False otherwise.
    """
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
    """
    Sorts perforations by their start MD.

    Args:
        perforations: A dictionary of perforation names to their ranges.

    Returns:
        A dictionary of perforation names to their ranges, sorted by start MD.
    """
    return dict(sorted(perforations.items(), key=lambda item: item[1].start_md))


def _duplicate_well_names(wells_name: list[str]) -> list[str]:
    """
    Returns duplicate well names (each duplicate appears once), preserving first-duplicate order.
    Example: [A, B, A, A, B] -> ["A", "B"]
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    result: list[str] = []

    for name in wells_name:
        if name in seen and name not in duplicates:
            duplicates.add(name)
            result.append(name)
        else:
            seen.add(name)

    return result


def _has_valid_range(start_md: float, end_md: float) -> bool:
    return end_md > start_md
