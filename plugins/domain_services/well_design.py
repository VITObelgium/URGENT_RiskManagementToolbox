"""Built-in well_design domain service plugin — self-contained well geometry kernel.

Translates well item states into the built-in geometry templates and returns a
simulation-ready payload column shaped ``{"wells": [...]}`` (a validated
``WellDesignServiceResponse`` dump).

User-written domain plugins must NOT import private helpers from this file.
Build your own geometry, or construct a minimal WellDesignServiceResponse
from the public types re-exported by DomainServiceInterface.

Geometry model: wells are built in the vertical x-z plane (z positive down)
and then rotated by azimuth and translated to the wellhead. A well is a list
of ``(md, dls)`` sections; ``dls == 0`` is a straight section, otherwise the
section is a circular arc whose inclination (signed angle from +z, in the
x-z plane) changes at ``dls`` degrees per 30 m. Each section continues
tangentially from the previous one.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Annotated, Any, Literal, Self, assert_never

from pydantic import BaseModel, Field, FiniteFloat, TypeAdapter, model_validator

from logger import get_logger
from services.problem_dispatcher_service.core.service.interface import (
    DomainServiceInterface,
)
from urgent_plugins import DomainServicePlugin

logger = get_logger(__name__)

MD_ATOL: float = 1e-4
DLS_LENGTH: float = 30.0  # DLS is expressed in degrees per DLS_LENGTH metres

# H-shaped (horizontal) well fixed geometry
H_WELL_DLS: float = 4.0  # deg / 30 m
H_WELL_ANGLE: float = 90.0  # deg
_H_CURVE_MD: float = H_WELL_ANGLE * DLS_LENGTH / H_WELL_DLS
_H_CURVE_RADIUS: float = 2.0 * _H_CURVE_MD / math.pi

type PerforationName = str
type Section = tuple[float, float]  # (measured length [m], dls [deg/30 m])


def _h_well_section_mds(tvd: float, lateral: float) -> tuple[float, float, float]:
    """Split an H-well into (vertical, curved, horizontal) section lengths."""
    if (vertical_md := tvd - _H_CURVE_RADIUS) < 0:
        raise ValueError(
            "Horizontal well true total depth is less than curved well "
            "section radius. Increase depth."
        )
    if (horizontal_md := lateral - _H_CURVE_RADIUS) < 0:
        raise ValueError(
            "Horizontal well width is less than curved well section radius. "
            "Increase width."
        )
    return vertical_md, _H_CURVE_MD, horizontal_md


def _duplicate_names(names: list[str]) -> list[str]:
    counts = Counter(names)
    return [name for name, count in counts.items() if count > 1]


# Request models (the item-state schema)


class PerforationRangeModel(BaseModel, extra="forbid"):
    start_md: float = Field(ge=0.0)
    end_md: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end_md <= self.start_md:
            raise ValueError(
                f"Invalid range: start ({self.start_md}) must be less than "
                f"end ({self.end_md})"
            )
        return self


class PositionModel(BaseModel, extra="forbid"):
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat


def _adjust_perforations(
    total_md: float,
    perforations: dict[PerforationName, PerforationRangeModel] | None,
) -> dict[PerforationName, PerforationRangeModel] | None:
    """Clip perforations to the well length; drop the ones fully outside."""
    if not perforations:
        return None
    adjusted = {
        name: PerforationRangeModel(start_md=p.start_md, end_md=min(p.end_md, total_md))
        for name, p in perforations.items()
        if p.start_md < total_md
    }
    return adjusted or None


class _BaseWellModel(BaseModel, extra="forbid", str_strip_whitespace=True):
    name: str
    wellhead: PositionModel
    md_step: float = Field(ge=0.1, default=0.5)
    perforations: dict[PerforationName, PerforationRangeModel] | None = Field(
        default=None
    )

    def _total_md(self) -> float:
        raise NotImplementedError

    @model_validator(mode="after")
    def normalize_perforations(self) -> Self:
        self.perforations = _adjust_perforations(self._total_md(), self.perforations)
        if self.perforations:
            self.perforations = dict(
                sorted(self.perforations.items(), key=lambda item: item[1].start_md)
            )
            ordered = list(self.perforations.values())
            if not all(
                cur.end_md <= nxt.start_md for cur, nxt in itertools.pairwise(ordered)
            ):
                raise ValueError("Perforations can't overlap")
        return self


class IWellModel(_BaseWellModel):
    well_type: Literal["IWell"] = Field(default="IWell")
    md: float = Field(gt=0.0)

    def _total_md(self) -> float:
        return self.md


class JWellModel(_BaseWellModel):
    well_type: Literal["JWell"] = Field(default="JWell")
    md_linear1: float = Field(gt=0.0)
    md_curved: float = Field(gt=0.0)
    dls: float = Field(gt=-45.00, le=45.00)
    md_linear2: float = Field(gt=0.0)
    azimuth: float = Field(ge=0.0, lt=360.0)

    def _total_md(self) -> float:
        return self.md_linear1 + self.md_curved + self.md_linear2


class SWellModel(_BaseWellModel):
    well_type: Literal["SWell"] = Field(default="SWell")
    md_linear1: float = Field(gt=0.0)
    md_curved1: float = Field(gt=0.0)
    dls1: float = Field(gt=-45.00, le=45.00)
    md_linear2: float = Field(gt=0.0)
    md_curved2: float = Field(gt=0.0)
    dls2: float = Field(gt=-45.00, le=45.00)
    md_linear3: float = Field(gt=0.0)
    azimuth: float = Field(ge=0.0, lt=360.0)

    def _total_md(self) -> float:
        return (
            self.md_linear1
            + self.md_curved1
            + self.md_linear2
            + self.md_curved2
            + self.md_linear3
        )


class HWellModel(_BaseWellModel):
    well_type: Literal["HWell"] = Field(default="HWell")
    TVD: float = Field(gt=0.0)
    md_lateral: float = Field(gt=0.0)
    azimuth: float = Field(ge=0.0, lt=360.0)

    def _total_md(self) -> float:
        return sum(_h_well_section_mds(self.TVD, self.md_lateral))


WellModel = Annotated[
    IWellModel | JWellModel | SWellModel | HWellModel,
    Field(discriminator="well_type"),
]


class WellDesignServiceRequest(BaseModel, extra="forbid"):
    models: list[WellModel]

    @model_validator(mode="after")
    def validate_unique_names(self) -> Self:
        if duplicates := _duplicate_names([m.name for m in self.models]):
            raise ValueError(f"Well names must be unique. Duplicate:{duplicates}")
        return self


# Simulation payload types (plugin-private — define what the simulator connector consumes)


class SimulationWellPerforationModel(BaseModel, extra="forbid"):
    name: str
    range: tuple[float, float]
    points: tuple[tuple[float, float, float], ...]

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        start_md, end_md = self.range
        if end_md <= start_md:
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


class WellDesignServiceResponse(BaseModel, extra="forbid"):
    wells: list[SimulationWellModel]

    @model_validator(mode="after")
    def validate_unique_names(self) -> Self:
        if duplicates := _duplicate_names([w.name for w in self.wells]):
            raise ValueError(f"Well names must be unique. Duplicate:{duplicates}")
        return self


# Geometry kernel


def _sections_for(model: Any) -> tuple[list[Section], float]:
    """Map a well model onto its (sections, azimuth) description."""
    match model:
        case IWellModel():
            return [(model.md, 0.0)], 0.0
        case JWellModel():
            return [
                (model.md_linear1, 0.0),
                (model.md_curved, model.dls),
                (model.md_linear2, 0.0),
            ], model.azimuth
        case SWellModel():
            return [
                (model.md_linear1, 0.0),
                (model.md_curved1, model.dls1),
                (model.md_linear2, 0.0),
                (model.md_curved2, model.dls2),
                (model.md_linear3, 0.0),
            ], model.azimuth
        case HWellModel():
            vertical, curved, horizontal = _h_well_section_mds(
                model.TVD, model.md_lateral
            )
            return [
                (vertical, 0.0),
                (curved, H_WELL_DLS),
                (horizontal, 0.0),
            ], model.azimuth
        case _ as unreachable:
            assert_never(unreachable)


def _sample_mds(
    sections: list[Section], md_step: float, extra_mds: list[float]
) -> list[float]:
    """MDs at which the trajectory is sampled.

    A regular md_step grid restarting at every section boundary (so each
    boundary is always an exact sample point), merged with extra MDs
    (perforation boundaries), deduplicated within MD_ATOL.
    """
    total_md = sum(md for md, _ in sections)
    raw = [0.0]
    section_start = 0.0
    for section_md, _ in sections:
        num_steps = int(section_md // md_step)
        raw.extend(section_start + i * md_step for i in range(1, num_steps + 1))
        section_start += section_md
        raw.append(section_start)
    raw.extend(md for md in extra_mds if 0.0 <= md <= total_md)
    raw.sort()

    mds = [raw[0]]
    for md in raw[1:]:
        if md - mds[-1] > MD_ATOL:
            mds.append(md)
    return mds


def _plane_positions(
    sections: list[Section], mds: list[float]
) -> list[tuple[float, float, float]]:
    """(x, z, md) for every sampled md, in the unrotated vertical plane.

    Straight section: x(s) = x0 + s*sin(incl), z(s) = z0 + s*cos(incl).
    Arc with curvature c = radians(dls)/30 [rad/m] (inclination incl0 + c*s):
        x(s) = x0 + (cos(incl0) - cos(incl0 + c*s)) / c
        z(s) = z0 + (sin(incl0 + c*s) - sin(incl0)) / c
    """
    points: list[tuple[float, float, float]] = []
    x, z, incl = 0.0, 0.0, 0.0  # incl in radians, measured from +z (down)
    section_start = 0.0
    i = 0
    for section_md, dls in sections:
        c = math.radians(dls) / DLS_LENGTH

        def at(s: float) -> tuple[float, float]:
            if c == 0.0:
                return x + s * math.sin(incl), z + s * math.cos(incl)
            return (
                x + (math.cos(incl) - math.cos(incl + c * s)) / c,
                z + (math.sin(incl + c * s) - math.sin(incl)) / c,
            )

        section_end = section_start + section_md
        while i < len(mds) and mds[i] <= section_end + MD_ATOL:
            px, pz = at(mds[i] - section_start)
            points.append((px, pz, mds[i]))
            i += 1

        x, z = at(section_md)
        incl += c * section_md
        section_start = section_end

    points.extend((x, z, md) for md in mds[i:])  # tolerance leftovers, if any
    return points


def _build_well(model: Any) -> SimulationWellModel:
    sections, azimuth = _sections_for(model)
    perforations = model.perforations or {}

    boundary_mds = [md for p in perforations.values() for md in (p.start_md, p.end_md)]
    mds = _sample_mds(sections, model.md_step, boundary_mds)
    plane = _plane_positions(sections, mds)

    azimuth_rad = math.radians(azimuth)
    cos_az, sin_az = math.cos(azimuth_rad), math.sin(azimuth_rad)
    wellhead = model.wellhead
    points = [
        (wellhead.x + x * cos_az, wellhead.y + x * sin_az, wellhead.z + z, md)
        for x, z, md in plane
    ]

    perforation_models = tuple(
        SimulationWellPerforationModel(
            name=name, range=(p.start_md, p.end_md), points=perf_points
        )
        for name, p in perforations.items()
        if (
            perf_points := tuple(
                (x, y, z)
                for x, y, z, md in points
                if p.start_md - MD_ATOL <= md <= p.end_md + MD_ATOL
            )
        )
    )
    return SimulationWellModel(
        name=model.name,
        trajectory=tuple((x, y, z) for x, y, z, _ in points),
        completion=(
            SimulationWellCompletionModel(perforations=perforation_models)
            if perforation_models
            else None
        ),
    )


# Domain service


class WellDesignDomainService(DomainServiceInterface):
    """Reference domain service: dispatches each well model to its geometry template."""

    ServiceName = "well_design"

    @classmethod
    def get_item_state_adapter(cls) -> TypeAdapter[Any]:
        return TypeAdapter(WellModel)

    def build_payload(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        request = WellDesignServiceRequest(models=items)
        wells = []
        for model in request.models:
            logger.debug("Building well %r (type=%s)", model.name, model.well_type)
            wells.append(_build_well(model))
        # Validate before returning — keeps unique-name and range checks on the outbound path.
        return WellDesignServiceResponse(wells=wells).model_dump()


# Plugin descriptor

plugin = DomainServicePlugin(
    name=WellDesignDomainService.ServiceName,
    implementation=WellDesignDomainService,
)
