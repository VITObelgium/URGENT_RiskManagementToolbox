"""Built-in well_design domain service plugin — self-contained well geometry kernel.

Translates well item states into the built-in geometry templates and returns a
simulation-ready payload column shaped ``{"wells": [...]}`` (a validated
``WellDesignServiceResponse`` dump).

User-written domain plugins must NOT import private geometry types from this
file. Build your own geometry, or construct a minimal WellDesignServiceResponse
from the public types re-exported by DomainServiceInterface.
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterator, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Self, assert_never

import numpy as np
from pydantic import BaseModel, Field, FiniteFloat, TypeAdapter, model_validator

from logger import get_logger
from services.problem_dispatcher_service.core.service.interface import (
    DomainServiceInterface,
)
from urgent_plugins import DomainServicePlugin

logger = get_logger(__name__)


# Constants

DLS_RATE: float = 30  # deg/30m
POINT_ATOL: float = 1e-4


class EmptySectionConfigurationException(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


# Geometry primitives


@dataclass(frozen=True, slots=True)
class Point:
    """Positive Z — downwards."""

    x: float
    y: float
    z: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            raise TypeError(f"Expected type is Point, actual is {type(other)}")
        return (
            math.isclose(self.x, other.x, abs_tol=POINT_ATOL)
            and math.isclose(self.y, other.y, abs_tol=POINT_ATOL)
            and math.isclose(self.z, other.z, abs_tol=POINT_ATOL)
        )

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y, self.z))

    def __str__(self) -> str:
        return f"Point({self.x}, {self.y}, {self.z})"


@dataclass(frozen=True, slots=True)
class TrajectoryPoint(Point):
    """Positive Z — downwards."""

    md: float

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y, self.z, self.md))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrajectoryPoint):
            raise TypeError(
                f"Expected type is TrajectoryPoint, actual is {type(other)}"
            )
        return (
            math.isclose(self.x, other.x, abs_tol=POINT_ATOL)
            and math.isclose(self.y, other.y, abs_tol=POINT_ATOL)
            and math.isclose(self.z, other.z, abs_tol=POINT_ATOL)
            and math.isclose(self.md, other.md, abs_tol=POINT_ATOL)
        )

    def is_close_to_md(self, other_md: float) -> bool:
        return math.isclose(other_md, self.md, abs_tol=POINT_ATOL)


class Trajectory(tuple[TrajectoryPoint, ...]):
    """Ordered sequence of TrajectoryPoints defining a well path in space."""

    def __new__(cls, iterable: tuple[TrajectoryPoint, ...]) -> Trajectory:
        if len(iterable) == 0:
            raise ValueError(
                "Can't create empty trajectory. Wellhead must be provided."
            )
        return super().__new__(cls, iterable)

    def get_last_trajectory_point(self) -> TrajectoryPoint:
        return self[-1]

    def get_xz_inclination(self) -> float:
        if len(self) < 2:
            return 0.0
        last_point: TrajectoryPoint = self[-1]
        second_last_point: TrajectoryPoint = self[-2]
        dz = second_last_point.z - last_point.z
        dx = second_last_point.x - last_point.x
        d = max(np.sqrt(dx**2 + dz**2), 1.0e-6)
        inclination = np.rad2deg(np.arccos(-dz / d))
        if dx > 0:
            inclination = -inclination
        return float(inclination)

    def get_points_in_md_range(
        self, md_start: float, md_end: float
    ) -> tuple[Point, ...]:
        tol = 1e-6
        return tuple(
            Point(t.x, t.y, t.z)
            for t in self
            if (md_start - tol) <= t.md <= (md_end + tol)
        )


@dataclass(frozen=True, slots=True)
class PerforationRange:
    start_md: float
    end_md: float

    def __iter__(self) -> Iterator[float]:
        return iter((self.start_md, self.end_md))


@dataclass(frozen=True, slots=True)
class Perforation:
    name: str
    range: PerforationRange
    points: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class Completion:
    perforations: tuple[Perforation, ...]


@dataclass(frozen=True, slots=True)
class Well:
    name: str
    trajectory: Trajectory
    completion: Completion | None = field(default=None)


# Section types


class SectionInterface(ABC):
    """md: measured length; user_inclination: override from trajectory."""

    def __init__(self, md: float, user_inclination: float | None = None):
        self._md: float = md
        self._user_inclination: float | None = user_inclination

    @abstractmethod
    def append_to_trajectory(
        self,
        trajectory: Trajectory,
        md_step: float,
        perforations: MutableMapping[str, PerforationRange] | None = None,
    ) -> Trajectory: ...


class PerforationMdProvider:
    def __init__(
        self,
        perforations: MutableMapping[str, PerforationRange] | None,
        section_start_md: float,
        section_end_md: float,
    ) -> None:
        boundary_points = self._get_perforations_boundary_points_inside_md_range(
            perforations, section_start_md, section_end_md
        )
        self.__perforation_md_iter = iter(boundary_points)

    @staticmethod
    def _get_perforations_boundary_points_inside_md_range(
        perforations: MutableMapping[str, PerforationRange] | None,
        section_start_md: float,
        section_end_md: float,
    ) -> list[float]:
        if not perforations:
            return []
        return [
            p_md
            for perforation in perforations.values()
            for p_md in perforation
            if section_start_md <= p_md <= section_end_md
        ]

    def get_next_md(self) -> float:
        return next(self.__perforation_md_iter, np.inf)


class LinearWellSection(SectionInterface):
    def __init__(self, md: float, user_inclination: float | None = None):
        super().__init__(md, user_inclination)
        self.__logger = get_logger(__name__)
        if user_inclination is None:
            self.__logger.debug(
                "Linear well section inclination is not provided. Calculating inclination from the trajectory"
            )

    def append_to_trajectory(
        self,
        trajectory: Trajectory,
        md_step: float,
        perforations: MutableMapping[str, PerforationRange] | None = None,
    ) -> Trajectory:
        last_point = trajectory.get_last_trajectory_point()
        x, y, z, point_md = last_point
        section_trajectory: list[TrajectoryPoint] = list(trajectory)

        num_steps = int(self._md // md_step)
        remaining_md = self._md % md_step
        inclination = (
            self._user_inclination
            if self._user_inclination is not None
            else trajectory.get_xz_inclination()
        )
        inclination_rad = np.deg2rad(inclination)
        cos_incl = np.cos(inclination_rad)
        sin_incl = np.sin(inclination_rad)

        self.__logger.debug(
            f"Appending linear well section to the trajectory with step: {md_step} [m] and inclination: {inclination} [deg]"
        )

        section_start_md = point_md
        section_end_md = section_start_md + self._md
        perf_provider = PerforationMdProvider(
            perforations, section_start_md, section_end_md
        )
        perforation_md = perf_provider.get_next_md()

        for _ in range(num_steps):
            z += md_step * cos_incl
            x += md_step * sin_incl
            point_md += md_step
            last_added = section_trajectory[-1]
            if last_added.is_close_to_md(perforation_md):
                perforation_md = perf_provider.get_next_md()
            while point_md > perforation_md and not last_added.is_close_to_md(
                perforation_md
            ):
                x_p = x - (point_md - perforation_md) * sin_incl
                z_p = z - (point_md - perforation_md) * cos_incl
                section_trajectory.append(TrajectoryPoint(x_p, y, z_p, perforation_md))
                perforation_md = perf_provider.get_next_md()
            section_trajectory.append(TrajectoryPoint(x, y, z, point_md))

        if remaining_md > 0.0:
            z += remaining_md * cos_incl
            x += remaining_md * sin_incl
            point_md += remaining_md
            last_added = section_trajectory[-1]
            if last_added.is_close_to_md(perforation_md):
                perforation_md = perf_provider.get_next_md()
            while point_md > perforation_md and not last_added.is_close_to_md(
                perforation_md
            ):
                x_p = x - (point_md - perforation_md) * sin_incl
                z_p = z - (point_md - perforation_md) * cos_incl
                section_trajectory.append(TrajectoryPoint(x_p, y, z_p, perforation_md))
                perforation_md = perf_provider.get_next_md()
            section_trajectory.append(TrajectoryPoint(x, y, z, point_md))

        return Trajectory(tuple(section_trajectory))


class CurvedWellSection(SectionInterface):
    def __init__(self, md: float, dls: float, user_inclination: float | None = None):
        super().__init__(md, user_inclination)
        self._dls: float = dls
        self.__logger = get_logger(__name__)
        if user_inclination is None:
            self.__logger.debug(
                "Curved well section inclination is not provided. Calculating inclination from the trajectory"
            )

    def append_to_trajectory(
        self,
        trajectory: Trajectory,
        md_step: float,
        perforations: MutableMapping[str, PerforationRange] | None = None,
    ) -> Trajectory:
        last_point = trajectory.get_last_trajectory_point()
        x, y, z, point_md = last_point
        section_trajectory: list[TrajectoryPoint] = list(trajectory)

        num_steps = int(self._md // md_step)
        remaining_md = self._md % md_step
        inclination_deg = (
            self._user_inclination
            if self._user_inclination is not None
            else trajectory.get_xz_inclination()
        )

        self.__logger.debug(
            f"Appending curved well section to the trajectory with step: {self._md} [m], inclination: {inclination_deg} [deg] and DLS: {self._dls} [deg/30 m]"
        )

        rotation_center = _rotation_center(Point(x, y, z), self._dls, inclination_deg)
        rotation_radius = _circle_radius(self._dls)
        rx = x - rotation_center.x
        rz = z - rotation_center.z
        theta_start_rad = _calculate_theta_start_rad(rx, rz, inclination_deg)
        sign_z = 1 if self._dls <= 0 else -1
        sign_x = 1 if inclination_deg >= 0 else -1

        section_start_md = point_md
        section_end_md = section_start_md + self._md
        perf_provider = PerforationMdProvider(
            perforations, section_start_md, section_end_md
        )
        perforation_md = perf_provider.get_next_md()

        step_arc_length = 0.0
        for _ in range(num_steps):
            step_arc_length += md_step
            theta_rad = theta_start_rad + _angle_increase_rad(
                step_arc_length, rotation_radius
            )
            x = rotation_center.x + rotation_radius * np.cos(theta_rad) * sign_x
            z = rotation_center.z + rotation_radius * np.sin(theta_rad) * sign_z
            point_md += md_step
            last_added = section_trajectory[-1]
            if last_added.is_close_to_md(perforation_md):
                perforation_md = perf_provider.get_next_md()
            while point_md > perforation_md and not last_added.is_close_to_md(
                perforation_md
            ):
                theta_p = theta_start_rad + _angle_increase_rad(
                    perforation_md - last_point.z, rotation_radius
                )
                x_p = rotation_center.x + rotation_radius * np.cos(theta_p) * sign_x
                z_p = rotation_center.z + rotation_radius * np.sin(theta_p) * sign_z
                section_trajectory.append(TrajectoryPoint(x_p, y, z_p, perforation_md))
                perforation_md = perf_provider.get_next_md()
            section_trajectory.append(TrajectoryPoint(x, y, z, point_md))

        if remaining_md > 0.0:
            theta_rad = theta_start_rad + _angle_increase_rad(self._md, rotation_radius)
            x = rotation_center.x + rotation_radius * np.cos(theta_rad) * sign_x
            z = rotation_center.z + rotation_radius * np.sin(theta_rad) * sign_z
            point_md += remaining_md
            last_added = section_trajectory[-1]
            if last_added.is_close_to_md(perforation_md):
                perforation_md = perf_provider.get_next_md()
            while point_md > perforation_md and not last_added.is_close_to_md(
                perforation_md
            ):
                theta_p = theta_start_rad + _angle_increase_rad(
                    perforation_md - last_point.z, rotation_radius
                )
                x_p = rotation_center.x + rotation_radius * np.cos(theta_p) * sign_x
                z_p = rotation_center.z + rotation_radius * np.sin(theta_p) * sign_z
                section_trajectory.append(TrajectoryPoint(x_p, y, z_p, perforation_md))
                perforation_md = perf_provider.get_next_md()
            section_trajectory.append(TrajectoryPoint(x, y, z, point_md))

        return Trajectory(tuple(section_trajectory))


def _rotation_center(last_point: Point, dls: float, inclination0_deg: float) -> Point:
    r = _circle_radius(dls)
    x, y, z = last_point.x, last_point.y, last_point.z
    same_sign = (dls >= 0 and inclination0_deg >= 0) or (
        dls < 0 and inclination0_deg < 0
    )
    sign_x = 1 if same_sign else -1
    sign_z = -1 if dls >= 0 else 1
    x0 = x + sign_x * r * np.cos(np.deg2rad(np.abs(inclination0_deg)))
    z0 = z + sign_z * r * np.sin(np.deg2rad(np.abs(inclination0_deg)))
    return Point(x0, y, z0)


def _circle_radius(dls: float) -> float:
    return float(np.abs(DLS_RATE / np.deg2rad(dls)))


def _angle_increase_rad(arc_length: float, circle_radius: float) -> float:
    return arc_length / circle_radius


def _calculate_theta_start_rad(rx: float, rz: float, inclination: float) -> float:
    if rx == 0 and rz == 0:
        raise ValueError(
            "rx and rz cannot be both 0 - rotation center cannot be in the well bore"
        )
    if rx == 0:
        if rz > 0:
            return float(-np.pi / 2 if inclination >= 0 else 3 * np.pi / 2)
        else:
            return float(np.pi / 2)
    if rx > 0 and rz > 0:
        return float(
            -np.atan2(rz, rx) if inclination >= 0 else np.pi + np.atan2(rz, rx)
        )
    elif rx <= 0 and rz < 0:
        return float(np.atan2(rz, rx) if inclination >= 0 else np.pi - np.atan2(rz, rx))
    elif rx < 0 <= rz:
        return float(
            2 * np.pi - np.atan2(rz, rx)
            if inclination >= 0
            else np.pi + np.atan2(rz, rx)
        )
    elif rx > 0 >= rz:
        return float(np.atan2(rz, rx) if inclination >= 0 else np.pi - np.atan2(rz, rx))
    raise ValueError("Invalid combination of rx and rz")


#  Well builder


class WellBuilder:
    def __init__(self) -> None:
        self._name: str = "Unknown"
        self._wellhead: TrajectoryPoint = TrajectoryPoint(0, 0, 0, 0)
        self._azimuth: float = 0.0
        self._translation: Point = Point(0, 0, 0)
        self._step: float = 0.5
        self._sections: tuple[SectionInterface, ...] | None = None
        self._perforations: MutableMapping[str, PerforationRange] | None = None
        self.__logger = get_logger(__name__)

    def well_name(self, name: str) -> WellBuilder:
        self.__logger.debug(f"Setting up well name: {name}")
        self._name = name
        return self

    def rotate(self, azimuth: float) -> WellBuilder:
        self.__logger.debug(f"Setting up well rotating azimuth: {azimuth} deg")
        self._azimuth = azimuth
        return self

    def translate(self, translation: Point) -> WellBuilder:
        self.__logger.debug(
            f"Setting up well translation vector: {str(TrajectoryPoint)}"
        )
        self._translation = translation
        return self

    def discretize(self, step: float) -> WellBuilder:
        self.__logger.debug(f"Setting up well trajectory sampling step: {step} m")
        self._step = step
        return self

    def sections(self, sections: Sequence[SectionInterface]) -> WellBuilder:
        self.__logger.debug("Setting up well sections...")
        self._sections = tuple(sections)
        return self

    def perforations(
        self, perforations: MutableMapping[str, PerforationRange] | None
    ) -> WellBuilder:
        self.__logger.debug("Setting up well perforation intervals...")
        if not perforations:
            self.__logger.debug("Perforation intervals are empty, skipping...")
            return self
        self._perforations = perforations
        return self

    def build(self) -> Well:
        self.__logger.debug(f"Well {self._name} building process is starting...")
        if not self._sections:
            raise EmptySectionConfigurationException(
                "Well builder doesn't contain valid sections configuration"
            )
        trajectory: Trajectory = Trajectory((self._wellhead,))
        for section in self._sections:
            trajectory = section.append_to_trajectory(
                trajectory, self._step, self._perforations
            )
        trajectory = _rotate_trajectory(self._azimuth, trajectory)
        trajectory = _move_trajectory(self._translation, trajectory)
        completion = _build_completion(trajectory, self._perforations)
        well = Well(name=self._name, trajectory=trajectory, completion=completion)
        self.__logger.debug(f"Well {self._name} building process completed")
        return well


def _rotate_trajectory(angle: float, trajectory: Trajectory) -> Trajectory:
    first_y = trajectory[0].y
    xz_traj = _move_trajectory(Point(x=0, y=-first_y, z=0), trajectory)
    rotated: list[TrajectoryPoint] = []
    angle_rad = np.deg2rad(angle)
    for p in xz_traj:
        x_rot = np.cos(angle_rad) * p.x - np.sin(angle_rad) * p.y
        y_rot = np.sin(angle_rad) * p.x + np.cos(angle_rad) * p.y
        rotated.append(TrajectoryPoint(x=x_rot, y=y_rot, z=p.z, md=p.md))
    return _move_trajectory(Point(x=0, y=first_y, z=0), Trajectory(tuple(rotated)))


def _move_trajectory(translation: Point, trajectory: Trajectory) -> Trajectory:
    return Trajectory(
        tuple(
            TrajectoryPoint(
                x=p.x + translation.x,
                y=p.y + translation.y,
                z=p.z + translation.z,
                md=p.md,
            )
            for p in trajectory
        )
    )


def _build_completion(
    trajectory: Trajectory,
    perforations: MutableMapping[str, PerforationRange] | None,
) -> Completion | None:
    if not perforations:
        return None
    completion_perforations = []
    for p_name, p_range in perforations.items():
        perforation_points = trajectory.get_points_in_md_range(
            p_range.start_md, p_range.end_md
        )
        if perforation_points:
            completion_perforations.append(
                Perforation(p_name, p_range, perforation_points)
            )
    if not completion_perforations:
        return None
    return Completion(tuple(completion_perforations))


# Well templates


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


class IWellTemplate(WellTemplateInterface):
    def __init__(
        self,
        name: str,
        md: float,
        wellhead: TrajectoryPoint,
        md_step: float,
        perforations: MutableMapping[str, PerforationRange] | None = None,
    ):
        zero_azimuth = 0
        super().__init__(name, md_step, wellhead, zero_azimuth, perforations)
        get_logger(__name__).debug(f"Creating IWellTemplate model with MD: {md} m")
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
        get_logger(__name__).debug(
            f"Creating JWellTemplate model with md_linear1: {md_linear1} m, md_curved: {md_curved} m, dls: {dls} deg/30 m, md_linear2: {md_linear2} m"
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
        get_logger(__name__).debug(
            f"Creating SWellTemplate model with md_linear1: {md_linear1} m, md_curved1: {md_curved1} m, dls1: {dls1} deg/30 m, md_linear2: {md_linear2} m, md_curved2: {md_curved2} m, dls2: {dls2} deg/30 m, md_linear3: {md_linear3} m"
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


class HWellTemplate(WellTemplateInterface):
    """H-shaped (horizontal) well template."""

    FIXED_ANGLE: float = 90.0  # degrees
    FIXED_DLS: float = 4.0  # degrees per 30 metres
    _CURVATURE_MD: float = (FIXED_ANGLE * 30) / FIXED_DLS
    _CURVATURE_RADIUS: float = 2 * _CURVATURE_MD / np.pi

    def __init__(
        self,
        name: str,
        TVD: float,
        md_lateral: float,
        wellhead: TrajectoryPoint,
        azimuth: float,
        md_step: float,
        perforations: MutableMapping[str, PerforationRange] | None = None,
    ):
        super().__init__(name, md_step, wellhead, azimuth, perforations)
        md_linear1, md_curved, md_linear2 = self.get_sections_mds(TVD, md_lateral)
        self._well_builder.sections(
            [
                LinearWellSection(md=md_linear1),
                CurvedWellSection(md=md_curved, dls=self.FIXED_DLS),
                LinearWellSection(md=md_linear2),
            ]
        )

    @classmethod
    def get_sections_mds(cls, depth: float, width: float) -> tuple[float, float, float]:
        if (vertical_LWS_md := depth - cls._CURVATURE_RADIUS) < 0:
            raise ValueError(
                "Horizontal well true total depth is less than curved well section radius. Increase depth."
            )
        if (horizontal_LWS_md := width - cls._CURVATURE_RADIUS) < 0:
            raise ValueError(
                "Horizontal well width is less than curved well section radius. Increase width."
            )
        return vertical_LWS_md, cls._CURVATURE_MD, horizontal_LWS_md

    @classmethod
    def from_model(cls, model: HWellModel) -> HWellTemplate:
        return cls(
            name=model.name,
            TVD=model.TVD,
            md_lateral=model.md_lateral,
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


#  Well models (Pydantic)

type PerforationName = str


def _has_valid_range(start_md: float, end_md: float) -> bool:
    return end_md > start_md


def _duplicate_well_names(wells_name: list[str]) -> list[str]:
    counts = Counter(wells_name)
    return [name for name in counts if counts[name] > 1]


class PerforationRangeModel(BaseModel, extra="forbid"):
    start_md: float = Field(ge=0.0)
    end_md: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_perforation_start_and_end(self) -> Self:
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
    def sort_perforations(self) -> Self:
        if self.perforations:
            self.perforations = _sort_perforations(self.perforations)
        return self

    @model_validator(mode="after")
    def validate_perforations_not_overlap(self) -> Self:
        if not _are_perforations_non_overlapping(self.perforations):
            raise ValueError("Perforations can't overlap")
        return self


class IWellModel(_BaseWellModel, extra="forbid", str_strip_whitespace=True):
    well_type: Literal["IWell"] = Field(default="IWell")
    md: float = Field(gt=0.0)

    @model_validator(mode="after")
    def adjust_perforations_to_well_md(self) -> Self:
        self.perforations = _adjust_perforations(self.md, self.perforations)
        return self


class JWellModel(_BaseWellModel, extra="forbid", str_strip_whitespace=True):
    well_type: Literal["JWell"] = Field(default="JWell")
    md_linear1: float = Field(gt=0.0)
    md_curved: float = Field(gt=0.0)
    dls: float = Field(gt=-45.00, le=45.00)
    md_linear2: float = Field(gt=0.0)
    azimuth: float = Field(ge=0.0, lt=360.0)

    @model_validator(mode="after")
    def adjust_perforations_to_well_md(self) -> Self:
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
    def adjust_perforations_to_well_md(self) -> Self:
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
    def adjust_perforations_to_well_md(self) -> Self:
        md_linear1, md_curved, md_linear2 = HWellTemplate.get_sections_mds(
            depth=self.TVD, width=self.md_lateral
        )
        total_md = md_linear1 + md_curved + md_linear2
        self.perforations = _adjust_perforations(total_md, self.perforations)
        return self


WellModel = Annotated[
    IWellModel | JWellModel | SWellModel | HWellModel,
    Field(discriminator="well_type"),
]


class WellDesignServiceRequest(BaseModel, extra="forbid"):
    models: list[WellModel]

    @model_validator(mode="after")
    def validate_wells_name(self) -> Self:
        wells_name = [n.name for n in self.models]
        if duplicates := _duplicate_well_names(wells_name):
            raise ValueError(f"Well names must be unique. Duplicate:{duplicates}")
        return self


def _adjust_perforations(
    total_md: float,
    perforations: dict[PerforationName, PerforationRangeModel] | None,
) -> dict[PerforationName, PerforationRangeModel] | None:
    if not perforations:
        return None
    adjusted: dict[str, PerforationRangeModel] = {}
    for name, perf in perforations.items():
        if perf.start_md >= total_md:
            continue
        new_end_md = min(perf.end_md, total_md)
        if perf.start_md < new_end_md:
            adjusted[name] = PerforationRangeModel(
                start_md=perf.start_md, end_md=new_end_md
            )
    return adjusted if adjusted else None


def _are_perforations_non_overlapping(
    perforations: dict[PerforationName, PerforationRangeModel] | None,
) -> bool:
    if not perforations or len(perforations) < 2:
        return True
    ordered = sorted(perforations.values(), key=lambda p: p.start_md)
    return all(cur.end_md <= nxt.start_md for cur, nxt in itertools.pairwise(ordered))


def _sort_perforations(
    perforations: dict[PerforationName, PerforationRangeModel],
) -> dict[PerforationName, PerforationRangeModel]:
    return dict(sorted(perforations.items(), key=lambda item: item[1].start_md))


# Simulation payload types (plugin-private — define what the simulator connector consumes)


def _has_valid_simulation_range(start_md: float, end_md: float) -> bool:
    return end_md > start_md


def _duplicate_simulation_names(names: list[str]) -> list[str]:
    counts = Counter(names)
    return [n for n in counts if counts[n] > 1]


class SimulationWellPerforationModel(BaseModel, extra="forbid"):
    name: str
    range: tuple[float, float]
    points: tuple[tuple[float, float, float], ...]

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        start_md, end_md = self.range
        if not _has_valid_simulation_range(start_md, end_md):
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
        names = [w.name for w in self.wells]
        if duplicates := _duplicate_simulation_names(names):
            raise ValueError(f"Well names must be unique. Duplicate:{duplicates}")
        return self


# Domain service


def _build_simulation_well_model(well: Well) -> SimulationWellModel:
    return SimulationWellModel(
        name=well.name,
        trajectory=tuple((t.x, t.y, t.z) for t in well.trajectory),
        completion=(
            SimulationWellCompletionModel(
                perforations=tuple(
                    SimulationWellPerforationModel(
                        name=p.name,
                        points=tuple((pt.x, pt.y, pt.z) for pt in p.points),
                        range=(p.range.start_md, p.range.end_md),
                    )
                    for p in well.completion.perforations
                )
            )
            if well.completion
            else None
        ),
    )


class WellDesignDomainService(DomainServiceInterface):
    """Reference domain service: dispatches each well model to its geometry template."""

    ServiceName = "well_design"

    @classmethod
    def get_item_state_adapter(cls) -> TypeAdapter[Any]:
        return TypeAdapter(WellModel)

    def build_payload(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        request = WellDesignServiceRequest(models=items)
        wells: list[SimulationWellModel] = []
        for model in request.models:
            logger.debug("Building well %r (type=%s)", model.name, type(model).__name__)
            match model:
                case IWellModel():
                    well = IWellTemplate.from_model(model).build()
                case JWellModel():
                    well = JWellTemplate.from_model(model).build()
                case SWellModel():
                    well = SWellTemplate.from_model(model).build()
                case HWellModel():
                    well = HWellTemplate.from_model(model).build()
                case _ as unreachable:
                    assert_never(unreachable)
            wells.append(_build_simulation_well_model(well))
        # Validate before returning — keeps unique-name and range checks on the outbound path.
        return WellDesignServiceResponse(wells=wells).model_dump()


# Plugin descriptor

plugin = DomainServicePlugin(
    name=WellDesignDomainService.ServiceName,
    implementation=WellDesignDomainService,
)
