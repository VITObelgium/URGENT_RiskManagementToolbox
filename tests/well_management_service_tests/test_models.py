import math
from typing import Type

import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest

from services.well_management_service.core.models import (
    PerforationRange,
    Point,
    Trajectory,
    TrajectoryPoint,
)
from services.well_management_service.core.utilities.constants import POINT_ATOL


@pytest.fixture
def trajectory_point_fixture(request: FixtureRequest) -> TrajectoryPoint:
    x, y, z, md = request.param
    return TrajectoryPoint(x, y, z, md)


@pytest.mark.parametrize(
    "trajectory_point_fixture, other_md, result",
    [
        ((0.0, 0.0, 0.0, 0.0), 0.0, True),
        ((0.0, 0.0, 0.0, 0.0), 1.0, False),
        ((0.0, 0.0, 0.0, 1.0), 1.0 + POINT_ATOL, True),
        ((0.0, 0.0, 0.0, 1.0), 1.0 - POINT_ATOL, True),
        ((0.0, 0.0, 0.0, 1.0), 1.0 + POINT_ATOL + 1e-6, False),
        ((0.0, 0.0, 0.0, 1.0), 1.0 - POINT_ATOL - 1e-6, False),
    ],
    indirect=["trajectory_point_fixture"],
)
def test_trajectory_point_close_to_md(
    trajectory_point_fixture: TrajectoryPoint, other_md: float, result: bool
) -> None:
    tp = trajectory_point_fixture
    assert tp.is_close_to_md(other_md) == result


@pytest.mark.parametrize(
    "x, y, z, md",
    [
        (1.0, 2.0, 3.0, 4.0),
    ],
)
def test_trajectory_point_iter(x: float, y: float, z: float, md: float) -> None:
    tp = TrajectoryPoint(x, y, z, md)
    x_tp, y_tp, z_tp, md_tp = tp
    assert x_tp == x
    assert y_tp == y
    assert z_tp == z
    assert md_tp == md


@pytest.mark.parametrize(
    "x, y, z",
    [
        (1.0, 2.0, 3.0),
    ],
)
def test_point_iter(x: float, y: float, z: float) -> None:
    p = Point(x, y, z)
    x_p, y_p, z_p = p
    assert x_p == x
    assert y_p == y
    assert z_p == z


@pytest.mark.parametrize(
    "start_md, end_md",
    [
        (1.0, 2.0),
    ],
)
def test_perforation_range_iter(start_md: float, end_md: float) -> None:
    pr = PerforationRange(start_md, end_md)
    pr_s, pr_e = pr
    assert pr_s == start_md
    assert pr_e == end_md


def test_trajectory_should_raise_exception_when_no_points() -> None:
    tp: tuple[TrajectoryPoint, ...] = tuple([])
    with pytest.raises(ValueError):
        Trajectory(tp)


@pytest.mark.parametrize(
    "trajectory_points",
    [
        (
            TrajectoryPoint(0, 1, 2, 3),
            TrajectoryPoint(4, 5, 6, 7),
            TrajectoryPoint(8, 9, 10, 11),
        )
    ],
)
def test_trajectory_get_last_point(
    trajectory_points: tuple[TrajectoryPoint, ...],
) -> None:
    t = Trajectory(trajectory_points)
    assert t.get_last_trajectory_point() == trajectory_points[-1]


@pytest.mark.parametrize(
    "trajectory_points, result_inclination",
    [
        ((TrajectoryPoint(0.0, 0.0, 0.0, 0.0),), 0.0),
        (
            (
                TrajectoryPoint(0.0, 0.0, 0.0, 0.0),
                TrajectoryPoint(0.0, 0.0, 10.0, 10.0),
            ),
            0.0,
        ),
        (
            (
                TrajectoryPoint(0.0, 0.0, 10.0, 10.0),
                TrajectoryPoint(0.0, 0.0, 0.0, 0.0),
            ),
            180.0,
        ),
        (
            (
                TrajectoryPoint(0.0, 0.0, 0.0, 0.0),
                TrajectoryPoint(10.0, 0.0, 0.0, 10.0),
            ),
            90.0,
        ),
        (
            (TrajectoryPoint(0, 0, 0, 0), TrajectoryPoint(10, 0, 10, np.sqrt(2) * 10)),
            45.0,
        ),
        ((TrajectoryPoint(0, 0, 0, 0), TrajectoryPoint(-10, 0, 0, 0)), -90.0),
    ],
)
def test_trajectory_get_xz_inclination(
    trajectory_points: tuple[TrajectoryPoint, ...], result_inclination: float
) -> None:
    t = Trajectory(trajectory_points)
    assert math.isclose(t.get_xz_inclination(), result_inclination)


@pytest.fixture
def trajectory_points() -> tuple[TrajectoryPoint, ...]:
    return (
        TrajectoryPoint(0, 0, 0, 0),
        TrajectoryPoint(0, 0, 1.1, 1.1),
        TrajectoryPoint(0, 0, 2.2, 2.2),
        TrajectoryPoint(0, 0, 3.3, 3.3),
        TrajectoryPoint(0, 0, 4.4, 4.4),
    )


@pytest.fixture
def trajectory(trajectory_points: tuple[TrajectoryPoint, ...]) -> Trajectory:
    return Trajectory(trajectory_points)


@pytest.mark.parametrize(
    "md_start, md_end, expected_points",
    [
        (0, 3, [Point(0, 0, 0), Point(0, 0, 1.1), Point(0, 0, 2.2)]),
        (1, 3.3, [Point(0, 0, 1.1), Point(0, 0, 2.2), Point(0, 0, 3.3)]),
        (3.3, 4.4, [Point(0, 0, 3.3), Point(0, 0, 4.4)]),
        (5.0, 6.0, []),
    ],
)
def test_trajectory_get_points_in_md_range(
    trajectory: Trajectory, md_start: float, md_end: float, expected_points: list[Point]
) -> None:
    result = trajectory.get_points_in_md_range(md_start, md_end)
    assert result == tuple(expected_points)


@pytest.mark.parametrize(
    "input_data, expected_exception",
    [
        (Point(0, 0, 0), TypeError),
        (TrajectoryPoint(0, 0, 0, 0), TypeError),
    ],
)
def test_point_should_raise_type_error(
    input_data: Point | TrajectoryPoint, expected_exception: Type[BaseException]
) -> None:
    with pytest.raises(expected_exception):
        _ = input_data[0]  # type: ignore
