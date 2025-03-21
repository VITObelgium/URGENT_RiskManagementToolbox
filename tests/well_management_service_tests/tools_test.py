import pytest

from services.well_management_service.core.models import TrajectoryPoint
from tests.well_management_service_tests.tools import (
    is_subsequence_points,
)


@pytest.mark.parametrize(
    "output_trajectory, trajectory_expected_points_in_order, result",
    [
        (
            (
                TrajectoryPoint(1.0, 2.0, 3.0, 4.0),
                TrajectoryPoint(5.0, 6.0, 7.0, 8.0),
                TrajectoryPoint(9.0, 10.0, 11.0, 12.0),
            ),
            (
                TrajectoryPoint(1.0, 2.0, 3.0, 4.0),
                TrajectoryPoint(9.0, 10.0, 11.0, 12.0),
            ),
            True,
        ),
        (
            (
                TrajectoryPoint(0, 0, 0, 0),
                TrajectoryPoint(0, 0, 0, 10),
                TrajectoryPoint(0, 0, 50, 50),
                TrajectoryPoint(0, 0, 0, 20),
                TrajectoryPoint(0, 0, 0, 100),
            ),
            (
                TrajectoryPoint(0, 0, 0, 0),
                TrajectoryPoint(0, 0, 0, 10),
                TrajectoryPoint(0, 0, 0, 20),
                TrajectoryPoint(0, 0, 0, 50),
                TrajectoryPoint(0, 0, 0, 100),
            ),
            False,
        ),
    ],
)
def test_is_subsequence(
    output_trajectory: tuple[TrajectoryPoint, ...],
    trajectory_expected_points_in_order: tuple[TrajectoryPoint, ...],
    result: bool,
) -> None:
    assert (
        is_subsequence_points(output_trajectory, trajectory_expected_points_in_order)
        == result
    )
