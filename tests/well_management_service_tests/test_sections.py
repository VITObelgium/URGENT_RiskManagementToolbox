from typing import Sequence

import numpy as np
import pytest

from services.well_management_service.core.models import (
    PerforationRange,
    Trajectory,
    TrajectoryPoint,
)
from services.well_management_service.core.sections import (
    CurvedWellSection,
    LinearWellSection,
)
from tests.well_management_service_tests.tools import (
    is_subsequence_points,
)


@pytest.mark.parametrize(
    "input_md,  input_user_inclination, input_trajectory, input_md_step, input_perforations, must_contain_trajectory_points_in_order",
    [
        (
            100.0,
            0.0,
            Trajectory((TrajectoryPoint(10.0, 20.0, 0, 0),)),
            200,
            None,
            (TrajectoryPoint(10.0, 20.0, 0, 0), TrajectoryPoint(10.0, 20.0, 100, 100)),
        ),
        (
            100.0,
            0.0,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            50,
            None,
            (TrajectoryPoint(0, 0, 50, 50), TrajectoryPoint(0, 0, 100, 100)),
        ),
        (
            100.0,
            None,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            50,
            None,
            (TrajectoryPoint(0, 0, 50, 50), TrajectoryPoint(0, 0, 100, 100)),
        ),
        (
            100.0,
            None,
            Trajectory((TrajectoryPoint(0, 0, 0, 0), TrajectoryPoint(50, 0, 50, 50))),
            200,
            None,
            (
                TrajectoryPoint(50, 0, 50, 50),
                TrajectoryPoint(120.710678, 0, 120.710678, 150),
            ),
        ),
        (
            100.0,
            0.0,
            Trajectory((TrajectoryPoint(0, 0, 0, 0), TrajectoryPoint(50, 0, 50, 50))),
            200,
            None,
            (
                TrajectoryPoint(50, 0, 50, 50),
                TrajectoryPoint(50, 0, 150, 150),
            ),
        ),
        (
            100.0,
            0.0,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            50.0,
            [PerforationRange(10.0, 20.0)],
            (
                TrajectoryPoint(0, 0, 0, 0),
                TrajectoryPoint(0, 0, 10, 10),
                TrajectoryPoint(0, 0, 20, 20),
                TrajectoryPoint(0, 0, 50, 50),
                TrajectoryPoint(0, 0, 100, 100),
            ),
        ),
        (
            100.0,
            0.0,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            50.0,
            [PerforationRange(10.0, 20.0), PerforationRange(50.0, 75.0)],
            (
                TrajectoryPoint(0, 0, 0, 0),
                TrajectoryPoint(0, 0, 10, 10),
                TrajectoryPoint(0, 0, 20, 20),
                TrajectoryPoint(0, 0, 50, 50),
                TrajectoryPoint(0, 0, 75, 75),
                TrajectoryPoint(0, 0, 100, 100),
            ),
        ),
        (
            100.0,
            0.0,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            30.0,
            [PerforationRange(10.0, 20.0), PerforationRange(91.0, 99.0)],
            (
                TrajectoryPoint(0, 0, 0, 0),
                TrajectoryPoint(0, 0, 10, 10),
                TrajectoryPoint(0, 0, 20, 20),
                TrajectoryPoint(0, 0, 30, 30),
                TrajectoryPoint(0, 0, 60, 60),
                TrajectoryPoint(0, 0, 90, 90),
                TrajectoryPoint(0, 0, 91, 91),
                TrajectoryPoint(0, 0, 99, 99),
                TrajectoryPoint(0, 0, 100, 100),
            ),
        ),
        (
            100.0,
            0.0,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            30.0,
            [PerforationRange(10.0, 20.0), PerforationRange(51.0, 99.0)],
            (
                TrajectoryPoint(0, 0, 0, 0),
                TrajectoryPoint(0, 0, 10, 10),
                TrajectoryPoint(0, 0, 20, 20),
                TrajectoryPoint(0, 0, 30, 30),
                TrajectoryPoint(0, 0, 51, 51),
                TrajectoryPoint(0, 0, 60, 60),
                TrajectoryPoint(0, 0, 90, 90),
                TrajectoryPoint(0, 0, 99, 99),
                TrajectoryPoint(0, 0, 100, 100),
            ),
        ),
    ],
)
def test_linear_well_section(
    input_md: float,
    input_user_inclination: float | None,
    input_trajectory: Trajectory,
    input_md_step: float,
    input_perforations: Sequence[PerforationRange] | None,
    must_contain_trajectory_points_in_order: tuple[TrajectoryPoint, ...],
) -> None:
    linear_well_section = LinearWellSection(input_md, input_user_inclination)
    output_trajectory = linear_well_section.append_to_trajectory(
        input_trajectory, input_md_step, input_perforations
    )
    assert is_subsequence_points(
        output_trajectory, must_contain_trajectory_points_in_order
    )


# The following combination of parameters should give us the 1/4 of full circle
md = 300.0
dls = 9.0
R = md / (np.deg2rad(dls) * md / 30.0)


@pytest.mark.parametrize(
    "input_md, input_dls, input_user_inclination, input_trajectory, input_md_step, input_perforations, must_contain_trajectory_points_in_order",
    [
        (
            md,
            dls,
            None,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            400,
            None,
            ((TrajectoryPoint(0, 0, 0, 0)), TrajectoryPoint(R, 0, R, md)),
        ),
        (
            md,
            -dls,
            None,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            400,
            None,
            ((TrajectoryPoint(0, 0, 0, 0)), TrajectoryPoint(-R, 0, R, md)),
        ),
        (
            md,
            dls,
            0.0,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            400,
            None,
            ((TrajectoryPoint(0, 0, 0, 0)), TrajectoryPoint(R, 0, R, md)),
        ),
        (
            md,
            dls,
            90.0,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            400,
            None,
            ((TrajectoryPoint(0, 0, 0, 0)), TrajectoryPoint(R, 0, -R, md)),
        ),
        (
            md,
            dls,
            -90.0,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            400,
            None,
            ((TrajectoryPoint(0, 0, 0, 0)), TrajectoryPoint(-R, 0, -R, md)),
        ),
        (
            md,
            -dls,
            -90.0,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            400,
            None,
            ((TrajectoryPoint(0, 0, 0, 0)), TrajectoryPoint(-R, 0, R, md)),
        ),
        (
            md,
            dls,
            None,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            150.0,
            None,
            (
                (TrajectoryPoint(0, 0, 0, 0)),
                (TrajectoryPoint(55.93848428670847, 0, 135.0474474235659, 150.0)),
                TrajectoryPoint(R, 0, R, md),
            ),
        ),
        (
            md,
            dls,
            None,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            140.0,
            None,
            (
                (TrajectoryPoint(0, 0, 0, 0)),
                (TrajectoryPoint(49.05572482, 0, 127.79453229, 140.0)),
                (TrajectoryPoint(171.02246576, 0, 189.93969079, 280.0)),
                TrajectoryPoint(R, 0, R, md),
            ),
        ),
        (
            md,
            dls,
            None,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            150,
            [PerforationRange(100, 200)],
            (
                (TrajectoryPoint(0, 0, 0, 0)),
                (TrajectoryPoint(25.58726308, 0, 95.49296586, 100)),
                (TrajectoryPoint(55.93848428670847, 0, 135.0474474235659, 150.0)),
                (TrajectoryPoint(95.49296586, 0, 165.3986686, 200)),
                TrajectoryPoint(R, 0, R, md),
            ),
        ),
        (
            md,
            dls,
            None,
            Trajectory((TrajectoryPoint(0, 0, 0, 0),)),
            140,
            [PerforationRange(100, 280), PerforationRange(295, 300)],
            (
                (TrajectoryPoint(0, 0, 0, 0)),
                (TrajectoryPoint(25.58726308, 0, 95.49296586, 100)),
                (TrajectoryPoint(49.05572482, 0, 127.79453229, 140.0)),
                (TrajectoryPoint(171.02246576, 0, 189.93969079, 280.0)),
                (TrajectoryPoint(185.9865028, 0, 190.9204856, 295.0)),
                TrajectoryPoint(R, 0, R, md),
            ),
        ),
        (
            md,
            dls,
            None,
            Trajectory((TrajectoryPoint(0, 0, 200, 200),)),
            140,
            [PerforationRange(300, 480), PerforationRange(495, 500)],
            (
                (TrajectoryPoint(0, 0, 200, 200)),
                (TrajectoryPoint(25.58726308, 0, 295.4929659, 300)),
                (TrajectoryPoint(49.05572482, 0, 327.7945323, 340.0)),
                (TrajectoryPoint(171.02246576, 0, 389.9396908, 480.0)),
                (TrajectoryPoint(185.9865028, 0, 390.9204856, 495.0)),
                TrajectoryPoint(R, 0, R + 200, md + 200),
            ),
        ),
    ],
)
def test_curved_well_section(
    input_md: float,
    input_dls: float,
    input_user_inclination: float | None,
    input_trajectory: Trajectory,
    input_md_step: float,
    input_perforations: Sequence[PerforationRange] | None,
    must_contain_trajectory_points_in_order: tuple[TrajectoryPoint, ...],
) -> None:
    curved_well_section = CurvedWellSection(input_md, input_dls, input_user_inclination)
    output_trajectory = curved_well_section.append_to_trajectory(
        input_trajectory, input_md_step, input_perforations
    )
    assert is_subsequence_points(
        output_trajectory, must_contain_trajectory_points_in_order
    )
