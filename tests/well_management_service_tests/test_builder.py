from typing import MutableMapping, Sequence

import numpy as np
import pytest

from services.well_management_service.core.models import (
    PerforationRange,
    Point,
    TrajectoryPoint,
)
from services.well_management_service.core.sections import (
    CurvedWellSection,
    LinearWellSection,
    SectionInterface,
)
from services.well_management_service.core.service import WellBuilder
from services.well_management_service.utils._exceptions import (
    EmptySectionConfigurationException,
)
from tests.well_management_service_tests.tools import (
    is_subsequence_points,
)

PerforationRangeAlias = tuple[float, float]
PerforationPointsAlias = tuple[Point, ...]
PerforationAlias = tuple[PerforationRangeAlias, PerforationPointsAlias]

# The following combination of parameters should give us the 1/4 of full circle
md = 300.0
dls = 9.0
R = md / (np.deg2rad(dls) * md / 30.0)


@pytest.mark.parametrize(
    "name, azimuth, translation, discretize, sections, perforations, expected_trajectory_points_in_order, expected_perforations",
    [
        (
            "W1",
            0.0,
            Point(0, 0, 0),
            100,
            [LinearWellSection(200)],
            None,
            (
                TrajectoryPoint(0, 0, 0, 0),
                TrajectoryPoint(0, 0, 100, 100),
                TrajectoryPoint(0, 0, 200, 200),
            ),
            None,
        ),
        (
            "W1",
            0.0,
            Point(0, 0, 0),
            100,
            [LinearWellSection(200)],
            {},
            (
                TrajectoryPoint(0, 0, 0, 0),
                TrajectoryPoint(0, 0, 100, 100),
                TrajectoryPoint(0, 0, 200, 200),
            ),
            None,
        ),
        (
            "W1",
            0.0,
            Point(0, 0, 0),
            100,
            [LinearWellSection(200)],
            {"p1": PerforationRange(300, 500)},
            (
                TrajectoryPoint(0, 0, 0, 0),
                TrajectoryPoint(0, 0, 100, 100),
                TrajectoryPoint(0, 0, 200, 200),
            ),
            None,
        ),
        (
            "W1",
            0.0,
            Point(0, 0, 0),
            100,
            [LinearWellSection(200)],
            {"p1": PerforationRange(10, 20)},
            (
                TrajectoryPoint(0, 0, 0, 0),
                TrajectoryPoint(0, 0, 10, 10),
                TrajectoryPoint(0, 0, 20, 20),
                TrajectoryPoint(0, 0, 100, 100),
                TrajectoryPoint(0, 0, 200, 200),
            ),
            (((10, 20), (Point(0, 0, 10), Point(0, 0, 20))),),
        ),
        (
            "W1",
            0.0,
            Point(0, 0, 0),
            100,
            [LinearWellSection(200), CurvedWellSection(md, dls)],
            None,
            (
                TrajectoryPoint(0, 0, 0, 0),
                TrajectoryPoint(0, 0, 100, 100),
                TrajectoryPoint(0, 0, 200, 200),
                TrajectoryPoint(25.58726308, 0, 295.4929659, 300),
                TrajectoryPoint(95.49296586, 0, 365.3986686, 400),
                TrajectoryPoint(R, 0, R + 200, 500),
            ),
            None,
        ),
        (
            "W1",
            0.0,
            Point(0, 0, 0),
            100,
            [LinearWellSection(200), CurvedWellSection(md, dls)],
            {"p1": PerforationRange(10, 350)},
            (
                TrajectoryPoint(0, 0, 0, 0),
                TrajectoryPoint(0, 0, 10, 10),
                TrajectoryPoint(0, 0, 100, 100),
                TrajectoryPoint(0, 0, 200, 200),
                TrajectoryPoint(25.58726308, 0, 295.4929659, 300),
                TrajectoryPoint(55.93848429, 0, 335.0474474, 350),
                TrajectoryPoint(95.49296586, 0, 365.3986686, 400),
                TrajectoryPoint(R, 0, R + 200, 500),
            ),
            (
                (
                    (10, 350),
                    (
                        Point(0, 0, 10),
                        Point(0, 0, 100),
                        Point(0, 0, 200),
                        Point(25.58726308, 0, 295.4929659),
                        Point(55.93848429, 0, 335.0474474),
                    ),
                ),
            ),
        ),
        (
            "W1",
            0.0,
            Point(0, 100, -10),
            100,
            [LinearWellSection(200), CurvedWellSection(md, dls)],
            {"p1": PerforationRange(10, 350)},
            (
                TrajectoryPoint(0, 100, -10, 0),
                TrajectoryPoint(0, 100, 0, 10),
                TrajectoryPoint(0, 100, 90, 100),
                TrajectoryPoint(0, 100, 190, 200),
                TrajectoryPoint(25.58726308, 100, 285.4929659, 300),
                TrajectoryPoint(55.93848429, 100, 325.0474474, 350),
                TrajectoryPoint(95.49296586, 100, 355.3986686, 400),
                TrajectoryPoint(R, 100, R + 200 - 10, 500),
            ),
            (
                (
                    (10, 350),
                    (
                        Point(0, 100, 0),
                        Point(0, 100, 90),
                        Point(0, 100, 190),
                        Point(25.58726308, 100, 285.4929659),
                        Point(55.93848429, 100, 325.0474474),
                    ),
                ),
            ),
        ),
        (
            "W1",
            90.0,
            Point(0, 0, 0),
            100,
            [LinearWellSection(200), CurvedWellSection(md, dls)],
            {"p1": PerforationRange(10, 350)},
            (
                TrajectoryPoint(0, 0, 0, 0),
                TrajectoryPoint(0, 0, 10, 10),
                TrajectoryPoint(0, 0, 100, 100),
                TrajectoryPoint(0, 0, 200, 200),
                TrajectoryPoint(0, 25.58726308, 295.4929659, 300),
                TrajectoryPoint(0, 55.93848429, 335.0474474, 350),
                TrajectoryPoint(0, 95.49296586, 365.3986686, 400),
                TrajectoryPoint(0, R, R + 200, 500),
            ),
            (
                (
                    (10, 350),
                    (
                        Point(0, 0, 10),
                        Point(0, 0, 100),
                        Point(0, 0, 200),
                        Point(0, 25.58726308, 295.4929659),
                        Point(0, 55.93848429, 335.0474474),
                    ),
                ),
            ),
        ),
        pytest.param(
            "W1",
            90,
            Point(0, 0, 0),
            100,
            [],
            None,
            None,
            None,
            marks=pytest.mark.xfail(
                strict=True, raises=EmptySectionConfigurationException
            ),
        ),
    ],
)
def test_well_builder(
    name: str,
    azimuth: float,
    translation: Point,
    discretize: float,
    sections: Sequence[SectionInterface],
    perforations: MutableMapping[str, PerforationRange] | None,
    expected_trajectory_points_in_order: tuple[TrajectoryPoint, ...],
    expected_perforations: tuple[PerforationAlias, ...] | None,
) -> None:
    wb = (
        WellBuilder()
        .well_name(name)
        .rotate(azimuth)
        .translate(translation)
        .discretize(discretize)
        .sections(sections)
        .perforations(perforations)
    )

    well = wb.build()

    assert well.name == name
    assert is_subsequence_points(well.trajectory, expected_trajectory_points_in_order)

    if expected_perforations is None:
        assert well.completion is None
    else:
        assert well.completion is not None
        for output_perforation, expected_perforation in zip(
            well.completion.perforations, expected_perforations
        ):
            assert output_perforation.range.start_md == expected_perforation[0][0]  #
            assert output_perforation.range.end_md == expected_perforation[0][1]
            assert is_subsequence_points(
                output_perforation.points, expected_perforation[1]
            )
