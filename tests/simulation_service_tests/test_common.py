import pytest

from services.simulation_service.core.connectors.common import (
    Point,
    WellManagementServiceResultSchema,
    WellName,
    extract_well_with_perforations_points,
)


@pytest.mark.parametrize(
    "wells_result, expected_output",
    [
        (
            {
                "wells": [
                    {
                        "name": "Test IWell",
                        "trajectory": [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 10.0],
                            [0.0, 0.0, 50.0],
                            [0.0, 0.0, 60.0],
                            [0.0, 0.0, 70.0],
                            [0.0, 0.0, 80.0],
                            [0.0, 0.0, 100.0],
                        ],
                        "completion": {
                            "perforations": [
                                {
                                    "range": [10.0, 60.0],
                                    "points": [
                                        [0.0, 0.0, 10.0],
                                        [0.0, 0.0, 50.0],
                                        [0.0, 0.0, 60.0],
                                    ],
                                },
                                {
                                    "range": [70.0, 80.0],
                                    "points": [[0.0, 0.0, 70.0], [0.0, 0.0, 80.0]],
                                },
                            ]
                        },
                    }
                ]
            },
            {
                "Test IWell": (
                    [0.0, 0.0, 10.0],
                    [0.0, 0.0, 50.0],
                    [0.0, 0.0, 60.0],
                    [0.0, 0.0, 70.0],
                    [0.0, 0.0, 80.0],
                )
            },
        ),
        (
            {
                "wells": [
                    {
                        "name": "Test IWell",
                        "trajectory": [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 10.0],
                            [0.0, 0.0, 50.0],
                            [0.0, 0.0, 60.0],
                            [0.0, 0.0, 70.0],
                            [0.0, 0.0, 80.0],
                            [0.0, 0.0, 100.0],
                        ],
                        "completion": None,
                    }
                ]
            },
            {"Test IWell": ()},
        ),
        (
            {
                "wells": [
                    {
                        "name": "Test IWell",
                        "trajectory": [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 10.0],
                            [0.0, 0.0, 50.0],
                            [0.0, 0.0, 60.0],
                            [0.0, 0.0, 70.0],
                            [0.0, 0.0, 80.0],
                            [0.0, 0.0, 100.0],
                        ],
                        "completion": {"perforations": []},
                    }
                ]
            },
            {"Test IWell": ()},
        ),
        (
            {
                "wells": [
                    {
                        "name": "Test IWell",
                        "trajectory": [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 10.0],
                            [0.0, 0.0, 50.0],
                            [0.0, 0.0, 60.0],
                            [0.0, 0.0, 70.0],
                            [0.0, 0.0, 80.0],
                            [0.0, 0.0, 100.0],
                        ],
                        "completion": {
                            "perforations": [
                                {
                                    "range": [10.0, 60.0],
                                    "points": [
                                        [0.0, 0.0, 10.0],
                                        [0.0, 0.0, 50.0],
                                        [0.0, 0.0, 60.0],
                                    ],
                                },
                                {
                                    "range": [70.0, 80.0],
                                    "points": [[0.0, 0.0, 70.0], [0.0, 0.0, 80.0]],
                                },
                            ]
                        },
                    },
                    {
                        "name": "Test JWell",
                        "trajectory": [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 10.0],
                            [0.0, 0.0, 20.0],
                            [0.0, 0.0, 50.0],
                            [0.0, 0.0, 60.0],
                            [0.0, 0.0, 70.0],
                            [0.0, 0.0, 80.0],
                            [0.0, 0.0, 100.0],
                        ],
                        "completion": {
                            "perforations": [
                                {
                                    "range": [20.0, 60.0],
                                    "points": [
                                        [0.0, 0.0, 20.0],
                                        [0.0, 0.0, 50.0],
                                        [0.0, 0.0, 60.0],
                                    ],
                                },
                                {
                                    "range": [70.0, 80.0],
                                    "points": [[0.0, 0.0, 70.0], [0.0, 0.0, 80.0]],
                                },
                            ]
                        },
                    },
                ]
            },
            {
                "Test IWell": (
                    [0.0, 0.0, 10.0],
                    [0.0, 0.0, 50.0],
                    [0.0, 0.0, 60.0],
                    [0.0, 0.0, 70.0],
                    [0.0, 0.0, 80.0],
                ),
                "Test JWell": (
                    [0.0, 0.0, 20.0],
                    [0.0, 0.0, 50.0],
                    [0.0, 0.0, 60.0],
                    [0.0, 0.0, 70.0],
                    [0.0, 0.0, 80.0],
                ),
            },
        ),
    ],
)
def test_extract_well_perforations_points(
    wells_result: WellManagementServiceResultSchema,
    expected_output: dict[WellName, tuple[Point, ...]],
) -> None:
    actual_result = extract_well_with_perforations_points(wells_result)
    assert all((isinstance(k, str) for k in actual_result.keys()))
    assert all((isinstance(v, tuple) for v in actual_result.values()))
    assert actual_result == expected_output
