from typing import Any

import pytest
from pydantic import ValidationError

from services.well_management_service import WellManagementService
from services.well_management_service.core.models import (
    HWellModel,
    IWellModel,
    JWellModel,
    PerforationRangeModel,
    SimulationWellModel,
    SWellModel,
)
from tests.well_management_service_tests.tools import (
    is_subsequence_tuple_of_float,
)

# Define sample data for testing
valid_position: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
valid_perforation: dict[str, float] = {"start_md": 100.0, "end_md": 200.0}
invalid_perforation: dict[str, float] = {"start_md": 300.0, "end_md": 200.0}
unsorted_perforations: list[dict[str, float]] = [
    {"start_md": 300.0, "end_md": 400.0},
    {"start_md": 100.0, "end_md": 200.0},
]
sorted_perforations: list[dict[str, float]] = [
    {"start_md": 100.0, "end_md": 200.0},
    {"start_md": 300.0, "end_md": 400.0},
]
overlapping_perforations: list[dict[str, float]] = [
    {"start_md": 100.0, "end_md": 200.0},
    {"start_md": 150.0, "end_md": 250.0},
]
out_of_bounds_perforation: dict[str, float] = {"start_md": 3000.0, "end_md": 4200.0}


@pytest.mark.parametrize(
    "model_class, input_data, expected_exception, expected_sorted_perforations",
    [
        (
            IWellModel,
            {
                "name": "Test IWell",
                "md": 1500.0,
                "wellhead": valid_position,
                "md_step": 10.0,
                "perforations": unsorted_perforations,
            },
            None,
            sorted_perforations,
        ),
        (
            IWellModel,
            {
                "name": "Test IWell",
                "md": 1500.0,
                "wellhead": valid_position,
                "md_step": 10.0,
                "perforations": invalid_perforation,
            },
            ValidationError,
            None,
        ),
        (
            IWellModel,
            {
                "well_type": "IWell",
                "name": "Test IWell2",
                "md": 1500.0,
                "wellhead": valid_position,
                "md_step": 10.0,
                "perforations": None,
            },
            None,
            None,
        ),
        (
            IWellModel,
            {
                "well_type": "IWell",
                "name": "Test IWell3",
                "md": 1500.0,
                "wellhead": valid_position,
                "md_step": 10.0,
                "perforations": overlapping_perforations,
            },
            ValidationError,
            None,
        ),
        (
            IWellModel,
            {
                "well_type": "IWell",
                "name": "Test IWell4",
                "md": 500.0,
                "wellhead": valid_position,
                "md_step": 10.0,
                "perforations": [out_of_bounds_perforation],
            },
            None,
            None,
        ),
        (
            JWellModel,
            {
                "well_type": "JWell",
                "name": "Test JWell",
                "md_linear1": 500.0,
                "md_curved": 300.0,
                "dls": 15.0,
                "md_linear2": 700.0,
                "wellhead": valid_position,
                "azimuth": 90.0,
                "md_step": 20.0,
                "perforations": unsorted_perforations,
            },
            None,
            sorted_perforations,
        ),
        (
            JWellModel,
            {
                "name": "Test JWell1",
                "md_linear1": 500.0,
                "md_curved": 300.0,
                "dls": 15.0,
                "md_linear2": 700.0,
                "wellhead": valid_position,
                "azimuth": 90.0,
                "md_step": 20.0,
                "perforations": None,
            },
            None,
            None,
        ),
        (
            JWellModel,
            {
                "name": "Test JWell2",
                "md_linear1": 500.0,
                "md_curved": 300.0,
                "dls": 15.0,
                "md_linear2": 700.0,
                "wellhead": valid_position,
                "azimuth": 90.0,
                "md_step": 20.0,
                "perforations": invalid_perforation,
            },
            ValidationError,
            None,
        ),
        (
            JWellModel,
            {
                "well_type": "JWell",
                "name": "Test JWell3",
                "md_linear1": 500.0,
                "md_curved": 300.0,
                "dls": 15.0,
                "md_linear2": 700.0,
                "wellhead": valid_position,
                "azimuth": 90.0,
                "md_step": 20.0,
                "perforations": overlapping_perforations,
            },
            ValidationError,
            None,
        ),
        (
            JWellModel,
            {
                "well_type": "JWell",
                "name": "Test JWell4",
                "md_linear1": 500.0,
                "md_curved": 300.0,
                "dls": 15.0,
                "md_linear2": 700.0,
                "wellhead": valid_position,
                "azimuth": 90.0,
                "md_step": 20.0,
                "perforations": [out_of_bounds_perforation],
            },
            None,
            None,
        ),
        (
            SWellModel,
            {
                "well_type": "SWell",
                "name": "Test SWell",
                "md_linear1": 400.0,
                "md_curved1": 200.0,
                "dls1": 10.0,
                "md_linear2": 500.0,
                "md_curved2": 300.0,
                "dls2": 20.0,
                "md_linear3": 600.0,
                "wellhead": valid_position,
                "azimuth": 180.0,
                "md_step": 30.0,
                "perforations": unsorted_perforations,
            },
            None,
            sorted_perforations,
        ),
        (
            SWellModel,
            {
                "well_type": "SWell",
                "name": "Test SWell1",
                "md_linear1": 400.0,
                "md_curved1": 200.0,
                "dls1": 10.0,
                "md_linear2": 500.0,
                "md_curved2": 300.0,
                "dls2": 20.0,
                "md_linear3": 600.0,
                "wellhead": valid_position,
                "azimuth": 180.0,
                "md_step": 30.0,
                "perforations": invalid_perforation,
            },
            ValidationError,
            None,
        ),
        (
            SWellModel,
            {
                "well_type": "SWell",
                "name": "Test SWell2",
                "md_linear1": 400.0,
                "md_curved1": 200.0,
                "dls1": 10.0,
                "md_linear2": 500.0,
                "md_curved2": 300.0,
                "dls2": 20.0,
                "md_linear3": 600.0,
                "wellhead": valid_position,
                "azimuth": 180.0,
                "md_step": 30.0,
                "perforations": overlapping_perforations,
            },
            ValidationError,
            None,
        ),
        (
            SWellModel,
            {
                "well_type": "SWell",
                "name": "Test SWell3",
                "md_linear1": 400.0,
                "md_curved1": 200.0,
                "dls1": 10.0,
                "md_linear2": 500.0,
                "md_curved2": 300.0,
                "dls2": 20.0,
                "md_linear3": 600.0,
                "wellhead": valid_position,
                "azimuth": 180.0,
                "md_step": 30.0,
                "perforations": [out_of_bounds_perforation],
            },
            None,
            None,
        ),
        (
            SWellModel,
            {
                "well_type": "SWell",
                "name": "Test SWell4",
                "md_linear1": 400.0,
                "md_curved1": 200.0,
                "dls1": 10.0,
                "md_linear2": 500.0,
                "md_curved2": 300.0,
                "dls2": 20.0,
                "md_linear3": 600.0,
                "wellhead": valid_position,
                "azimuth": 180.0,
                "md_step": 30.0,
                "perforations": None,
            },
            None,
            None,
        ),
        (
            HWellModel,
            {
                "well_type": "HWell",
                "name": "Test HWell",
                "TVD": 1000.0,
                "md_lateral": 1500.0,
                "wellhead": valid_position,
                "azimuth": 45.0,
                "md_step": 10.0,
                "perforations": unsorted_perforations,
            },
            None,
            sorted_perforations,
        ),
        (
            HWellModel,
            {
                "name": "Test HWell1",
                "TVD": 1000.0,
                "md_lateral": 1500.0,
                "wellhead": valid_position,
                "azimuth": 45.0,
                "md_step": 10.0,
                "perforations": invalid_perforation,
            },
            ValidationError,
            None,
        ),
        (
            HWellModel,
            {
                "well_type": "HWell",
                "name": "Test HWell2",
                "TVD": 1000.0,
                "md_lateral": 1500.0,
                "wellhead": valid_position,
                "azimuth": 45.0,
                "md_step": 10.0,
                "perforations": overlapping_perforations,
            },
            ValidationError,
            None,
        ),
        (
            HWellModel,
            {
                "well_type": "HWell",
                "name": "Test HWell3",
                "TVD": 100.0,
                "md_lateral": 150.0,
                "wellhead": valid_position,
                "azimuth": 45.0,
                "md_step": 10.0,
                "perforations": [valid_perforation],
            },
            ValidationError,
            None,
        ),
    ],
)
def test_well_models(
    model_class: Any,
    input_data: dict[str, Any],
    expected_exception: type[BaseException] | None,
    expected_sorted_perforations: list[dict[str, float]] | None,
) -> None:
    if expected_exception:
        with pytest.raises(expected_exception):
            model_class(**input_data)
    else:
        instance = model_class(**input_data)
        assert instance.name == input_data["name"]
        if expected_sorted_perforations:
            assert [
                {"start_md": p.start_md, "end_md": p.end_md}
                for p in instance.perforations
            ] == expected_sorted_perforations


@pytest.mark.parametrize(
    "start_md, end_md, expected_exception", [(1, 5, None), (5, 1, ValidationError)]
)
def test_perforation_range(
    start_md: float, end_md: float, expected_exception: type[BaseException] | None
) -> None:
    if expected_exception:
        with pytest.raises(expected_exception):
            PerforationRangeModel(start_md=start_md, end_md=end_md)
    else:
        pr = PerforationRangeModel(start_md=start_md, end_md=end_md)
        assert pr.start_md == start_md
        assert pr.end_md == end_md


@pytest.mark.parametrize(
    "input_data, expected_output, expected_exception",
    [
        (
            {
                "models": [
                    {
                        "well_type": "IWell",
                        "name": "Test IWell",
                        "md": 100,
                        "wellhead": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "md_step": 50.0,
                        "perforations": [
                            {"start_md": 10.0, "end_md": 60.0},
                            {"start_md": 70.0, "end_md": 80.0},
                        ],
                    },
                ]
            },
            {
                "results": [
                    {
                        "name": "Test IWell",
                        "expected_points_in_order": (
                            (0, 0, 0),
                            (0, 0, 10),
                            (0, 0, 50),
                            (0, 0, 60),
                            (0, 0, 100),
                        ),
                        "expected_perforations_points": [
                            ((0, 0, 10), (0, 0, 50), (0, 0, 60)),
                            ((0, 0, 70), (0, 0, 80)),
                        ],
                    },
                ]
            },
            None,
        ),
        (
            {
                "models": [
                    {
                        "well_type": "JWell",
                        "name": "Test JWell",
                        "md_linear1": 10.0,
                        "md_curved": 300,
                        "dls": 9.0,
                        "md_linear2": 100,
                        "wellhead": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "azimuth": 0,
                        "md_step": 10,
                        "perforations": None,
                    },
                ]
            },
            {
                "results": [
                    {
                        "name": "Test JWell",
                        "expected_points_in_order": (
                            (0, 0, 0),
                            (0, 0, 10),
                            (0.261739582, 0, 19.99543137),
                            (190.9859317, 0, 200.9859317),
                        ),
                        "expected_perforations_points": None,
                    },
                ]
            },
            None,
        ),
        (
            {
                "models": [
                    {
                        "well_type": "JWell",
                        "name": "Test JWell",
                        "md_linear1": 10.0,
                        "md_curved": 300,
                        "dls": 9.0,
                        "md_linear2": 100,
                        "wellhead": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "azimuth": 0,
                        "md_step": 10,
                        "perforations": None,
                    },
                    {
                        "well_type": "SWell",
                        "name": "Test SWell",
                        "md_linear1": 100.0,
                        "md_curved1": 300.0,
                        "dls1": 9.0,
                        "md_linear2": 100,
                        "md_curved2": 300.0,
                        "dls2": -9.0,
                        "md_linear3": 100,
                        "wellhead": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "azimuth": 0.0,
                        "md_step": 10,
                        "perforations": [{"start_md": 10, "end_md": 30}],
                    },
                ]
            },
            {
                "results": [
                    {
                        "name": "Test JWell",
                        "expected_points_in_order": (
                            (0, 0, 0),
                            (0, 0, 10),
                            (0.261739582, 0, 19.99543137),
                            (190.9859317, 0, 200.9859317),
                        ),
                        "expected_perforations_points": None,
                    },
                    {
                        "name": "Test SWell",
                        "expected_points_in_order": (
                            (0, 0, 0),
                            (0, 0, 10),
                            (0.261739582, 0, 109.99543137),
                            (190.9859317, 0, 290.9859317),
                        ),
                        "expected_perforations_points": [((0, 0, 10), (0, 0, 30))],
                    },
                ]
            },
            None,
        ),
        (
            {
                "models": [
                    {
                        "well_type": "IWell",
                        "name": "Test IWell",
                        "md": 100,
                        "wellhead": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "md_step": 50.0,
                        "perforations": [
                            {"start_md": 10.0, "end_md": 60.0},
                            {"start_md": 70.0, "end_md": 80.0},
                        ],
                    },
                    {
                        "well_type": "IWell",
                        "name": "Test IWell",
                        "md": 100,
                        "wellhead": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "md_step": 50.0,
                        "perforations": [
                            {"start_md": 10.0, "end_md": 60.0},
                            {"start_md": 70.0, "end_md": 80.0},
                        ],
                    },
                ]
            },
            {},
            ValidationError,
        ),
        (
            {
                "models": [
                    {
                        "well_type": "HWell",
                        "name": "Test HWell",
                        "TVD": 1000.0,
                        "md_lateral": 1500.0,
                        "wellhead": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "azimuth": 0.0,
                        "md_step": 1,
                        "perforations": [
                            {"start_md": 100.0, "end_md": 200.0},
                            {"start_md": 300.0, "end_md": 400.0},
                        ],
                    },
                ]
            },
            {
                "results": [
                    {
                        "name": "Test HWell",
                        "expected_points_in_order": (
                            (0, 0, 0),
                            (0, 0, 10),
                            (0, 0, 100),
                            (0, 0, 200),
                            (0, 0, 300),
                            (0, 0, 400),
                            (0, 0, 500),
                        ),
                        "expected_perforations_points": [
                            ((0, 0, 100), (0, 0, 200)),
                            ((0, 0, 300), (0, 0, 400)),
                        ],
                    },
                ]
            },
            None,
        ),
        (
            {
                "models": [
                    {
                        "well_type": "HWell",
                        "name": "Test HWell",
                        "TVD": 100.0,
                        "md_lateral": 150.0,
                        "wellhead": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "azimuth": 45.0,
                        "md_step": 10.0,
                        "perforations": [valid_perforation],
                    },
                ]
            },
            {},
            ValidationError,
        ),
    ],
)
def test_well_management_service(
    input_data: dict[str, Any],
    expected_output: dict[str, Any],
    expected_exception: type[BaseException] | None,
) -> None:
    if expected_exception:
        with pytest.raises(expected_exception):
            WellManagementService.process_request(input_data)
    else:
        models = WellManagementService.process_request(input_data)
        assert all(isinstance(m, SimulationWellModel) for m in models.wells)
        for m, r in zip(models.wells, expected_output["results"]):
            assert m is not None
            assert m.name == r["name"]
            assert is_subsequence_tuple_of_float(
                m.trajectory, r["expected_points_in_order"]
            )
            if m.completion is not None:
                for cm, ce in zip(
                    m.completion.perforations, r["expected_perforations_points"]
                ):
                    assert is_subsequence_tuple_of_float(cm.points, ce)
