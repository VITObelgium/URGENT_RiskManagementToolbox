from services.problem_dispatcher_service.core.utils import (
    CandidateGenerator,
    get_corresponding_initial_state_as_flat_dict,
    parse_flat_dict_to_nested,
)


def test_parse_flat_dict_to_nested():
    flat = {
        "a#b#c": 1,
        "x#y": 2,
        "z": 3,
    }
    nested = parse_flat_dict_to_nested(flat)
    assert nested == {
        "a": {"b": {"c": 1}},
        "x": {"y": 2},
        "z": 3,
    }


def test_get_corresponding_initial_state_as_flat_dict():
    initial = {
        "well_design": {
            "INJ": {
                "md": 3000.0,
                "md_step": 5.0,
                "name": "INJ",
                "perforations": [{"end_md": 3000.0, "start_md": 0.0}],
                "well_type": "IWell",
                "wellhead": {"x": 400.0, "y": 400.0, "z": 0.0},
            },
            "PRO": {
                "md": 3000.0,
                "md_step": 5.0,
                "name": "PRO",
                "perforations": [{"end_md": 3000.0, "start_md": 0.0}],
                "well_type": "IWell",
                "wellhead": {"x": 700.0, "y": 700.0, "z": 0.0},
            },
        }
    }

    variable_source = ["well_design#INJ#md", "well_design#PRO#wellhead#x"]

    results = get_corresponding_initial_state_as_flat_dict(initial, variable_source)
    assert results == {
        "well_design#INJ#md": 3000.0,
        "well_design#PRO#wellhead#x": 700.0,
    }


def test_candidate_generator():
    constraints = {"a": (0.0, 1.0), "b": (10.0, 20.0)}
    initial_state = {"a": 0.5, "b": 15.0}
    candidates = CandidateGenerator.generate(
        constraints,
        5,
        random_fn=lambda lb, ub: (lb + ub) / 2,
        initial_state=initial_state,
    )
    assert len(candidates) == 5
    for c in candidates:
        assert c["a"] == 0.5
        assert c["b"] == 15.0
