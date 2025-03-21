from services.problem_dispatcher_service.core.utils import (
    CandidateGenerator,
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


def test_candidate_generator():
    constraints = {"a": (0.0, 1.0), "b": (10.0, 20.0)}
    candidates = CandidateGenerator.generate(
        constraints, 5, random_fn=lambda lb, ub: (lb + ub) / 2
    )
    assert len(candidates) == 5
    for c in candidates:
        assert c["a"] == 0.5
        assert c["b"] == 15.0
