import pytest
from pydantic import ValidationError

from services.problem_dispatcher_service.core.models.user import OptimizationParameters


@pytest.fixture
def valid_linear_inequalities():
    return {
        "A": [{"INJ.md": 1.0, "PRO.md": 1.0}],
        "b": [3000.0],
        "sense": ["<="],
    }


def test_valid_linear_inequalities(valid_linear_inequalities):
    params = OptimizationParameters(linear_inequalities=valid_linear_inequalities)
    assert params.linear_inequalities == valid_linear_inequalities


def test_linear_inequalities_missing_b():
    with pytest.raises(KeyError, match="must contain both 'A' and 'b'"):
        OptimizationParameters(linear_inequalities={"A": [{"INJ.md": 1.0}]})


def test_linear_inequalities_A_not_list():
    with pytest.raises(ValidationError):
        OptimizationParameters(linear_inequalities={"A": "not a list", "b": [1]})


def test_linear_inequalities_mismatched_lengths():
    with pytest.raises(ValueError, match="Number of rows in A must match length of b"):
        OptimizationParameters(linear_inequalities={"A": [{}, {}], "b": [1]})


def test_linear_inequalities_invalid_sense_type():
    with pytest.raises(ValidationError):
        OptimizationParameters(
            linear_inequalities={"A": [{}], "b": [1], "sense": "not a list"}
        )


def test_linear_inequalities_mismatched_sense_length():
    with pytest.raises(
        ValueError, match="Length of 'sense' must match number of rows in A"
    ):
        OptimizationParameters(
            linear_inequalities={"A": [{}], "b": [1], "sense": ["<=", ">="]}
        )


def test_linear_inequalities_invalid_sense_value():
    with pytest.raises(ValueError, match="Invalid inequality direction"):
        OptimizationParameters(
            linear_inequalities={"A": [{}], "b": [1], "sense": ["invalid"]}
        )


def test_linear_inequalities_default_sense():
    params = OptimizationParameters(linear_inequalities={"A": [{"X.md": 1}], "b": [1]})
    assert params.linear_inequalities["sense"] == ["<="]


def test_linear_inequalities_A_row_not_dict():
    with pytest.raises(TypeError, match="Row 0 in A must be a dict"):
        OptimizationParameters(linear_inequalities={"A": ["not a dict"], "b": [1]})


def test_linear_inequalities_empty_A_row():
    with pytest.raises(ValueError, match="Row 0 in A is empty"):
        OptimizationParameters(linear_inequalities={"A": [{}], "b": [1]})


def test_linear_inequalities_non_numeric_coefficient():
    with pytest.raises(
        TypeError, match="Coefficient for X.md in row 0 must be numeric"
    ):
        OptimizationParameters(linear_inequalities={"A": [{"X.md": "a"}], "b": [1]})


def test_linear_inequalities_variable_no_dot():
    with pytest.raises(
        ValueError, match="must contain a '.' separating well and attribute"
    ):
        OptimizationParameters(linear_inequalities={"A": [{"INJ": 1.0}], "b": [1]})


def test_linear_inequalities_mixed_attributes():
    with pytest.raises(
        ValueError,
        match="All variables in linear_inequalities must refer to the same attribute",
    ):
        OptimizationParameters(
            linear_inequalities={"A": [{"INJ.md": 1, "PRO.x": 1}], "b": [1]}
        )


def test_linear_inequalities_non_numeric_b():
    with pytest.raises(TypeError, match="b\\[0\\] must be numeric"):
        OptimizationParameters(linear_inequalities={"A": [{"X.md": 1}], "b": ["a"]})
