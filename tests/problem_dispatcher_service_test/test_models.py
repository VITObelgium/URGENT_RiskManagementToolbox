import math

import psutil
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


# Tests for default values
def test_optimization_parameters_defaults():
    params = OptimizationParameters()
    assert params.max_generations == 10
    assert params.population_size == 10
    assert params.patience == 10
    assert params.worker_count == 4


# Tests for positive integer validation
def test_max_generations_must_be_positive():
    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(max_generations=0)

    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(max_generations=-5)


def test_population_size_must_be_positive():
    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(population_size=0)

    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(population_size=-10)


def test_patience_must_be_positive():
    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(patience=0)

    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(patience=-3)


def test_worker_count_must_be_positive():
    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(worker_count=0)

    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(worker_count=-2)


# Tests for worker_count physical core validation
def test_worker_count_exceeds_physical_cores():
    physical_cores = psutil.cpu_count(logical=False)
    max_allowed = max(1, math.floor(physical_cores / 2))

    with pytest.raises(
        ValidationError,
        match=f"worker_count {max_allowed + 1} exceeds available physical cores",
    ):
        OptimizationParameters(worker_count=max_allowed + 1)


def test_worker_count_within_limits():
    physical_cores = psutil.cpu_count(logical=False)
    max_allowed = max(1, math.floor(physical_cores / 2))

    # Should not raise an error
    params = OptimizationParameters(worker_count=max_allowed)
    assert params.worker_count == max_allowed

    # Test with 1 worker (always valid)
    params = OptimizationParameters(worker_count=1)
    assert params.worker_count == 1


# Tests for valid custom values
def test_optimization_parameters_with_valid_custom_values():
    params = OptimizationParameters(
        max_generations=50, population_size=100, patience=15, worker_count=2
    )
    assert params.max_generations == 50
    assert params.population_size == 100
    assert params.patience == 15
    assert params.worker_count == 2
