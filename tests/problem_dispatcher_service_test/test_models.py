import math

import psutil
import pytest
from pydantic import ValidationError

from services.problem_dispatcher_service.core.models.user import OptimizationParameters


@pytest.fixture
def valid_linear_inequalities():
    return {
        "A": [{"well_design.INJ.md": 1.0, "well_design.PRO.md": 1.0}],
        "b": [3000.0],
        "sense": ["<="],
    }


def test_valid_linear_inequalities(valid_linear_inequalities):
    params = OptimizationParameters(
        linear_inequalities=valid_linear_inequalities,
        objectives={"metrics1": "minimize"},
    )
    assert params.linear_inequalities == valid_linear_inequalities


def test_linear_inequalities_missing_b():
    with pytest.raises(KeyError, match="must contain both 'A' and 'b'"):
        OptimizationParameters(
            linear_inequalities={"A": [{"INJ.md": 1.0}]},
            objectives={"metrics1": "minimize"},
        )


def test_linear_inequalities_A_not_list():
    with pytest.raises(ValidationError):
        OptimizationParameters(
            linear_inequalities={"A": "not a list", "b": [1]},
            objectives={"metrics1": "minimize"},
        )


def test_linear_inequalities_mismatched_lengths():
    with pytest.raises(
        ValueError, match=r"Number of rows in A \(\d+\) must match length of b \(\d+\)"
    ):
        OptimizationParameters(
            linear_inequalities={"A": [{}, {}], "b": [1]},
            objectives={"metrics1": "minimize"},
        )


def test_linear_inequalities_invalid_sense_type():
    with pytest.raises(ValidationError):
        OptimizationParameters(
            linear_inequalities={"A": [{}], "b": [1], "sense": "not a list"},
            objectives={"metrics1": "minimize"},
        )


def test_linear_inequalities_mismatched_sense_length():
    with pytest.raises(
        ValueError,
        match="Length of 'sense' \(\d+\) must match number of constraints \(\d+\)",
    ):
        OptimizationParameters(
            linear_inequalities={"A": [{}], "b": [1], "sense": ["<=", ">="]},
            objectives={"metrics1": "minimize"},
        )


def test_linear_inequalities_invalid_sense_value():
    with pytest.raises(ValueError, match="Invalid inequality direction"):
        OptimizationParameters(
            linear_inequalities={"A": [{}], "b": [1], "sense": ["invalid"]},
            objectives={"metrics1": "minimize"},
        )


def test_linear_inequalities_default_sense():
    params = OptimizationParameters(
        linear_inequalities={"A": [{"X.md": 1}], "b": [1]},
        objectives={"metrics1": "minimize"},
    )
    assert params.linear_inequalities["sense"] == ["<="]


def test_linear_inequalities_A_row_not_dict():
    with pytest.raises(TypeError, match="Row 0 in A must be a dict"):
        OptimizationParameters(
            linear_inequalities={"A": ["not a dict"], "b": [1]},
            objectives={"metrics1": "minimize"},
        )


def test_linear_inequalities_empty_A_row():
    with pytest.raises(ValueError, match="Row 0 in A is empty"):
        OptimizationParameters(
            linear_inequalities={"A": [{}], "b": [1]},
            objectives={"metrics1": "minimize"},
        )


def test_linear_inequalities_non_numeric_coefficient():
    with pytest.raises(
        TypeError,
        match="Coefficient for 'well_design.X.md' in row 0 must be numeric, got str",
    ):
        OptimizationParameters(
            linear_inequalities={"A": [{"well_design.X.md": "a"}], "b": [1]},
            objectives={"metrics1": "minimize"},
        )


def test_linear_inequalities_variable_no_dot():
    with pytest.raises(
        ValueError, match="must contain a '.' separating well and attribute"
    ):
        OptimizationParameters(
            linear_inequalities={"A": [{"INJ": 1.0}], "b": [1]},
            objectives={"metrics1": "minimize"},
        )


def test_linear_inequalities_non_numeric_b():
    with pytest.raises(TypeError, match="b\\[0\\] must be numeric"):
        OptimizationParameters(
            linear_inequalities={"A": [{"X.md": 1}], "b": ["a"]},
            objectives={"metrics1": "minimize"},
        )


# Tests for default values
def test_optimization_parameters_defaults():
    params = OptimizationParameters(objectives={"metrics1": "minimize"})
    assert params.max_generations == 10
    assert params.population_size == 10
    assert params.patience == 10
    assert params.worker_count == 4


# Tests for positive integer validation
def test_max_generations_must_be_positive():
    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(max_generations=0, objectives={"metrics1": "minimize"})

    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(max_generations=-5, objectives={"metrics1": "minimize"})


def test_population_size_must_be_positive():
    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(population_size=0, objectives={"metrics1": "minimize"})

    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(population_size=-10, objectives={"metrics1": "minimize"})


def test_patience_must_be_positive():
    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(patience=0, objectives={"metrics1": "minimize"})

    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(patience=-3, objectives={"metrics1": "minimize"})


def test_worker_count_must_be_positive():
    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(worker_count=0, objectives={"metrics1": "minimize"})

    with pytest.raises(ValidationError, match="Value must be a positive integer"):
        OptimizationParameters(worker_count=-2, objectives={"metrics1": "minimize"})


# Tests for worker_count physical core validation
def test_worker_count_exceeds_physical_cores():
    physical_cores = psutil.cpu_count(logical=False)
    max_allowed = max(1, math.floor(physical_cores / 2))

    with pytest.raises(
        ValidationError,
        match=f"worker_count {max_allowed + 1} exceeds available physical cores",
    ):
        OptimizationParameters(
            worker_count=max_allowed + 1, objectives={"metrics1": "minimize"}
        )


def test_worker_count_within_limits():
    physical_cores = psutil.cpu_count(logical=False)
    max_allowed = max(1, math.floor(physical_cores / 2))

    # Should not raise an error
    params = OptimizationParameters(
        worker_count=max_allowed, objectives={"metrics1": "minimize"}
    )
    assert params.worker_count == max_allowed

    # Test with 1 worker (always valid)
    params = OptimizationParameters(worker_count=1, objectives={"metrics1": "minimize"})
    assert params.worker_count == 1


# Tests for valid custom values
def test_optimization_parameters_with_valid_custom_values():
    params = OptimizationParameters(
        max_generations=50,
        population_size=100,
        patience=15,
        worker_count=1,
        objectives={"metrics1": "minimize"},
    )
    assert params.max_generations == 50
    assert params.population_size == 100
    assert params.patience == 15
    assert params.worker_count == 1
