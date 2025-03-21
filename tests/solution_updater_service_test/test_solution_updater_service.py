import numpy as np
import pytest

from services.solution_updater_service.core.models import (
    OptimizationEngine,
    SolutionUpdaterServiceResponse,
)
from services.solution_updater_service.core.service import (
    SolutionUpdaterService,
)
from services.solution_updater_service.core.utils import get_numpy_values

engine = OptimizationEngine.PSO


@pytest.fixture
def mocked_engine():  # type: ignore
    """Fixture for a mocked OptimizationEngineInterface."""

    class MockedOptimizationEngine:
        def update_solution_to_next_iter(  # type: ignore
            self, parameters, results, lb, ub
        ):
            # Mock behavior: Add 1.0 to each parameter value as a simple transformation
            return parameters + 1.0

    return MockedOptimizationEngine()


@pytest.fixture
def mocked_engine_with_bnb():  # type: ignore
    """Fixture for a mocked OptimizationEngineInterface."""

    class MockedOptimizationEngine:
        def update_solution_to_next_iter(  # type: ignore
            self, parameters, results, lb, ub
        ):
            # Simulate parameter transformation and ensure constraints are applied
            updated_parameters = np.clip(parameters + 1.0, lb, ub)
            return updated_parameters

    return MockedOptimizationEngine()


@pytest.mark.parametrize(
    "config_json, expected_result_parameters",
    [
        # Case 1: Single solution candidate
        (
            {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param1": 1.0, "param2": 2.0}},
                        "cost_function_results": {"values": {"metric1": 10.0}},
                    }
                ],
            },
            [
                {"param1": 2.0, "param2": 3.0}
            ],  # Mocked engine behavior adds 1.0 to each param
        ),
        # Case 2: Multiple solution candidates
        (
            {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param1": 3.0, "param2": 4.0}},
                        "cost_function_results": {"values": {"metric1": 15.0}},
                    },
                    {
                        "control_vector": {"items": {"param1": 5.0, "param2": 6.0}},
                        "cost_function_results": {"values": {"metric1": 20.0}},
                    },
                ],
            },
            [
                {"param1": 4.0, "param2": 5.0},
                {"param1": 6.0, "param2": 7.0},
            ],
        ),
    ],
)
def test_update_solution_for_next_iteration_single_call(  # type: ignore
    config_json, expected_result_parameters, mocked_engine, monkeypatch
):
    # Arrange
    service = SolutionUpdaterService(optimization_engine=engine)

    # Monkeypatch engine
    monkeypatch.setattr(service, "_engine", mocked_engine)

    # Handle "ensure" monkeypatch to bypass type-checking for simplicity

    # Act
    result = service.process_request(config_json)

    # Assert
    assert isinstance(result, SolutionUpdaterServiceResponse)
    assert len(result.next_iter_solutions) == len(expected_result_parameters)
    for actual, expected in zip(result.next_iter_solutions, expected_result_parameters):
        assert actual.items == expected


@pytest.mark.parametrize(
    "config_json_1, config_json_2, expected_result_1, expected_result_2",
    [
        # Test for two consecutive calls
        (
            # First call input
            {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param1": 1.0, "param2": 2.0}},
                        "cost_function_results": {"values": {"metric1": 10.0}},
                    }
                ],
            },
            # Second call input
            {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param2": 3.0, "param1": 2.0}},
                        "cost_function_results": {"values": {"metric1": 15.0}},
                    }
                ],
            },
            # Expected result after first call
            [{"param1": 2.0, "param2": 3.0}],
            # Expected result after second call
            [{"param1": 3.0, "param2": 4.0}],
        ),
    ],
)
def test_update_solution_for_next_iteration_multiple_calls(  # type: ignore
    config_json_1,
    config_json_2,
    expected_result_1,
    expected_result_2,
    mocked_engine,
    monkeypatch,
):
    # Arrange
    service = SolutionUpdaterService(optimization_engine=engine)

    # Monkeypatch engine
    monkeypatch.setattr(service, "_engine", mocked_engine)

    # Act: first call
    result_1 = service.process_request(config_json_1)
    # Act: second call
    result_2 = service.process_request(config_json_2)

    # Assert first call
    assert isinstance(result_1, SolutionUpdaterServiceResponse)
    assert len(result_1.next_iter_solutions) == len(expected_result_1)
    for actual, expected in zip(result_1.next_iter_solutions, expected_result_1):
        assert actual.items == expected

    # Assert second call
    assert isinstance(result_2, SolutionUpdaterServiceResponse)
    assert len(result_2.next_iter_solutions) == len(expected_result_2)
    for actual, expected in zip(result_2.next_iter_solutions, expected_result_2):
        assert actual.items == expected


@pytest.mark.parametrize(
    "config_json, expected_result_parameters",
    [
        # Case 1: Parameters remain within boundaries
        (
            {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param1": 1.0, "param2": 2.0}},
                        "cost_function_results": {"values": {"metric1": 10.0}},
                    }
                ],
                "optimization_constraints": {
                    "boundaries": {
                        "param1": [0.0, 5.0],
                        "param2": [0.0, 3.0],
                    }
                },
            },
            np.array(
                [[2.0, 3.0]]
            ),  # Mocked behavior adds 1.0 while staying within bounds
        ),
        # Case 2: Parameters exceed boundaries and should be capped
        (
            {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param1": 4.5, "param2": 2.2}},
                        "cost_function_results": {"values": {"metric1": 15.0}},
                    }
                ],
                "optimization_constraints": {
                    "boundaries": {
                        "param1": [0.0, 4.0],
                        "param2": [0.0, 3.0],
                    }
                },
            },
            np.array([[4.0, 3.0]]),  # Results are capped to the upper bounds
        ),
    ],
)
def test_update_solution_with_boundaries_np(  # type: ignore
    config_json, expected_result_parameters, mocked_engine_with_bnb, monkeypatch
):
    # Arrange
    service = SolutionUpdaterService(optimization_engine=engine)

    # Monkeypatch engine to use mocked behavior
    monkeypatch.setattr(service, "_engine", mocked_engine_with_bnb)

    # Act
    result = service.process_request(config_json)

    # Assert
    assert isinstance(result, SolutionUpdaterServiceResponse)
    assert len(result.next_iter_solutions) == expected_result_parameters.shape[0]
    # Validate the resulting parameter values with NumPy array comparisons
    for actual, expected in zip(result.next_iter_solutions, expected_result_parameters):
        np.testing.assert_almost_equal(
            np.array(list(actual.items.values())), expected, decimal=6
        )


def rosenbrock_function(x: np.ndarray) -> float:
    """Rosenbrock function for minimization. Global minimum at [1, ..., 1]."""
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def sphere_function(x: np.ndarray) -> float:
    """Sphere function for minimization. Global minimum at [0, ..., 0]."""
    return np.sum(x**2)


def rastrigin_function(x: np.ndarray) -> float:
    """Rastrigin function for minimization. Global minimum at [0, ..., 0]."""
    return 10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def evaluate_function(particles: np.ndarray, function) -> np.ndarray:
    """Evaluate the benchmark function for the given positions of particles."""
    return np.apply_along_axis(function, 1, particles)


def create_candidates(
    positions: np.ndarray, results: np.ndarray, param_names: list
) -> list:
    """Generate candidate solutions based on positions and function results."""
    return [
        {
            "control_vector": {"items": dict(zip(param_names, pos))},
            "cost_function_results": {"values": {"metric1": res}},
        }
        for pos, res in zip(positions, results)
    ]


test_cases = [
    {
        "function": rosenbrock_function,
        "dimensions": 2,
        "bounds": (-5, 5),
        "expected_minimum": [1.0, 1.0],
        "tolerance": 1e-2,
    },
    {
        "function": sphere_function,
        "dimensions": 3,
        "bounds": (-10, 10),
        "expected_minimum": [0.0, 0.0, 0.0],
        "tolerance": 1e-4,
    },
    {
        "function": rastrigin_function,
        "dimensions": 5,
        "bounds": (-5.12, 5.12),
        "expected_minimum": [0.0, 0.0, 0.0, 0.0, 0.0],
        "tolerance": 1e-2,
    },
]


@pytest.mark.parametrize("test_case", test_cases)
def test_optimization_service_full_round(test_case):
    function = test_case["function"]
    dim = test_case["dimensions"]
    bounds = np.array(test_case["bounds"])
    expected_min = test_case["expected_minimum"]
    tol = test_case["tolerance"]

    lb, ub = bounds[0], bounds[1]
    param_names = [f"x{i}" for i in range(dim)]
    num_particles = 50
    iterations = 1000

    np.random.seed(42)

    # Initialize particle positions
    positions = np.random.uniform(lb, ub, (num_particles, dim))
    results = evaluate_function(positions, function)

    candidates = create_candidates(positions, results, param_names)

    config = {
        "solution_candidates": candidates,
        "optimization_constraints": {"boundaries": {k: [lb, ub] for k in param_names}},
    }

    service = SolutionUpdaterService(optimization_engine=engine)

    for _ in range(iterations):
        result = service.process_request(config)
        positions = np.array(
            [get_numpy_values(vec.items) for vec in result.next_iter_solutions]
        )
        results = evaluate_function(positions, function)
        config["solution_candidates"] = create_candidates(
            positions, results, param_names
        )

    best_particle = positions[np.argmin(results)]
    best_result = np.min(results)

    assert np.allclose(best_particle, expected_min, atol=tol), (
        f"PSO failed to converge to the expected minimum. "
        f"Best found: {best_particle}, Expected: {expected_min}"
    )

    assert (
        best_result <= tol
    ), f"Best result {best_result} is not close enough to the global minimum value."
