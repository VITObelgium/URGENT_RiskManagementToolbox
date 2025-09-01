import numpy as np
import numpy.typing as npt
import pytest

from common import OptimizationStrategy
from services.solution_updater_service.core.engines import SolutionMetrics
from services.solution_updater_service.core.models import (
    OptimizationEngine,
    SolutionUpdaterServiceResponse,
)
from services.solution_updater_service.core.service import (
    SolutionUpdaterService,
)
from services.solution_updater_service.core.utils import (
    ensure_not_none,
    get_numpy_values,
)

engine = OptimizationEngine.PSO


@pytest.fixture
def mocked_engine():  # type: ignore
    """Fixture for a mocked OptimizationEngineInterface."""

    class MockedOptimizationEngine:
        def __init__(self):
            constant_best = 0
            self.global_best_result = constant_best

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


@pytest.fixture
def mocked_engine_with_2_stagnant_result():  # type: ignore
    """Fixture for a mocked OptimizationEngineInterface."""

    class MockedOptimizationEngine:
        def __init__(self):
            self.global_best_result = 1000
            self._iter = 1
            self.stagnation_break = 3

        def update_solution_to_next_iter(  # type: ignore
            self, parameters, results, lb, ub
        ):
            if self._iter % self.stagnation_break == 0:
                self.global_best_result -= 1

            self._iter += 1
            return parameters + 1

    return MockedOptimizationEngine()


@pytest.fixture
def mocked_engine_with_metrics():  # type: ignore
    """Fixture for a mocked OptimizationEngineInterface that tracks metrics."""

    class MockedOptimizationEngine:
        def __init__(self) -> None:
            self._metrics: SolutionMetrics | None = None
            self._results_history: list[npt.NDArray[np.float64]] = []

        def update_solution_to_next_iter(
            self,
            parameters: npt.NDArray[np.float64],
            results: npt.NDArray[np.float64],
            lb: npt.NDArray[np.float64],
            ub: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            self._results_history.append(results)
            self._update_metrics(results)
            return parameters + 1.0

        def _update_metrics(self, new_results: npt.NDArray[np.float64]) -> None:
            population_min = float(new_results.min())
            population_max = float(new_results.max())
            population_avg = float(np.average(new_results))
            population_std = float(np.std(new_results))

            if self._metrics is None:  # first run
                global_min = population_min
            else:
                global_min = min(population_min, self._metrics.global_min)

            self._metrics = SolutionMetrics(
                global_min=global_min,
                last_population_min=population_min,
                last_population_max=population_max,
                last_population_avg=population_avg,
                last_population_std=population_std,
            )

        @property
        def metrics(self) -> SolutionMetrics:
            return ensure_not_none(self._metrics)

        @property
        def global_best_result(self) -> float:
            return self.metrics.global_min

        @property
        def global_best_controll_vector(self) -> npt.NDArray[np.float64]:
            return np.array([0.0])  # Mock value

    return MockedOptimizationEngine()


@pytest.mark.parametrize(
    "config_json, expected_result_parameters, optimization_strategy",
    [
        # Case 1: Single solution candidate, MINIMIZE strategy (default)
        (
            {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param1": 1.0, "param2": 2.0}},
                        "cost_function_results": {"values": {"metric1": 10.0}},
                    }
                ],
            },
            [{"param1": 2.0, "param2": 3.0}],
            OptimizationStrategy.MINIMIZE,
        ),
        # Case 2: Multiple solution candidates, MAXIMIZE strategy
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
            OptimizationStrategy.MAXIMIZE,
        ),
    ],
)
def test_update_solution_for_next_iteration_with_strategy(  # type: ignore
    config_json,
    expected_result_parameters,
    optimization_strategy,
    mocked_engine,
    monkeypatch,
):
    # Arrange
    service = SolutionUpdaterService(
        optimization_engine=engine,
        max_generations=100,
        patience=101,
        optimization_strategy=optimization_strategy,
    )

    # Monkeypatch engine
    monkeypatch.setattr(service, "_engine", mocked_engine)

    # Act
    result = service.process_request(config_json)

    # Assert
    assert isinstance(result, SolutionUpdaterServiceResponse)
    assert len(result.next_iter_solutions) == len(expected_result_parameters)
    for actual, expected in zip(result.next_iter_solutions, expected_result_parameters):
        assert actual.items == expected


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
    service = SolutionUpdaterService(
        optimization_engine=engine, max_generations=100, patience=101
    )

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
    service = SolutionUpdaterService(
        optimization_engine=engine, max_generations=100, patience=101
    )

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
    service = SolutionUpdaterService(
        optimization_engine=engine, max_generations=100, patience=101
    )

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
    patience = 1001  # infinite patience

    np.random.seed(42)

    # Initialize particle positions
    positions = np.random.uniform(lb, ub, (num_particles, dim))
    results = evaluate_function(positions, function)

    candidates = create_candidates(positions, results, param_names)

    config = {
        "solution_candidates": candidates,
        "optimization_constraints": {"boundaries": {k: [lb, ub] for k in param_names}},
    }

    service = SolutionUpdaterService(
        optimization_engine=engine, max_generations=iterations, patience=patience
    )
    loop_controller = service.loop_controller

    while loop_controller.running():
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

    assert best_result <= tol, (
        f"Best result {best_result} is not close enough to the global minimum value."
    )


@pytest.mark.parametrize(
    "patience, engine_fixture, patience_checks",
    [
        # Case 1: Engine that updates global_best_result every 3 iterations
        (
            2,
            "mocked_engine_with_2_stagnant_result",
            [
                {"generation": 1, "running": True, "patience_left": 2},
                {"generation": 2, "running": True, "patience_left": 1},
                {
                    "generation": 3,
                    "running": True,
                    "patience_left": 2,
                },  # Reset due to improvement
            ],
        ),
        # Case 2: Engine with constant result (no improvement)
        (
            1,
            "mocked_engine",
            [
                {"generation": 1, "running": True, "patience_left": 1},
                {"generation": 2, "running": True, "patience_left": 0},
                {
                    "generation": 3,
                    "running": False,
                    "patience_left": -1,
                },  # Should stop here
            ],
        ),
    ],
)
def test_patience_handling(
    patience, engine_fixture, patience_checks, request, monkeypatch
):
    # Arrange
    service = SolutionUpdaterService(
        optimization_engine=engine, max_generations=10, patience=patience
    )

    # Get the engine fixture based on the parameter
    mocked_engine = request.getfixturevalue(engine_fixture)
    monkeypatch.setattr(service, "_engine", mocked_engine)

    # Test data
    config_json = {
        "solution_candidates": [
            {
                "control_vector": {"items": {"param1": 1.0, "param2": 2.0}},
                "cost_function_results": {"values": {"metric1": 10.0}},
            }
        ],
    }

    loop_controller = service.loop_controller

    # Initial check
    assert loop_controller.running() is True, (
        "Loop controller should be running after initialization"
    )

    # Run through the patience checks
    for i, check in enumerate(patience_checks, 1):
        service.process_request(config_json)

        expected_generation = check["generation"]
        expected_running = check["running"]
        expected_patience = check["patience_left"]

        assert loop_controller.current_generation == expected_generation, (
            f"Iteration {i}: wrong generation count"
        )
        assert loop_controller.running() is expected_running, (
            f"Iteration {i}: wrong running state - expected {expected_running}"
        )
        assert loop_controller._patience_left == expected_patience, (
            f"Iteration {i}: wrong patience value - expected {expected_patience}, got {loop_controller._patience_left}"
        )


def test_solution_metrics_calculation(mocked_engine_with_metrics, monkeypatch):
    """Test that SolutionMetrics are properly calculated and updated.

    This test verifies that:
    1. Metrics are properly initialized on first run
    2. Global minimum is correctly updated when better values are found
    3. Global minimum is preserved when no better values are found
    4. All population statistics are correctly calculated
    5. The metrics object is properly updated after each iteration
    """
    # Arrange
    service = SolutionUpdaterService(
        optimization_engine=engine, max_generations=3, patience=4
    )
    monkeypatch.setattr(service, "_engine", mocked_engine_with_metrics)

    # Test data with known values for easy verification
    test_cases = [
        # First population: [1.0, 2.0, 3.0]
        {
            "request": {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param1": 1.0}},
                        "cost_function_results": {"values": {"metric1": 1.0}},
                    },
                    {
                        "control_vector": {"items": {"param1": 2.0}},
                        "cost_function_results": {"values": {"metric1": 2.0}},
                    },
                    {
                        "control_vector": {"items": {"param1": 3.0}},
                        "cost_function_results": {"values": {"metric1": 3.0}},
                    },
                ],
            },
            "expected_metrics": {
                "global_min": 1.0,
                "last_population_min": 1.0,
                "last_population_max": 3.0,
                "last_population_avg": 2.0,
                "last_population_std": 0.816496580927726,  # sqrt(2/3)
            },
        },
        # Second population: [0.5, 1.5, 2.5] - new global min
        {
            "request": {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param1": 0.5}},
                        "cost_function_results": {"values": {"metric1": 0.5}},
                    },
                    {
                        "control_vector": {"items": {"param1": 1.5}},
                        "cost_function_results": {"values": {"metric1": 1.5}},
                    },
                    {
                        "control_vector": {"items": {"param1": 2.5}},
                        "cost_function_results": {"values": {"metric1": 2.5}},
                    },
                ],
            },
            "expected_metrics": {
                "global_min": 0.5,  # Updated global min
                "last_population_min": 0.5,
                "last_population_max": 2.5,
                "last_population_avg": 1.5,
                "last_population_std": 0.816496580927726,  # sqrt(2/3)
            },
        },
        # Third population: [2.0, 3.0, 4.0] - no new global min
        {
            "request": {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param1": 2.0}},
                        "cost_function_results": {"values": {"metric1": 2.0}},
                    },
                    {
                        "control_vector": {"items": {"param1": 3.0}},
                        "cost_function_results": {"values": {"metric1": 3.0}},
                    },
                    {
                        "control_vector": {"items": {"param1": 4.0}},
                        "cost_function_results": {"values": {"metric1": 4.0}},
                    },
                ],
            },
            "expected_metrics": {
                "global_min": 0.5,  # Keeps previous global min
                "last_population_min": 2.0,
                "last_population_max": 4.0,
                "last_population_avg": 3.0,
                "last_population_std": 0.816496580927726,  # sqrt(2/3)
            },
        },
    ]

    # Act & Assert
    for i, test_case in enumerate(test_cases, 1):
        # Process the request
        service.process_request(test_case["request"])

        # Get the current metrics
        metrics = service.get_optimization_metrics()
        expected = test_case["expected_metrics"]

        # Verify all metric values
        assert metrics.global_min == expected["global_min"], (
            f"Population {i}: Wrong global minimum"
        )
        assert metrics.last_population_min == expected["last_population_min"], (
            f"Population {i}: Wrong population minimum"
        )
        assert metrics.last_population_max == expected["last_population_max"], (
            f"Population {i}: Wrong population maximum"
        )
        assert np.isclose(
            metrics.last_population_avg, expected["last_population_avg"]
        ), f"Population {i}: Wrong population average"
        assert np.isclose(
            metrics.last_population_std, expected["last_population_std"]
        ), f"Population {i}: Wrong population standard deviation"


def inverted_sphere_function(x: np.ndarray) -> float:
    """Inverted Sphere function for maximization. Global maximum at [0, ..., 0] with value 10."""
    return 10 - np.sum(x**2)


maximization_test_cases = [
    {
        "function": inverted_sphere_function,
        "dimensions": 3,
        "bounds": (-10, 10),
        "expected_maximum": [0.0, 0.0, 0.0],
        "expected_max_value": 10.0,
        "tolerance": 1e-2,
    },
]


@pytest.mark.parametrize("test_case", maximization_test_cases)
def test_maximization_service_full_round(test_case):
    function = test_case["function"]
    dim = test_case["dimensions"]
    bounds = np.array(test_case["bounds"])
    expected_max = test_case["expected_maximum"]
    expected_max_val = test_case["expected_max_value"]
    tol = test_case["tolerance"]

    lb, ub = bounds[0], bounds[1]
    param_names = [f"x{i}" for i in range(dim)]
    num_particles = 50
    iterations = 1000
    patience = 1001  # infinite patience

    np.random.seed(42)

    # Initialize particle positions
    positions = np.random.uniform(lb, ub, (num_particles, dim))
    results = evaluate_function(positions, function)

    candidates = create_candidates(positions, results, param_names)

    config = {
        "solution_candidates": candidates,
        "optimization_constraints": {"boundaries": {k: [lb, ub] for k in param_names}},
    }

    service = SolutionUpdaterService(
        optimization_engine=engine,
        max_generations=iterations,
        patience=patience,
        optimization_strategy=OptimizationStrategy.MAXIMIZE,
    )
    loop_controller = service.loop_controller

    while loop_controller.running():
        result = service.process_request(config)
        positions = np.array(
            [get_numpy_values(vec.items) for vec in result.next_iter_solutions]
        )
        results = evaluate_function(positions, function)
        config["solution_candidates"] = create_candidates(
            positions, results, param_names
        )

    best_particle = positions[np.argmax(results)]
    best_result = np.max(results)

    assert np.allclose(best_particle, expected_max, atol=tol), (
        f"PSO failed to converge to the expected maximum. "
        f"Best found: {best_particle}, Expected: {expected_max}"
    )

    assert np.isclose(best_result, expected_max_val, atol=tol), (
        f"Best result {best_result} is not close enough to the global maximum value {expected_max_val}."
    )

    # Verify that the service's reported best result is the negated value of the true maximum
    assert np.isclose(service.global_best_result, -expected_max_val, atol=tol), (
        "Service's global_best_result should be the negated value for maximization."
    )
