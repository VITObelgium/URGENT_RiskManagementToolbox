import numpy as np
import numpy.typing as npt
import pytest

from common import OptimizationStrategy
from services.solution_updater_service.core.engines import GenerationSummary
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
            self, parameters, results, lb, ub, indexed_objectives_strategy, **kwargs
        ):
            # Mock behavior: Add 1.0 to each parameter value as a simple transformation
            return parameters + 1.0

    return MockedOptimizationEngine()


@pytest.fixture
def mocked_engine_with_bnb():  # type: ignore
    """Fixture for a mocked OptimizationEngineInterface."""

    class MockedOptimizationEngine:
        def update_solution_to_next_iter(  # type: ignore
            self, parameters, results, lb, ub, indexed_objectives_strategy, **kwargs
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
            self, parameters, results, lb, ub, indexed_objectives_strategy, **kwargs
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
            self._generation_summary: GenerationSummary | None = None
            self._results_history: list[npt.NDArray[np.float64]] = []

        def update_solution_to_next_iter(
            self,
            parameters: npt.NDArray[np.float64],
            results: npt.NDArray[np.float64],
            lb: npt.NDArray[np.float64],
            ub: npt.NDArray[np.float64],
            indexed_objectives_strategy: dict[int, OptimizationStrategy],
            **kwargs,
        ) -> npt.NDArray[np.float64]:
            self._results_history.append(results)
            self._update_metrics(results)
            return parameters + 1.0

        def _update_metrics(self, new_results: npt.NDArray[np.float64]) -> None:
            population_min = float(new_results.min())
            population_max = float(new_results.max())
            population_avg = float(np.average(new_results))
            population_std = float(np.std(new_results))

            if self._generation_summary is None:  # first run
                global_min = population_min
            else:
                global_min = min(population_min, self._generation_summary.global_best)

            self._generation_summary = GenerationSummary(
                global_best=global_min,
                min=population_min,
                max=population_max,
                avg=population_avg,
                std=population_std,
                population=new_results.flatten().tolist(),
            )

        @property
        def generation_summary(self) -> GenerationSummary:
            return ensure_not_none(self._generation_summary)

        @property
        def global_best_result(self) -> float:
            return self.generation_summary.global_best

        @property
        def global_best_control_vector(self) -> npt.NDArray[np.float64]:
            return np.array([0.0])  # Mock value

    return MockedOptimizationEngine()


@pytest.mark.parametrize(
    "config_json, expected_result_parameters, objectives",
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
            {"metric1": OptimizationStrategy.MINIMIZE},
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
            {"metric1": OptimizationStrategy.MAXIMIZE},
        ),
    ],
)
def test_update_solution_for_next_iteration_with_strategy(  # type: ignore
    config_json,
    expected_result_parameters,
    objectives,
    mocked_engine,
    monkeypatch,
):
    # Arrange
    service = SolutionUpdaterService(
        optimization_engine=engine,
        max_generations=100,
        patience=101,
        objectives=objectives,
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
        optimization_engine=engine,
        max_generations=100,
        patience=101,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
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
        optimization_engine=engine,
        max_generations=100,
        patience=101,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
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
        optimization_engine=engine,
        max_generations=100,
        patience=101,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
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
    num_particles = 200
    iterations = 100
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
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
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
        loop_controller.increment_generation()

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
                {
                    "generation": 2,
                    "running": False,
                    "patience_left": 0,
                },  # Should stop here (patience <= 0)
            ],
        ),
    ],
)
def test_patience_handling(
    patience, engine_fixture, patience_checks, request, monkeypatch
):
    # Arrange
    service = SolutionUpdaterService(
        optimization_engine=engine,
        max_generations=10,
        patience=patience,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
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
        loop_controller.increment_generation()

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
    """Test that SolutionMetrics are properly calculated and updated."""
    # Arrange
    service = SolutionUpdaterService(
        optimization_engine=engine,
        max_generations=3,
        patience=4,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
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
                "min": 1.0,
                "max": 3.0,
                "avg": 2.0,
                "std": 0.816496580927726,  # sqrt(2/3)
                "population": [1.0, 2.0, 3.0],
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
                "min": 0.5,
                "max": 2.5,
                "avg": 1.5,
                "std": 0.816496580927726,  # sqrt(2/3),
                "population": [0.5, 1.5, 2.5],
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
                "min": 2.0,
                "max": 4.0,
                "avg": 3.0,
                "std": 0.816496580927726,  # sqrt(2/3)
                "population": [2.0, 3.0, 4.0],
            },
        },
    ]

    # Act & Assert
    for i, test_case in enumerate(test_cases, 1):
        # Process the request
        service.process_request(test_case["request"])

        # Get the current metrics
        metrics = service.get_generation_summary()
        expected = test_case["expected_metrics"]

        # Verify all metric values
        assert metrics.global_best == expected["global_min"], (
            f"Population {i}: Wrong global minimum"
        )
        assert metrics.min == expected["min"], (
            f"Population {i}: Wrong population minimum"
        )
        assert metrics.max == expected["max"], (
            f"Population {i}: Wrong population maximum"
        )
        assert np.isclose(metrics.avg, expected["avg"]), (
            f"Population {i}: Wrong population average"
        )
        assert np.isclose(metrics.std, expected["std"]), (
            f"Population {i}: Wrong population standard deviation"
        )
        assert metrics.population == expected["population"], (
            f"Population {i}: Wrong population values"
        )


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
        objectives={"metric1": OptimizationStrategy.MAXIMIZE},
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
        loop_controller.increment_generation()

    best_particle = positions[np.argmax(results)]
    best_result = np.max(results)

    assert np.allclose(best_particle, expected_max, atol=tol), (
        f"PSO failed to converge to the expected maximum. "
        f"Best found: {best_particle}, Expected: {expected_max}"
    )

    assert np.isclose(best_result, expected_max_val, atol=tol), (
        f"Best result {best_result} is not close enough to the global maximum value {expected_max_val}."
    )


def test_pareto_optimization_zdt1():
    """
    Test multi-objective optimization using ZDT1 benchmark problem.
    """

    EPS = 1e-9

    def zdt1_objectives(x: np.ndarray) -> tuple[float, float]:
        """Calculate both objectives for ZDT1 problem."""
        n = len(x)
        f1 = x[0]

        if n > 1:
            g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
        else:
            g = 1.0

        h = 1.0 - np.sqrt(f1 / g) if g > 0 else 0.0
        f2 = g * h

        return f1, f2

    def evaluate_zdt1_population(particles: np.ndarray) -> np.ndarray:
        """Evaluate ZDT1 for all particles, returning [f1, f2] for each."""
        results = np.array([zdt1_objectives(p) for p in particles])
        return results

    def create_multi_objective_candidates(
        positions: np.ndarray, results: np.ndarray, param_names: list
    ) -> list:
        """Generate candidates with two objectives."""
        return [
            {
                "control_vector": {"items": dict(zip(param_names, pos))},
                "cost_function_results": {
                    "values": {"objective_f1": res[0], "objective_f2": res[1]}
                },
            }
            for pos, res in zip(positions, results)
        ]

    def is_dominated(point, other_points):
        """Check if point is dominated by any point in other_points."""
        for other in other_points:
            if np.all(other <= point + EPS) and np.any(other < point - EPS):
                return True
        return False

    # Test configuration
    dim = 5
    lb, ub = 0.0, 1.0
    param_names = [f"x{i}" for i in range(dim)]
    num_particles = 100
    iterations = 500
    patience = 501

    np.random.seed(42)

    positions = np.random.uniform(lb, ub, (num_particles, dim))
    results = evaluate_zdt1_population(positions)

    candidates = create_multi_objective_candidates(positions, results, param_names)

    config = {
        "solution_candidates": candidates,
        "optimization_constraints": {"boundaries": {k: [lb, ub] for k in param_names}},
    }

    service = SolutionUpdaterService(
        optimization_engine=engine,
        max_generations=iterations,
        patience=patience,
        objectives={
            "objective_f1": OptimizationStrategy.MINIMIZE,
            "objective_f2": OptimizationStrategy.MINIMIZE,
        },
    )
    loop_controller = service.loop_controller

    while loop_controller.running():
        result = service.process_request(config)
        positions = np.array(
            [get_numpy_values(vec.items) for vec in result.next_iter_solutions]
        )
        results = evaluate_zdt1_population(positions)
        config["solution_candidates"] = create_multi_objective_candidates(
            positions, results, param_names
        )
        loop_controller.increment_generation()

    final_f1 = results[:, 0]
    final_f2 = results[:, 1]

    non_dominated_mask = np.array(
        [
            not is_dominated(results[i], np.delete(results, i, axis=0))
            for i in range(len(results))
        ]
    )

    pareto_front = results[non_dominated_mask]
    pareto_f1 = pareto_front[:, 0]
    pareto_f2 = pareto_front[:, 1]

    assert len(pareto_front) >= 10, (
        f"Expected at least 10 non-dominated solutions, got {len(pareto_front)}"
    )

    f1_range = pareto_f1.max() - pareto_f1.min()
    assert f1_range > 0.3, (
        f"Pareto solutions should span across f1 space, got range: {f1_range}"
    )

    theoretical_f2 = 1.0 - np.sqrt(pareto_f1)
    distance_to_front = np.abs(pareto_f2 - theoretical_f2)
    avg_distance = np.mean(distance_to_front)

    assert avg_distance < 0.20, (
        f"Average distance to theoretical Pareto front too large: {avg_distance:.4f}"
    )

    dominated_ratio = 1.0 - (len(pareto_front) / len(results))

    assert dominated_ratio < 0.85, (
        f"Too many dominated solutions in final population: {dominated_ratio:.2%}"
    )

    print("✓ ZDT1 Pareto optimization test passed:")
    print(f"  - F1 range: [{final_f1.min():.3f}, {final_f1.max():.3f}]")
    print(f"  - F2 range: [{final_f2.min():.3f}, {final_f2.max():.3f}]")
    print(f"  - Non-dominated solutions: {len(pareto_front)}/{len(results)}")
    print(f"  - Pareto F1 range: [{pareto_f1.min():.3f}, {pareto_f1.max():.3f}]")
    print(f"  - Avg distance to theoretical front: {avg_distance:.4f}")
    print(f"  - Dominated solutions: {dominated_ratio:.1%}")


def test_pareto_optimization_schaffer_n1():
    """Test Pareto optimization with Schaffer's function N.1."""

    def schaffer_n1_objectives(x: float) -> tuple[float, float]:
        """Calculate both objectives for Schaffer N.1 problem."""
        f1 = x**2
        f2 = (x - 2.0) ** 2
        return f1, f2

    def evaluate_schaffer_population(particles: np.ndarray) -> np.ndarray:
        """Evaluate Schaffer N.1 for all particles."""
        results = np.array([schaffer_n1_objectives(p[0]) for p in particles])
        return results

    def create_multi_objective_candidates(
        positions: np.ndarray, results: np.ndarray, param_names: list
    ) -> list:
        """Generate candidates with two objectives."""
        return [
            {
                "control_vector": {"items": dict(zip(param_names, pos))},
                "cost_function_results": {
                    "values": {"objective_f1": res[0], "objective_f2": res[1]}
                },
            }
            for pos, res in zip(positions, results)
        ]

    dim = 1
    lb, ub = -10.0, 10.0
    param_names = ["x"]
    num_particles = 20
    iterations = 50
    patience = 51

    np.random.seed(123)

    positions = np.random.uniform(lb, ub, (num_particles, dim))
    results = evaluate_schaffer_population(positions)

    candidates = create_multi_objective_candidates(positions, results, param_names)

    config = {
        "solution_candidates": candidates,
        "optimization_constraints": {"boundaries": {"x": [lb, ub]}},
    }

    service = SolutionUpdaterService(
        optimization_engine=engine,
        max_generations=iterations,
        patience=patience,
        objectives={
            "objective_f1": OptimizationStrategy.MINIMIZE,
            "objective_f2": OptimizationStrategy.MINIMIZE,
        },
    )
    loop_controller = service.loop_controller

    while loop_controller.running():
        result = service.process_request(config)
        positions = np.array(
            [get_numpy_values(vec.items) for vec in result.next_iter_solutions]
        )
        results = evaluate_schaffer_population(positions)
        config["solution_candidates"] = create_multi_objective_candidates(
            positions, results, param_names
        )
        loop_controller.increment_generation()

    final_x = positions[:, 0]
    final_f1 = results[:, 0]
    final_f2 = results[:, 1]

    pareto_optimal_mask = (final_x >= -0.5) & (final_x <= 2.5)
    pareto_ratio = np.sum(pareto_optimal_mask) / len(final_x)

    assert pareto_ratio > 0.5, (
        f"At least 50% of solutions should be near Pareto optimal region [0, 2], "
        f"got {pareto_ratio:.1%}"
    )

    pareto_x = final_x[pareto_optimal_mask]
    if len(pareto_x) > 1:
        x_range = pareto_x.max() - pareto_x.min()
        assert x_range > 0.5, (
            f"Solutions should span Pareto front, got range: {x_range:.3f}"
        )

    print("✓ Schaffer N.1 Pareto optimization test passed:")
    print(f"  - X range: [{final_x.min():.3f}, {final_x.max():.3f}]")
    print(f"  - Pareto optimal ratio: {pareto_ratio:.1%}")
    print(f"  - F1 range: [{final_f1.min():.3f}, {final_f1.max():.3f}]")
    print(f"  - F2 range: [{final_f2.min():.3f}, {final_f2.max():.3f}]")


def test_pareto_optimization_zdt3():
    """Test multi-objective optimization using ZDT3 benchmark problem."""

    EPS = 1e-9

    def zdt3_objectives(x: np.ndarray) -> tuple[float, float]:
        n = len(x)
        f1 = x[0]

        if n > 1:
            g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
        else:
            g = 1.0

        h = (
            1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
            if g > 0
            else 0.0
        )
        f2 = g * h

        return f1, f2

    def evaluate_zdt3_population(particles: np.ndarray) -> np.ndarray:
        results = np.array([zdt3_objectives(p) for p in particles])
        return results

    def create_multi_objective_candidates(
        positions: np.ndarray, results: np.ndarray, param_names: list
    ) -> list:
        return [
            {
                "control_vector": {"items": dict(zip(param_names, pos))},
                "cost_function_results": {
                    "values": {"objective_f1": res[0], "objective_f2": res[1]}
                },
            }
            for pos, res in zip(positions, results)
        ]

    def is_dominated(point, other_points):
        for other in other_points:
            if np.all(other <= point + EPS) and np.any(other < point - EPS):
                return True
        return False

    dim = 10
    lb, ub = 0.0, 1.0
    param_names = [f"x{i}" for i in range(dim)]
    num_particles = 200
    iterations = 500
    patience = 501

    np.random.seed(100)

    positions = np.zeros((num_particles, dim))
    positions[:, 0] = np.random.uniform(lb, ub, num_particles)
    positions[:, 1:] = np.random.uniform(
        0.0, 0.05, (num_particles, dim - 1)
    )  # Near zero

    results = evaluate_zdt3_population(positions)

    candidates = create_multi_objective_candidates(positions, results, param_names)

    config = {
        "solution_candidates": candidates,
        "optimization_constraints": {"boundaries": {k: [lb, ub] for k in param_names}},
    }

    service = SolutionUpdaterService(
        optimization_engine=engine,
        max_generations=iterations,
        patience=patience,
        objectives={
            "objective_f1": OptimizationStrategy.MINIMIZE,
            "objective_f2": OptimizationStrategy.MINIMIZE,
        },
    )
    loop_controller = service.loop_controller

    while loop_controller.running():
        result = service.process_request(config)
        positions = np.array(
            [get_numpy_values(vec.items) for vec in result.next_iter_solutions]
        )
        results = evaluate_zdt3_population(positions)
        config["solution_candidates"] = create_multi_objective_candidates(
            positions, results, param_names
        )
        loop_controller.increment_generation()

    final_f1 = results[:, 0]
    final_f2 = results[:, 1]

    non_dominated_mask = np.array(
        [
            not is_dominated(results[i], np.delete(results, i, axis=0))
            for i in range(len(results))
        ]
    )

    pareto_front = results[non_dominated_mask]
    pareto_f1 = pareto_front[:, 0]
    pareto_f2 = pareto_front[:, 1]

    assert len(pareto_front) >= 8, (
        f"Expected at least 8 non-dominated solutions, got {len(pareto_front)}"
    )

    f1_range = pareto_f1.max() - pareto_f1.min()
    assert f1_range > 0.25, (
        f"Pareto solutions should span across f1 space, got range: {f1_range}"
    )

    assert np.max(pareto_f2) < 2.0, (
        f"F2 values should be reasonable, got max: {np.max(pareto_f2):.4f}"
    )

    dominated_ratio = 1.0 - (len(pareto_front) / len(results))

    assert dominated_ratio < 0.95, (
        f"Too many dominated solutions in final population: {dominated_ratio:.2%}"
    )

    print("✓ ZDT3 Pareto optimization test passed:")
    print(f"  - F1 range: [{final_f1.min():.3f}, {final_f1.max():.3f}]")
    print(f"  - F2 range: [{final_f2.min():.3f}, {final_f2.max():.3f}]")
    print(f"  - Non-dominated solutions: {len(pareto_front)}/{len(results)}")
    print(f"  - Pareto F1 range: [{pareto_f1.min():.3f}, {pareto_f1.max():.3f}]")
    print(f"  - Pareto F2 range: [{pareto_f2.min():.3f}, {pareto_f2.max():.3f}]")
    print(f"  - Dominated solutions: {dominated_ratio:.1%}")


def test_pareto_optimization_kursawe():
    """Test multi-objective optimization using Kursawe benchmark problem."""

    EPS = 1e-9

    def kursawe_objectives(x: np.ndarray) -> tuple[float, float]:
        n = len(x)

        f1 = 0.0
        for i in range(n - 1):
            f1 += -10.0 * np.exp(-0.2 * np.sqrt(x[i] ** 2 + x[i + 1] ** 2))

        f2 = 0.0
        for i in range(n):
            f2 += np.abs(x[i]) ** 0.8 + 5.0 * np.sin(x[i] ** 3)

        return f1, f2

    def evaluate_kursawe_population(particles: np.ndarray) -> np.ndarray:
        results = np.array([kursawe_objectives(p) for p in particles])
        return results

    def create_multi_objective_candidates(
        positions: np.ndarray, results: np.ndarray, param_names: list
    ) -> list:
        return [
            {
                "control_vector": {"items": dict(zip(param_names, pos))},
                "cost_function_results": {
                    "values": {"objective_f1": res[0], "objective_f2": res[1]}
                },
            }
            for pos, res in zip(positions, results)
        ]

    def is_dominated(point, other_points):
        for other in other_points:
            if np.all(other <= point + EPS) and np.any(other < point - EPS):
                return True
        return False

    dim = 3
    lb, ub = -5.0, 5.0
    param_names = [f"x{i}" for i in range(dim)]
    num_particles = 80
    iterations = 300
    patience = 301

    np.random.seed(200)

    positions = np.random.uniform(lb, ub, (num_particles, dim))
    results = evaluate_kursawe_population(positions)

    candidates = create_multi_objective_candidates(positions, results, param_names)

    config = {
        "solution_candidates": candidates,
        "optimization_constraints": {"boundaries": {k: [lb, ub] for k in param_names}},
    }

    service = SolutionUpdaterService(
        optimization_engine=engine,
        max_generations=iterations,
        patience=patience,
        objectives={
            "objective_f1": OptimizationStrategy.MINIMIZE,
            "objective_f2": OptimizationStrategy.MINIMIZE,
        },
    )
    loop_controller = service.loop_controller

    while loop_controller.running():
        result = service.process_request(config)
        positions = np.array(
            [get_numpy_values(vec.items) for vec in result.next_iter_solutions]
        )
        results = evaluate_kursawe_population(positions)
        config["solution_candidates"] = create_multi_objective_candidates(
            positions, results, param_names
        )
        loop_controller.increment_generation()

    final_f1 = results[:, 0]
    final_f2 = results[:, 1]

    non_dominated_mask = np.array(
        [
            not is_dominated(results[i], np.delete(results, i, axis=0))
            for i in range(len(results))
        ]
    )

    pareto_front = results[non_dominated_mask]
    pareto_f1 = pareto_front[:, 0]
    pareto_f2 = pareto_front[:, 1]

    assert len(pareto_front) >= 5, (
        f"Expected at least 5 non-dominated solutions, got {len(pareto_front)}"
    )

    f1_range = pareto_f1.max() - pareto_f1.min()
    f2_range = pareto_f2.max() - pareto_f2.min()

    assert f1_range > 2.0, (
        f"Pareto solutions should span f1 space, got range: {f1_range:.3f}"
    )
    assert f2_range > 2.0, (
        f"Pareto solutions should span f2 space, got range: {f2_range:.3f}"
    )

    assert pareto_f1.max() < -10.0, (
        f"F1 maximum should be around -14, got: {pareto_f1.max():.4f}"
    )
    assert pareto_f1.min() > -25.0, (
        f"F1 minimum should be around -20, got: {pareto_f1.min():.4f}"
    )

    dominated_ratio = 1.0 - (len(pareto_front) / len(results))

    assert dominated_ratio < 0.85, (
        f"Too many dominated solutions in final population: {dominated_ratio:.2%}"
    )

    print("✓ Kursawe Pareto optimization test passed:")
    print(f"  - F1 range: [{final_f1.min():.3f}, {final_f1.max():.3f}]")
    print(f"  - F2 range: [{final_f2.min():.3f}, {final_f2.max():.3f}]")
    print(f"  - Non-dominated solutions: {len(pareto_front)}/{len(results)}")
    print(f"  - Pareto F1 range: [{pareto_f1.min():.3f}, {pareto_f1.max():.3f}]")
    print(f"  - Pareto F2 range: [{pareto_f2.min():.3f}, {pareto_f2.max():.3f}]")
    print(f"  - Dominated solutions: {dominated_ratio:.1%}")
