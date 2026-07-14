import numpy as np
import pytest

from common import OptimizationStrategy
from plugins.optimizers.pso import PSOEngine
from services.solution_updater_service.core.models import (
    SolutionUpdaterServiceResponse,
)
from services.solution_updater_service.core.service import (
    SolutionUpdaterService,
)

engine = PSOEngine.EngineName


@pytest.fixture
def mocked_engine():  # type: ignore
    """Fixture for a mocked OptimizationEngineInterface."""

    class MockedOptimizationEngine:
        def __init__(self):
            constant_best = 0.0
            self.global_best_result = constant_best
            self.global_best_control_vector = np.array([0.0, 0.0])

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
        def __init__(self) -> None:
            self.global_best_result = 0.0
            self.global_best_control_vector = np.array([0.0, 0.0])

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
            self.global_best_result = 1000.0
            self.global_best_control_vector = np.array([0.0, 0.0])
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


@pytest.mark.parametrize(
    "config_json, expected_result_parameters, objectives",
    [
        (
            {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param1": 1.0, "param2": 2.0}},
                        "cost_function_results": {"values": {"metric1": 10.0}},
                    }
                ],
                "optimization_constrains": {
                    "boundaries": {
                        "param1": {"lb": -10, "ub": 10.0},
                        "param2": {"lb": -1.0, "ub": 1.00},
                    },
                },
            },
            [{"param1": 2.0, "param2": 3.0}],
            {"metric1": OptimizationStrategy.MINIMIZE},
        ),
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
                "optimization_constrains": {
                    "boundaries": {
                        "param1": {"lb": -10, "ub": 10.0},
                        "param2": {"lb": -1.0, "ub": 1.00},
                    },
                },
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
    service = SolutionUpdaterService(
        optimization_engine=engine,
        max_generations=100,
        max_stall_generations=101,
        objectives=objectives,
    )

    monkeypatch.setattr(service, "_engine", mocked_engine)

    result = service.process_request(config_json)

    assert isinstance(result, SolutionUpdaterServiceResponse)
    assert len(result.next_iter_solutions) == len(expected_result_parameters)
    for actual, expected in zip(result.next_iter_solutions, expected_result_parameters):
        assert actual.items == expected


@pytest.mark.parametrize(
    "config_json, expected_result_parameters",
    [
        (
            {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param1": 1.0, "param2": 2.0}},
                        "cost_function_results": {"values": {"metric1": 10.0}},
                    }
                ],
                "optimization_constrains": {
                    "boundaries": {
                        "param1": {"lb": -10, "ub": 10.0},
                        "param2": {"lb": -1.0, "ub": 1.00},
                    },
                },
            },
            [{"param1": 2.0, "param2": 3.0}],
        ),
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
                "optimization_constrains": {
                    "boundaries": {
                        "param1": {"lb": -10, "ub": 10.0},
                        "param2": {"lb": -1.0, "ub": 1.00},
                    },
                },
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
    service = SolutionUpdaterService(
        optimization_engine=engine,
        max_generations=100,
        max_stall_generations=101,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
    )

    monkeypatch.setattr(service, "_engine", mocked_engine)

    result = service.process_request(config_json)

    assert isinstance(result, SolutionUpdaterServiceResponse)
    assert len(result.next_iter_solutions) == len(expected_result_parameters)
    for actual, expected in zip(result.next_iter_solutions, expected_result_parameters):
        assert actual.items == expected


@pytest.mark.parametrize(
    "config_json_1, config_json_2, expected_result_1, expected_result_2",
    [
        (
            {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param1": 1.0, "param2": 2.0}},
                        "cost_function_results": {"values": {"metric1": 10.0}},
                    }
                ],
                "optimization_constrains": {
                    "boundaries": {
                        "param1": {"lb": -10, "ub": 10.0},
                        "param2": {"lb": -1.0, "ub": 1.00},
                    },
                },
            },
            {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param2": 3.0, "param1": 2.0}},
                        "cost_function_results": {"values": {"metric1": 15.0}},
                    }
                ],
                "optimization_constrains": {
                    "boundaries": {
                        "param1": {"lb": -10, "ub": 10.0},
                        "param2": {"lb": -1.0, "ub": 1.00},
                    },
                },
            },
            [{"param1": 2.0, "param2": 3.0}],
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
    service = SolutionUpdaterService(
        optimization_engine=engine,
        max_generations=100,
        max_stall_generations=101,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
    )

    monkeypatch.setattr(service, "_engine", mocked_engine)

    result_1 = service.process_request(config_json_1)
    result_2 = service.process_request(config_json_2)

    assert isinstance(result_1, SolutionUpdaterServiceResponse)
    assert len(result_1.next_iter_solutions) == len(expected_result_1)
    for actual, expected in zip(result_1.next_iter_solutions, expected_result_1):
        assert actual.items == expected

    assert isinstance(result_2, SolutionUpdaterServiceResponse)
    assert len(result_2.next_iter_solutions) == len(expected_result_2)
    for actual, expected in zip(result_2.next_iter_solutions, expected_result_2):
        assert actual.items == expected


@pytest.mark.parametrize(
    "config_json, expected_result_parameters",
    [
        (
            {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param1": 1.0, "param2": 2.0}},
                        "cost_function_results": {"values": {"metric1": 10.0}},
                    }
                ],
                "optimization_constrains": {
                    "boundaries": {
                        "param1": {"lb": 0, "ub": 5},
                        "param2": {"lb": 0, "ub": 3},
                    },
                },
            },
            np.array([[2.0, 3.0]]),
        ),
        (
            {
                "solution_candidates": [
                    {
                        "control_vector": {"items": {"param1": 4.5, "param2": 2.2}},
                        "cost_function_results": {"values": {"metric1": 15.0}},
                    }
                ],
                "optimization_constrains": {
                    "boundaries": {
                        "param1": {"lb": 0, "ub": 4},
                        "param2": {"lb": 0, "ub": 3},
                    },
                },
            },
            np.array([[4.0, 3.0]]),
        ),
    ],
)
def test_update_solution_with_boundaries_np(  # type: ignore
    config_json, expected_result_parameters, mocked_engine_with_bnb, monkeypatch
):
    service = SolutionUpdaterService(
        optimization_engine=engine,
        max_generations=100,
        max_stall_generations=101,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
    )

    monkeypatch.setattr(service, "_engine", mocked_engine_with_bnb)

    result = service.process_request(config_json)

    assert isinstance(result, SolutionUpdaterServiceResponse)
    assert len(result.next_iter_solutions) == expected_result_parameters.shape[0]
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
    return np.apply_along_axis(function, 1, particles)


def create_candidates(
    positions: np.ndarray, results: np.ndarray, param_names: list
) -> list:
    return [
        {
            "control_vector": {"items": dict(zip(param_names, pos))},
            "cost_function_results": {"values": {"metric1": res}},
        }
        for pos, res in zip(positions, results)
    ]


@pytest.mark.parametrize(
    "max_stall_generations, engine_fixture, max_stall_generations_checks",
    [
        (
            2,
            "mocked_engine_with_2_stagnant_result",
            [
                {"generation": 1, "running": True, "max_stall_generations_left": 2},
                {"generation": 2, "running": True, "max_stall_generations_left": 1},
                {"generation": 3, "running": True, "max_stall_generations_left": 2},
            ],
        ),
        (
            1,
            "mocked_engine",
            [
                {"generation": 1, "running": True, "max_stall_generations_left": 1},
                {"generation": 2, "running": False, "max_stall_generations_left": 0},
            ],
        ),
    ],
)
def test_max_stall_generations_handling(
    max_stall_generations,
    engine_fixture,
    max_stall_generations_checks,
    request,
    monkeypatch,
):
    service = SolutionUpdaterService(
        optimization_engine=engine,
        max_generations=10,
        max_stall_generations=max_stall_generations,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
    )

    mocked_engine = request.getfixturevalue(engine_fixture)
    monkeypatch.setattr(service, "_engine", mocked_engine)

    config_json = {
        "solution_candidates": [
            {
                "control_vector": {"items": {"param1": 1.0, "param2": 2.0}},
                "cost_function_results": {"values": {"metric1": 10.0}},
            }
        ],
        "optimization_constrains": {
            "boundaries": {"param1": {"lb": 0, "ub": 5}, "param2": {"lb": 0, "ub": 3}},
        },
    }

    loop_controller = service.loop_controller
    assert loop_controller.running() is True

    for i, check in enumerate(max_stall_generations_checks, 1):
        service.process_request(config_json)
        loop_controller.increment_generation()

        assert loop_controller.current_generation == check["generation"]
        assert loop_controller.running() is check["running"]
        assert loop_controller._stall_left == check["max_stall_generations_left"]
