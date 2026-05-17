from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from common import OptimizationStrategy
from orchestration.risk_management_service.core.service.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from services.problem_dispatcher_service.core.models import ProblemDispatcherDefinition
from plugins.optimizers.pso import PSOEngine
from services.solution_updater_service.core.service import SolutionUpdaterService


def _make_problem_definition(tmp_path: Path) -> ProblemDispatcherDefinition:
    return ProblemDispatcherDefinition.model_validate(
        {
            "well_design": [
                {
                    "well_name": "W1",
                    "initial_state": {
                        "well_type": "IWell",
                        "wellhead": {"x": 0, "y": 50, "z": 0},
                        "md": 200,
                        "perforations": {"p1": {"start_md": 100.0, "end_md": 200.0}},
                    },
                    "parameter_bounds": {
                        "wellhead": {"x": {"lb": 0, "ub": 100}},
                        "md": {"lb": 0, "ub": 300},
                    },
                }
            ],
            "optimization_parameters": {
                "objectives": {"metric1": "minimize"},
                "max_generations": 20,
                "population_size": 2,
                "max_stall_generations": 10,
            },
            "simulation_config": {
                "worker_count": 1,
                "checkpoint_interval": 5,
                "checkpoint_path": str(tmp_path),
            },
            "plugins": {
                "connector": "opendarts",
                "optimizer": "pso",
                "well_management": "builtin",
            },
        }
    )


def _run_n_generations(
    service: SolutionUpdaterService,
    n: int,
    param_names: list[str],
    n_particles: int,
) -> list[Any]:
    """Run *n* generations on *service* using deterministic fake results."""
    boundaries = {p: {"lb": -10.0, "ub": 10.0} for p in param_names}
    rng = np.random.default_rng(0)
    params: list[dict[str, float]] = [
        dict(zip(param_names, rng.uniform(-5, 5, len(param_names))))
        for _ in range(n_particles)
    ]

    next_cvs: list[Any] = []
    for gen in range(n):
        results = [float(gen * 10 + i) for i in range(n_particles)]
        request = {
            "solution_candidates": [
                {
                    "control_vector": {"items": p},
                    "cost_function_results": {"values": {"metric1": r}},
                }
                for p, r in zip(params, results)
            ],
            "optimization_constrains": {"boundaries": boundaries},
        }
        response = service.process_request(request)
        service.loop_controller.increment_generation()
        next_cvs = response.next_iter_solutions
        params = [cv.items for cv in next_cvs]

    return next_cvs


@pytest.fixture
def synthetic_engine_state() -> dict[str, Any]:
    """Synthetic PSO engine state arrays for roundtrip testing."""
    n_particles, n_dims, n_obj = 3, 2, 1
    rng = np.random.default_rng(42)
    return {
        "particles_best_positions": rng.random((n_particles, n_dims)),
        "particles_best_results": rng.random((n_particles, n_obj)),
        "global_best_position": rng.random((n_dims,)),
        "global_best_result": np.array([0.5]),
        "velocities": rng.random((n_particles, n_dims)) * 0.1,
        "external_archive_positions": np.empty((0, n_dims), dtype=np.float64),
        "external_archive_results": np.empty((0, n_obj), dtype=np.float64),
        "rng_state_json": json.dumps(rng.bit_generator.state),
    }


@pytest.fixture
def full_state(synthetic_engine_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "engine_state": synthetic_engine_state,
        "mapper_state": {
            "control_vector_mapping": {"param1": 0, "param2": 1},
            "results_mapping": {"metric1": 0},
            "population_size": 3,
        },
        "loop_controller_state": {
            "current_generation": 7,
            "stall_left": 4,
            "last_best": 0.5,
            "pareto_min_history": np.empty((0,), dtype=np.float64),
            "pareto_mean_history": np.empty((0,), dtype=np.float64),
        },
        "next_positions": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    }


def test_save_load_preserves_config(tmp_path: Path, full_state: dict[str, Any]) -> None:
    problem_def = _make_problem_definition(tmp_path)
    cp = tmp_path / "cp.npz"

    save_checkpoint(cp, problem_def, full_state, "deadbeef", "2b1e7585")
    loaded = load_checkpoint(cp)

    assert loaded["config"].model_dump_json() == problem_def.model_dump_json()


def test_save_load_preserves_model_hash(
    tmp_path: Path, full_state: dict[str, Any]
) -> None:
    problem_def = _make_problem_definition(tmp_path)
    cp = tmp_path / "cp.npz"

    save_checkpoint(cp, problem_def, full_state, "abc123", "2b1e7585")
    loaded = load_checkpoint(cp)

    assert loaded["model_hash"] == "abc123"


def test_save_load_preserves_generation_and_stall(
    tmp_path: Path, full_state: dict[str, Any]
) -> None:
    problem_def = _make_problem_definition(tmp_path)
    cp = tmp_path / "cp.npz"

    save_checkpoint(cp, problem_def, full_state, "h", "2b1e7585")
    loaded = load_checkpoint(cp)

    lc = loaded["loop_controller_state"]
    assert lc["current_generation"] == 7
    assert lc["stall_left"] == 4
    assert lc["last_best"] == pytest.approx(0.5)


def test_save_load_last_best_none_roundtrips(
    tmp_path: Path, full_state: dict[str, Any]
) -> None:
    full_state["loop_controller_state"]["last_best"] = None
    problem_def = _make_problem_definition(tmp_path)
    cp = tmp_path / "cp.npz"

    save_checkpoint(cp, problem_def, full_state, "h", "2b1e7585")
    loaded = load_checkpoint(cp)

    assert loaded["loop_controller_state"]["last_best"] is None


def test_save_load_preserves_next_positions(
    tmp_path: Path, full_state: dict[str, Any]
) -> None:
    problem_def = _make_problem_definition(tmp_path)
    cp = tmp_path / "cp.npz"

    save_checkpoint(cp, problem_def, full_state, "h", "2b1e7585")
    loaded = load_checkpoint(cp)

    np.testing.assert_array_almost_equal(
        loaded["next_positions"], full_state["next_positions"]
    )


def test_save_load_preserves_engine_arrays(
    tmp_path: Path, full_state: dict[str, Any], synthetic_engine_state: dict[str, Any]
) -> None:
    problem_def = _make_problem_definition(tmp_path)
    cp = tmp_path / "cp.npz"

    save_checkpoint(cp, problem_def, full_state, "h", "2b1e7585")
    loaded = load_checkpoint(cp)

    for key in ("particles_best_positions", "global_best_position", "velocities"):
        np.testing.assert_array_almost_equal(
            loaded["engine_state"][key], synthetic_engine_state[key]
        )


def test_save_load_preserves_mapper_state(
    tmp_path: Path, full_state: dict[str, Any]
) -> None:
    problem_def = _make_problem_definition(tmp_path)
    cp = tmp_path / "cp.npz"

    save_checkpoint(cp, problem_def, full_state, "h", "2b1e7585")
    loaded = load_checkpoint(cp)

    ms = loaded["mapper_state"]
    assert ms["control_vector_mapping"] == {"param1": 0, "param2": 1}
    assert ms["results_mapping"] == {"metric1": 0}
    assert ms["population_size"] == 3


def test_save_checkpoint_writes_file(
    tmp_path: Path, full_state: dict[str, Any]
) -> None:
    problem_def = _make_problem_definition(tmp_path)
    cp = tmp_path / "cp.npz"

    save_checkpoint(cp, problem_def, full_state, "h", "2b1e7585")

    assert cp.exists()
    assert not (tmp_path / "cp.tmp.npz").exists(), "temp file should be cleaned up"


# SolutionUpdaterService checkpoint: state restoration

PARAM_NAMES = ["param1", "param2"]
N_PARTICLES = 3
OBJECTIVES = {"metric1": OptimizationStrategy.MINIMIZE}


@pytest.fixture
def warmed_up_service() -> tuple[SolutionUpdaterService, list[Any]]:
    """Service that has completed 5 generations."""
    service = SolutionUpdaterService(
        optimization_engine=PSOEngine.EngineName,
        max_generations=20,
        max_stall_generations=10,
        objectives=OBJECTIVES,
        seed=42,
    )
    next_cvs = _run_n_generations(service, 5, PARAM_NAMES, N_PARTICLES)
    return service, next_cvs


def test_restore_resumes_at_correct_generation(
    warmed_up_service: tuple[SolutionUpdaterService, list[Any]],
) -> None:
    service_a, next_cvs = warmed_up_service
    assert service_a.loop_controller.current_generation == 5

    state = service_a.get_checkpoint_state(next_cvs)

    service_b = SolutionUpdaterService(
        optimization_engine=PSOEngine.EngineName,
        max_generations=20,
        max_stall_generations=10,
        objectives=OBJECTIVES,
        seed=99,
    )
    service_b.restore_checkpoint_state(state)

    assert service_b.loop_controller.current_generation == 5


def test_restore_does_not_start_from_generation_zero(
    warmed_up_service: tuple[SolutionUpdaterService, list[Any]],
) -> None:
    service_a, next_cvs = warmed_up_service
    state = service_a.get_checkpoint_state(next_cvs)

    service_b = SolutionUpdaterService(
        optimization_engine=PSOEngine.EngineName,
        max_generations=20,
        max_stall_generations=10,
        objectives=OBJECTIVES,
    )
    service_b.restore_checkpoint_state(state)

    assert service_b.loop_controller.current_generation != 0


def test_restore_preserves_stall_left(
    warmed_up_service: tuple[SolutionUpdaterService, list[Any]],
) -> None:
    service_a, next_cvs = warmed_up_service
    state = service_a.get_checkpoint_state(next_cvs)
    expected_stall = service_a.loop_controller._stall_left

    service_b = SolutionUpdaterService(
        optimization_engine=PSOEngine.EngineName,
        max_generations=20,
        max_stall_generations=10,
        objectives=OBJECTIVES,
    )
    service_b.restore_checkpoint_state(state)

    assert service_b.loop_controller._stall_left == expected_stall


def test_restore_preserves_global_best_result(
    warmed_up_service: tuple[SolutionUpdaterService, list[Any]],
) -> None:
    service_a, next_cvs = warmed_up_service
    state = service_a.get_checkpoint_state(next_cvs)

    service_b = SolutionUpdaterService(
        optimization_engine=PSOEngine.EngineName,
        max_generations=20,
        max_stall_generations=10,
        objectives=OBJECTIVES,
    )
    service_b.restore_checkpoint_state(state)

    np.testing.assert_array_almost_equal(
        service_b.global_best_result,
        service_a.global_best_result,
    )


def test_restore_preserves_mapper_parameter_names(
    warmed_up_service: tuple[SolutionUpdaterService, list[Any]],
) -> None:
    service_a, next_cvs = warmed_up_service
    state = service_a.get_checkpoint_state(next_cvs)

    service_b = SolutionUpdaterService(
        optimization_engine=PSOEngine.EngineName,
        max_generations=20,
        max_stall_generations=10,
        objectives=OBJECTIVES,
    )
    service_b.restore_checkpoint_state(state)

    assert service_b._mapper.parameters_name == service_a._mapper.parameters_name


def test_restore_loop_controller_still_running(
    warmed_up_service: tuple[SolutionUpdaterService, list[Any]],
) -> None:
    service_a, next_cvs = warmed_up_service
    state = service_a.get_checkpoint_state(next_cvs)

    service_b = SolutionUpdaterService(
        optimization_engine=PSOEngine.EngineName,
        max_generations=20,
        max_stall_generations=10,
        objectives=OBJECTIVES,
    )
    service_b.restore_checkpoint_state(state)

    # 5 gens done, 20 max — should still be running
    assert service_b.loop_controller.running()


def test_restore_returns_next_positions_as_control_vectors(
    warmed_up_service: tuple[SolutionUpdaterService, list[Any]],
) -> None:
    service_a, next_cvs = warmed_up_service
    state = service_a.get_checkpoint_state(next_cvs)

    service_b = SolutionUpdaterService(
        optimization_engine=PSOEngine.EngineName,
        max_generations=20,
        max_stall_generations=10,
        objectives=OBJECTIVES,
    )
    restored_cvs = service_b.restore_checkpoint_state(state)

    assert len(restored_cvs) == len(next_cvs)
    for restored, original in zip(restored_cvs, next_cvs):
        for key in original.items:
            assert restored.items[key] == pytest.approx(original.items[key])


# Full file roundtrip: save → load → restore


def test_file_roundtrip_restores_generation(tmp_path: Path) -> None:
    problem_def = _make_problem_definition(tmp_path)

    service_a = SolutionUpdaterService(
        optimization_engine=PSOEngine.EngineName,
        max_generations=20,
        max_stall_generations=10,
        objectives=OBJECTIVES,
        seed=42,
    )
    next_cvs = _run_n_generations(service_a, 5, PARAM_NAMES, N_PARTICLES)
    state = service_a.get_checkpoint_state(next_cvs)

    cp = tmp_path / "cp.npz"
    save_checkpoint(cp, problem_def, state, "model_hash_abc", "2b1e7585")
    loaded = load_checkpoint(cp)

    service_b = SolutionUpdaterService(
        optimization_engine=PSOEngine.EngineName,
        max_generations=20,
        max_stall_generations=10,
        objectives=OBJECTIVES,
        seed=99,
    )
    service_b.restore_checkpoint_state(loaded)

    assert service_b.loop_controller.current_generation == 5


def test_file_roundtrip_global_best_matches(tmp_path: Path) -> None:
    problem_def = _make_problem_definition(tmp_path)

    service_a = SolutionUpdaterService(
        optimization_engine=PSOEngine.EngineName,
        max_generations=20,
        max_stall_generations=10,
        objectives=OBJECTIVES,
        seed=42,
    )
    next_cvs = _run_n_generations(service_a, 5, PARAM_NAMES, N_PARTICLES)
    state = service_a.get_checkpoint_state(next_cvs)

    cp = tmp_path / "cp.npz"
    save_checkpoint(cp, problem_def, state, "model_hash_abc", "2b1e7585")
    loaded = load_checkpoint(cp)

    service_b = SolutionUpdaterService(
        optimization_engine=PSOEngine.EngineName,
        max_generations=20,
        max_stall_generations=10,
        objectives=OBJECTIVES,
    )
    service_b.restore_checkpoint_state(loaded)

    np.testing.assert_array_almost_equal(
        service_b.global_best_result,
        service_a.global_best_result,
    )


def test_file_roundtrip_service_can_continue(tmp_path: Path) -> None:
    """After restore, the service can process another generation without error."""
    problem_def = _make_problem_definition(tmp_path)

    service_a = SolutionUpdaterService(
        optimization_engine=PSOEngine.EngineName,
        max_generations=20,
        max_stall_generations=10,
        objectives=OBJECTIVES,
        seed=42,
    )
    next_cvs = _run_n_generations(service_a, 3, PARAM_NAMES, N_PARTICLES)
    state = service_a.get_checkpoint_state(next_cvs)

    cp = tmp_path / "cp.npz"
    save_checkpoint(cp, problem_def, state, "h", "2b1e7585")
    loaded = load_checkpoint(cp)

    service_b = SolutionUpdaterService(
        optimization_engine=PSOEngine.EngineName,
        max_generations=20,
        max_stall_generations=10,
        objectives=OBJECTIVES,
    )
    restored_cvs = service_b.restore_checkpoint_state(loaded)

    # Drive one more generation using the restored positions
    boundaries = {p: {"lb": -10.0, "ub": 10.0} for p in PARAM_NAMES}
    request = {
        "solution_candidates": [
            {
                "control_vector": {"items": cv.items},
                "cost_function_results": {"values": {"metric1": 1.0}},
            }
            for cv in restored_cvs
        ],
        "optimization_constrains": {"boundaries": boundaries},
    }
    response = service_b.process_request(request)
    service_b.loop_controller.increment_generation()

    assert service_b.loop_controller.current_generation == 4
    assert len(response.next_iter_solutions) == N_PARTICLES
