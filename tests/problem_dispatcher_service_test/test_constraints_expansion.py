import random

import pytest

from services.problem_dispatcher_service import ProblemDispatcherService


@pytest.fixture
def md_problem_definition():
    return {
        "well_placement": [
            {
                "well_name": "INJ",
                "initial_state": {
                    "well_type": "IWell",
                    "wellhead": {"x": 0, "y": 0, "z": 0},
                    "md": 2500,
                    "perforations": [{"start_md": 0.0, "end_md": 2500.0}],
                },
                "optimization_constraints": {
                    "wellhead": {
                        "x": {"lb": 10, "ub": 3190},
                        "y": {"lb": 10, "ub": 3190},
                    },
                    "md": {"lb": 2000, "ub": 2700},
                },
            },
            {
                "well_name": "PRO",
                "initial_state": {
                    "well_type": "IWell",
                    "wellhead": {"x": 0, "y": 0, "z": 0},
                    "md": 2500,
                    "perforations": [{"start_md": 0.0, "end_md": 2500.0}],
                },
                "optimization_constraints": {
                    "wellhead": {
                        "x": {"lb": 10, "ub": 3190},
                        "y": {"lb": 10, "ub": 3190},
                    },
                    "md": {"lb": 2000, "ub": 2700},
                },
            },
        ],
        "optimization_parameters": {
            "optimization_strategy": "maximize",
            "linear_inequalities": {
                "A": [{"INJ.md": 1.0, "PRO.md": 1.0}],
                "b": [3000.0],
                "sense": ["<="],
            },
        },
    }


def test_boundaries_include_md(md_problem_definition):
    svc = ProblemDispatcherService(problem_definition=md_problem_definition, n_size=1)
    boundaries = svc.get_boundaries()
    assert boundaries["well_placement#INJ#md"] == (2000.0, 2700.0)
    assert boundaries["well_placement#PRO#md"] == (2000.0, 2700.0)


def test_generation_uses_md_bounds(md_problem_definition, monkeypatch):
    svc = ProblemDispatcherService(problem_definition=md_problem_definition, n_size=2)

    # Monkeypatch random.uniform to midpoint
    def mid(a, b):
        return (a + b) / 2.0

    monkeypatch.setattr(random, "uniform", mid)
    resp = svc.process_iteration()
    assert len(resp.solution_candidates) == 2
    for sc in resp.solution_candidates:
        task = next(iter(sc.tasks.values()))
        md_inj = task.control_vector.items["well_placement#INJ#md"]
        md_pro = task.control_vector.items["well_placement#PRO#md"]
        assert md_inj == pytest.approx((2000.0 + 2700.0) / 2.0)
        assert md_pro == pytest.approx((2000.0 + 2700.0) / 2.0)


def test_pso_with_optimum_beyond_md_bound_moves_toward_ub(md_problem_definition):
    """Ensure PSO tries to move towards the unconstrained optimum (2800) but respects ub=2700."""
    import numpy as np

    from services.solution_updater_service.core.engines.pso import PSOEngine

    svc = ProblemDispatcherService(problem_definition=md_problem_definition, n_size=1)
    boundaries = svc.get_boundaries()
    lb, ub = boundaries["well_placement#INJ#md"]

    engine = PSOEngine(seed=42)

    parameters = np.array([[2500.0], [2400.0], [2600.0]], dtype=float)

    target = 2800.0
    results = np.abs(parameters - target)

    lb_arr = np.array([lb], dtype=float)
    ub_arr = np.array([ub], dtype=float)

    new_positions = engine.update_solution_to_next_iter(
        parameters, results, lb_arr, ub_arr
    )

    # Final positions must be within bounds
    assert np.all(new_positions >= lb_arr - 1e-8)
    assert np.all(new_positions <= ub_arr + 1e-8)

    # Since the target is above current positions, mean should increase (move toward ub)
    assert float(new_positions.mean()) >= float(parameters.mean())
