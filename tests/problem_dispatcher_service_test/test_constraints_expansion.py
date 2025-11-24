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
            "optimization_strategy": "minimize",
            "linear_inequalities": {
                "A": [{"INJ.md": 1.0, "PRO.md": 1.0}],
                "b": [5000.0],
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

    engine = PSOEngine(svc.get_optimization_strategy(), seed=42)

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


def test_initial_generation_respects_linear_inequalities(md_problem_definition):
    """First population should satisfy INJ.md + PRO.md <= 5000 given md bounds [2000, 2700]."""
    svc = ProblemDispatcherService(problem_definition=md_problem_definition, n_size=50)
    resp = svc.process_iteration()
    assert len(resp.solution_candidates) == 50

    for sc in resp.solution_candidates:
        payload = next(iter(sc.tasks.values()))
        cv = payload.control_vector.items
        inj = cv["well_placement#INJ#md"]
        pro = cv["well_placement#PRO#md"]

        assert 2000.0 <= inj <= 2700.0
        assert 2000.0 <= pro <= 2700.0

        assert inj + pro <= 5000.0 + 1e-6


def test_jwell_constraints_respected():
    problem_definition = {
        "well_placement": [
            {
                "well_name": "J1",
                "initial_state": {
                    "well_type": "JWell",
                    "wellhead": {"x": 0, "y": 0, "z": 0},
                    "md_linear1": 500.0,
                    "md_curved": 300.0,
                    "dls": 15.0,
                    "md_linear2": 700.0,
                    "azimuth": 90.0,
                    "md_step": 20.0,
                    "perforations": [{"start_md": 100.0, "end_md": 200.0}],
                },
                "optimization_constraints": {
                    "md_linear1": {"lb": 400, "ub": 600},
                    "azimuth": {"lb": 0, "ub": 180},
                },
            }
        ],
        "optimization_parameters": {"optimization_strategy": "maximize"},
    }

    svc = ProblemDispatcherService(problem_definition=problem_definition, n_size=10)
    resp = svc.process_iteration()

    for sc in resp.solution_candidates:
        task = next(iter(sc.tasks.values()))
        cv = task.control_vector.items

        md_l1 = cv["well_placement#J1#md_linear1"]
        azi = cv["well_placement#J1#azimuth"]

        assert 400 <= md_l1 <= 600
        assert 0 <= azi <= 180


def test_hwell_constraints_respected():
    problem_definition = {
        "well_placement": [
            {
                "well_name": "H1",
                "initial_state": {
                    "well_type": "HWell",
                    "wellhead": {"x": 0, "y": 0, "z": 0},
                    "TVD": 1000.0,
                    "md_lateral": 1500.0,
                    "azimuth": 45.0,
                    "md_step": 10.0,
                    "perforations": [{"start_md": 100.0, "end_md": 200.0}],
                },
                "optimization_constraints": {
                    "TVD": {"lb": 900, "ub": 1100},
                    "md_lateral": {"lb": 1400, "ub": 1600},
                },
            }
        ],
        "optimization_parameters": {"optimization_strategy": "maximize"},
    }

    svc = ProblemDispatcherService(problem_definition=problem_definition, n_size=10)
    resp = svc.process_iteration()

    for sc in resp.solution_candidates:
        task = next(iter(sc.tasks.values()))
        cv = task.control_vector.items

        tvd = cv["well_placement#H1#TVD"]
        lat = cv["well_placement#H1#md_lateral"]

        assert 900 <= tvd <= 1100
        assert 1400 <= lat <= 1600


def test_linear_inequalities_mixed_wells():
    # Constraint: J1.md_linear1 + I1.md <= 1000
    problem_definition = {
        "well_placement": [
            {
                "well_name": "J1",
                "initial_state": {
                    "well_type": "JWell",
                    "wellhead": {"x": 0, "y": 0, "z": 0},
                    "md_linear1": 500.0,
                    "md_curved": 300.0,
                    "dls": 15.0,
                    "md_linear2": 700.0,
                    "azimuth": 90.0,
                    "md_step": 20.0,
                    "perforations": [{"start_md": 100.0, "end_md": 200.0}],
                },
                "optimization_constraints": {
                    "md_linear1": {"lb": 400, "ub": 600},
                },
            },
            {
                "well_name": "I1",
                "initial_state": {
                    "well_type": "IWell",
                    "wellhead": {"x": 0, "y": 0, "z": 0},
                    "md": 500.0,
                    "perforations": [{"start_md": 0.0, "end_md": 500.0}],
                },
                "optimization_constraints": {
                    "md": {"lb": 400, "ub": 600},
                },
            },
        ],
        "optimization_parameters": {
            "optimization_strategy": "maximize",
            "linear_inequalities": {
                "A": [{"J1.md_linear1": 1.0, "I1.md": 1.0}],
                "b": [1000.0],
                "sense": ["<="],
            },
        },
    }

    svc = ProblemDispatcherService(problem_definition=problem_definition, n_size=20)
    resp = svc.process_iteration()

    for sc in resp.solution_candidates:
        task = next(iter(sc.tasks.values()))
        cv = task.control_vector.items

        j1_md = cv["well_placement#J1#md_linear1"]
        i1_md = cv["well_placement#I1#md"]

        assert j1_md + i1_md <= 1000.0 + 1e-6
