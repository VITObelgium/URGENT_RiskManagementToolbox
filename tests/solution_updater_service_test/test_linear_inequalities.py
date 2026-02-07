from common import OptimizationStrategy
from services.solution_updater_service.core.models import (
    OptimizationEngine,
)
from services.solution_updater_service.core.service import SolutionUpdaterService


def _build_candidate(param_values: dict[str, float], metric: float):
    return {
        "control_vector": {"items": param_values},
        "cost_function_results": {"values": {"metric1": metric}},
    }


def test_pso_respects_linear_inequality_sum_constraint():
    """Particles should stay (or be repaired) within INJ.md + PRO.md <= 3000 after several iterations."""
    service = SolutionUpdaterService(
        optimization_engine=OptimizationEngine.PSO,
        max_generations=5,
        patience=5,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
    )

    request = {
        "solution_candidates": [
            _build_candidate(
                {"well_design#INJ#md": 2500.0, "well_design#PRO#md": 1500.0}, 10.0
            ),  # infeasible
            _build_candidate(
                {"well_design#INJ#md": 2000.0, "well_design#PRO#md": 1000.0}, 9.0
            ),  # feasible
        ],
        "parameter_bounds": {
            "boundaries": {
                "well_design#INJ#md": [0.0, 3000.0],
                "well_design#PRO#md": [0.0, 3000.0],
            },
            "A": [
                {"INJ.md": 1.0, "PRO.md": 1.0},
            ],
            "b": [3000.0],
            "sense": ["<="],
        },
    }

    for _ in range(3):
        resp = service.process_request(request)
        request["solution_candidates"] = [
            {
                "control_vector": {"items": cv.items},
                "cost_function_results": {"values": {"metric1": 10.0}},
            }
            for cv in resp.next_iter_solutions
        ]

    for cv in resp.next_iter_solutions:
        inj = cv.items["well_design#INJ#md"]
        pro = cv.items["well_design#PRO#md"]
        assert inj + pro <= 3000.0 + 1e-6


def test_penalty_applied_when_violating_constraint():
    """First iteration should set global best from feasible candidate despite worse raw metric if infeasible penalized."""
    service = SolutionUpdaterService(
        optimization_engine=OptimizationEngine.PSO,
        max_generations=1,
        patience=1,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
    )

    request = {
        "solution_candidates": [
            _build_candidate(
                {"well_design#INJ#md": 1600.0, "well_design#PRO#md": 1600.0}, 5.0
            ),  # infeasible (sum 3200)
            _build_candidate(
                {"well_design#INJ#md": 1500.0, "well_design#PRO#md": 1500.0}, 6.0
            ),  # feasible (sum 3000)
        ],
        "parameter_bounds": {
            "boundaries": {
                "well_design#INJ#md": [0.0, 3000.0],
                "well_design#PRO#md": [0.0, 3000.0],
            },
            "A": [{"INJ.md": 1.0, "PRO.md": 1.0}],
            "b": [3000.0],
            "sense": ["<="],
        },
    }

    service.process_request(request)
    best = service.global_best_control_vector.items
    assert best["well_design#INJ#md"] + best["well_design#PRO#md"] <= 3000.0 + 1e-6


def test_direction_greater_equal_transforms_and_enforced():
    """Test that a >= constraint is transformed and enforced (e.g., INJ.md + PRO.md >= 1000)."""
    service = SolutionUpdaterService(
        optimization_engine=OptimizationEngine.PSO,
        max_generations=3,
        patience=3,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
    )
    request = {
        "solution_candidates": [
            _build_candidate(
                {"well_design#INJ#md": 100.0, "well_design#PRO#md": 100.0}, 5.0
            ),  # violates >= 1000
            _build_candidate(
                {"well_design#INJ#md": 600.0, "well_design#PRO#md": 600.0}, 6.0
            ),  # satisfies
        ],
        "parameter_bounds": {
            "boundaries": {
                "well_design#INJ#md": [0.0, 3000.0],
                "well_design#PRO#md": [0.0, 3000.0],
            },
            "A": [{"INJ.md": 1.0, "PRO.md": 1.0}],
            "b": [1000.0],
            "sense": [">="],
        },
    }
    for _ in range(2):
        resp = service.process_request(request)
        request["solution_candidates"] = [
            {
                "control_vector": {"items": cv.items},
                "cost_function_results": {"values": {"metric1": 5.0}},
            }
            for cv in resp.next_iter_solutions
        ]
    for cv in resp.next_iter_solutions:
        assert (
            cv.items["well_design#INJ#md"] + cv.items["well_design#PRO#md"]
            >= 1000.0 - 1e-6
        )


def test_strict_less_treated_as_less_equal():
    """Ensure '<' behaves like '<=' (INJ.md + PRO.md < 1200)."""
    service = SolutionUpdaterService(
        optimization_engine=OptimizationEngine.PSO,
        max_generations=2,
        patience=2,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
    )
    request = {
        "solution_candidates": [
            _build_candidate(
                {"well_design#INJ#md": 800.0, "well_design#PRO#md": 600.0}, 5.0
            ),  # violates <1200
            _build_candidate(
                {"well_design#INJ#md": 400.0, "well_design#PRO#md": 400.0}, 6.0
            ),  # satisfies
        ],
        "parameter_bounds": {
            "boundaries": {
                "well_design#INJ#md": [0.0, 3000.0],
                "well_design#PRO#md": [0.0, 3000.0],
            },
            "A": [{"INJ.md": 1.0, "PRO.md": 1.0}],
            "b": [1200.0],
            "sense": ["<"],
        },
    }
    resp = service.process_request(request)
    for cv in resp.next_iter_solutions:
        assert (
            cv.items["well_design#INJ#md"] + cv.items["well_design#PRO#md"]
            <= 1200.0 + 1e-6
        )


def test_mixed_constraints_greater_and_less_than():
    """Test that both >= and <= constraints are enforced correctly."""
    service = SolutionUpdaterService(
        optimization_engine=OptimizationEngine.PSO,
        max_generations=5,
        patience=5,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
    )

    request = {
        "solution_candidates": [
            _build_candidate(
                {"well_design#INJ#md": 500.0, "well_design#PRO#md": 400.0}, 10.0
            ),
            _build_candidate(
                {"well_design#INJ#md": 2000.0, "well_design#PRO#md": 1500.0}, 9.0
            ),
        ],
        "parameter_bounds": {
            "boundaries": {
                "well_design#INJ#md": [0.0, 3000.0],
                "well_design#PRO#md": [0.0, 3000.0],
            },
            "A": [
                {"INJ.md": 1.0, "PRO.md": 1.0},
                {"INJ.md": 1.0, "PRO.md": 1.0},
            ],
            "b": [1000.0, 3000.0],
            "sense": [">=", "<="],
        },
    }

    for _ in range(5):
        resp = service.process_request(request)
        request["solution_candidates"] = [
            {
                "control_vector": {"items": cv.items},
                "cost_function_results": {"values": {"metric1": 10.0}},
            }
            for cv in resp.next_iter_solutions
        ]

    for cv in resp.next_iter_solutions:
        inj_md = cv.items["well_design#INJ#md"]
        pro_md = cv.items["well_design#PRO#md"]
        assert inj_md + pro_md >= 1000.0 - 1e-6
        assert inj_md + pro_md <= 3000.0 + 1e-6


def test_multiple_inequalities_enforced():
    """Test that multiple inequality constraints are enforced simultaneously."""
    service = SolutionUpdaterService(
        optimization_engine=OptimizationEngine.PSO,
        max_generations=5,
        patience=5,
        objectives={"metric1": OptimizationStrategy.MINIMIZE},
    )

    request = {
        "solution_candidates": [
            _build_candidate(
                {"well_design#INJ#md": 1000.0, "well_design#PRO#md": 1000.0}, 10.0
            ),
            _build_candidate(
                {"well_design#INJ#md": 500.0, "well_design#PRO#md": 200.0}, 9.0
            ),
        ],
        "parameter_bounds": {
            "boundaries": {
                "well_design#INJ#md": [0.0, 3000.0],
                "well_design#PRO#md": [0.0, 3000.0],
            },
            "A": [
                {"INJ.md": 1.0},
                {"PRO.md": 1.0},
            ],
            "b": [800.0, 800.0],
            "sense": [">=", ">="],
        },
    }

    for _ in range(5):
        resp = service.process_request(request)
        request["solution_candidates"] = [
            {
                "control_vector": {"items": cv.items},
                "cost_function_results": {"values": {"metric1": 10.0}},
            }
            for cv in resp.next_iter_solutions
        ]

    for cv in resp.next_iter_solutions:
        assert cv.items["well_design#INJ#md"] >= 800.0 - 1e-6
        assert cv.items["well_design#PRO#md"] >= 800.0 - 1e-6
