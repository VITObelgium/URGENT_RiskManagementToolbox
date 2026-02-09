import pytest

from common import OptimizationStrategy
from services.solution_updater_service.core.models import OptimizationEngine
from services.solution_updater_service.core.service import SolutionUpdaterService


@pytest.fixture
def mixed_constraints_request():
    """
    Test with population where particles violate one, the other, or both constraints.
    """
    return {
        "solution_candidates": [
            {
                "control_vector": {
                    "items": {
                        "well_design#INJ#md": 2800.0,  # Violates ub
                        "well_design#INJ#wellhead#x": 150.0,
                    }
                },
                "cost_function_results": {"values": {"metric": 10.0}},
            },
            {
                "control_vector": {
                    "items": {
                        "well_design#INJ#md": 2500.0,
                        "well_design#INJ#wellhead#x": 5.0,  # Violates lb
                    }
                },
                "cost_function_results": {"values": {"metric": 12.0}},
            },
            {
                "control_vector": {
                    "items": {
                        "well_design#INJ#md": 1900.0,  # Violates lb
                        "well_design#INJ#wellhead#x": 3200.0,  # Violates ub
                    }
                },
                "cost_function_results": {"values": {"metric": 9.0}},
            },
            {
                "control_vector": {
                    "items": {
                        "well_design#INJ#md": 2500.0,
                        "well_design#INJ#wellhead#x": 1500.0,  # Feasible
                    }
                },
                "cost_function_results": {"values": {"metric": 15.0}},
            },
        ],
        "optimization_constrains": {
            "boundaries": {
                "well_design#INJ#md": {"lb": 2000.0, "ub": 2700.0},
                "well_design#INJ#wellhead#x": {"lb": 10.0, "ub": 3190.0},
            }
        },
    }


def test_pso_handles_mixed_md_and_wellhead_constraints(mixed_constraints_request):
    """
    Verify that PSO correctly handles and repairs a population with mixed
    and independent violations of 'md' and 'wellhead.x' bounds.
    """
    service = SolutionUpdaterService(
        optimization_engine=OptimizationEngine.PSO,
        max_generations=2,
        max_stall_generations=3,
        objectives={"metric": OptimizationStrategy.MAXIMIZE},
    )

    response = service.process_request(mixed_constraints_request)

    md_bounds = mixed_constraints_request["optimization_constrains"]["boundaries"][
        "well_design#INJ#md"
    ]
    x_bounds = mixed_constraints_request["optimization_constrains"]["boundaries"][
        "well_design#INJ#wellhead#x"
    ]

    for solution in response.next_iter_solutions:
        md_val = solution.items["well_design#INJ#md"]
        x_val = solution.items["well_design#INJ#wellhead#x"]

        assert md_bounds["lb"] <= md_val <= md_bounds["ub"]
        assert x_bounds["lb"] <= x_val <= x_bounds["ub"]
