"""Dispatcher behaviour when several domain services are configured at once."""

import pytest

from services.problem_dispatcher_service import ProblemDispatcherService
from services.problem_dispatcher_service.core.models import ProblemDispatcherDefinition
from services.solution_updater_service import ControlVector


@pytest.fixture
def multi_domain_problem_definition():
    return {
        "domain_services": {
            "well_design": [
                {
                    "name": "INJ",
                    "initial_state": {
                        "well_type": "IWell",
                        "wellhead": {"x": 50, "y": 50, "z": 0},
                        "md": 2500,
                    },
                    "parameter_bounds": {
                        "wellhead": {"x": {"lb": 10, "ub": 3190}},
                        "md": {"lb": 2000, "ub": 2700},
                    },
                }
            ],
            "well_counter": [
                {
                    "name": "field",
                    "initial_state": {"count": 2},
                    "parameter_bounds": {"count": {"lb": 1, "ub": 5}},
                }
            ],
        },
        "optimization_parameters": {
            "objectives": {"Heat": "maximize"},
            "population_size": 4,
            "linear_inequalities": {
                "A": [{"well_design.INJ.md": 1.0, "well_counter.field.count": 100.0}],
                "b": [3200.0],
                "sense": ["<="],
            },
        },
        "plugins": {
            "connector": "opendarts",
            "optimizer": "pso",
            "domain_services": ["well_design", "well_counter"],
        },
    }


def test_process_iteration_produces_tasks_for_both_services(
    multi_domain_problem_definition,
):
    problem_definition = ProblemDispatcherDefinition.model_validate(
        multi_domain_problem_definition
    )
    service = ProblemDispatcherService(problem_definition=problem_definition)
    response = service.process_iteration()

    assert len(response.solution_candidates) == 4
    for candidate in response.solution_candidates:
        assert set(candidate.tasks) == {"well_design", "well_counter"}
        design_task = candidate.tasks["well_design"]
        counter_task = candidate.tasks["well_counter"]
        assert design_task.request[0]["name"] == "INJ"
        assert counter_task.request[0]["name"] == "field"
        # Both tasks carry the same full control vector for the candidate.
        assert design_task.control_vector == counter_task.control_vector


def test_boundaries_carry_both_service_prefixes(multi_domain_problem_definition):
    problem_definition = ProblemDispatcherDefinition.model_validate(
        multi_domain_problem_definition
    )
    service = ProblemDispatcherService(problem_definition=problem_definition)
    boundaries = service.full_key_boundaries

    assert set(boundaries) == {
        "well_design#INJ#wellhead#x",
        "well_design#INJ#md",
        "well_counter#field#count",
    }
    assert boundaries["well_counter#field#count"].lb == 1.0
    assert boundaries["well_counter#field#count"].ub == 5.0


def test_cross_domain_linear_inequality_converts_to_hash_keys(
    multi_domain_problem_definition,
):
    problem_definition = ProblemDispatcherDefinition.model_validate(
        multi_domain_problem_definition
    )
    service = ProblemDispatcherService(problem_definition=problem_definition)
    inequalities = service.full_key_linear_inequalities

    assert inequalities is not None
    assert inequalities.A == [
        {"well_design#INJ#md": 1.0, "well_counter#field#count": 100.0}
    ]
    assert inequalities.b == [3200.0]
    assert inequalities.sense == ["<="]


def test_initial_population_satisfies_cross_domain_inequality(
    multi_domain_problem_definition,
):
    problem_definition = ProblemDispatcherDefinition.model_validate(
        multi_domain_problem_definition
    )
    service = ProblemDispatcherService(problem_definition=problem_definition)
    response = service.process_iteration()

    for candidate in response.solution_candidates:
        cv = candidate.tasks["well_design"].control_vector.items
        md = cv["well_design#INJ#md"]
        count = cv["well_counter#field#count"]
        assert 2000.0 <= md <= 2700.0
        assert 1.0 <= count <= 5.0
        assert md + 100.0 * count <= 3200.0 + 1e-6


def test_control_vector_updates_route_to_owning_service(
    multi_domain_problem_definition,
):
    problem_definition = ProblemDispatcherDefinition.model_validate(
        multi_domain_problem_definition
    )
    service = ProblemDispatcherService(problem_definition=problem_definition)
    control_vector = ControlVector(
        items={"well_design#INJ#md": 2100.0, "well_counter#field#count": 4.0}
    )
    response = service.process_iteration([control_vector])

    assert len(response.solution_candidates) == 1
    tasks = response.solution_candidates[0].tasks
    assert tasks["well_design"].request[0]["md"] == 2100.0
    assert tasks["well_counter"].request[0]["count"] == 4.0
