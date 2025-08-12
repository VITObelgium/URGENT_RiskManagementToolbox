# tests/test_dispatcher.py

import pytest
from pydantic import ValidationError

from services.problem_dispatcher_service import ProblemDispatcherService
from services.problem_dispatcher_service.core.models import (
    ControlVector,
    ProblemDispatcherServiceResponse,
    ServiceType,
    SolutionCandidateServicesTasks,
)


@pytest.fixture
def dict_problem_definition():
    return {
        "well_placement": [
            {
                "well_name": "W1",
                "initial_state": {
                    "well_type": "IWell",
                    "wellhead": {"x": 0, "y": 0, "z": 0},
                    "md": 200,
                    "perforations": [{"start_md": 100.0, "end_md": 200.0}],
                },
                "optimization_constrains": {
                    "wellhead": {
                        "x": {"lb": 0, "ub": 100},
                        "y": {"lb": 10, "ub": 200},
                    },
                    "md": {"lb": 0, "ub": 100},
                },
            },
            {
                "well_name": "W2",
                "initial_state": {
                    "well_type": "IWell",
                    "wellhead": {"x": 10, "y": 10, "z": 0},
                    "md": 200,
                    "perforations": [{"start_md": 100.0, "end_md": 200.0}],
                },
                "optimization_constrains": {
                    "wellhead": {"x": {"lb": 0, "ub": 100}, "y": {"lb": 10, "ub": 200}}
                },
            },
        ],
        "optimization_parameters": {"optimization_strategy": "maximize"},
    }


@pytest.mark.parametrize("n_size", [1, 3])
def test_handle_initial_request_returns_expected_structure(
    dict_problem_definition, n_size
):
    service = ProblemDispatcherService(
        problem_definition=dict_problem_definition, n_size=n_size
    )
    response = service.process_iteration()

    assert isinstance(response, ProblemDispatcherServiceResponse)
    assert len(response.solution_candidates) == n_size

    for candidate in response.solution_candidates:
        assert isinstance(candidate, SolutionCandidateServicesTasks)
        assert ServiceType.WellManagementService in candidate.tasks


@pytest.mark.parametrize(
    "control_vector_items",
    [
        [{"well_placement#W1#wellhead#x": 5.0, "well_placement#W1#md": 100.0}],
        [{"well_placement#W1#md": 90.0, "well_placement#W2#wellhead#y": 100.0}],
        [{"well_placement#W1#wellhead#x": 50.0, "well_placement#W2#wellhead#x": 10.0}],
    ],
)
def test_handle_iteration_loop_valid_input(
    dict_problem_definition, control_vector_items
):
    service = ProblemDispatcherService(problem_definition=dict_problem_definition)
    control_vectors = [ControlVector(items=items) for items in control_vector_items]
    response = service.process_iteration(control_vectors)

    assert isinstance(response, ProblemDispatcherServiceResponse)
    assert len(response.solution_candidates) == len(control_vector_items)

    for candidate in response.solution_candidates:
        assert ServiceType.WellManagementService in candidate.tasks


@pytest.mark.parametrize(
    "control_vector_items, expected_updates",
    [
        (
            {"well_placement#W1#md": 75.0, "well_placement#W2#md": 155.0},
            {"W1.md": 75.0, "W2.md": 155.0},
        ),
        (
            {
                "well_placement#W1#wellhead#x": 42.0,
                "well_placement#W2#wellhead#x": 55.5,
            },
            {"W1.wellhead.x": 42.0, "W2.wellhead.x": 55.5},
        ),
        (
            {
                "well_placement#W1#md": 50.0,
                "well_placement#W2#md": 250.0,
                "well_placement#W1#wellhead#y": 88.8,
                "well_placement#W2#wellhead#y": 199.9,
            },
            {
                "W1.md": 50.0,
                "W2.md": 250.0,
                "W1.wellhead.y": 88.8,
                "W2.wellhead.y": 199.9,
            },
        ),
    ],
)
def test_control_vector_multiple_wells(
    dict_problem_definition, control_vector_items, expected_updates
):
    service = ProblemDispatcherService(problem_definition=dict_problem_definition)
    control_vector = ControlVector(items=control_vector_items)
    response = service.process_iteration([control_vector])

    assert isinstance(response, ProblemDispatcherServiceResponse)
    assert len(response.solution_candidates) == 1

    tasks: SolutionCandidateServicesTasks = response.solution_candidates[0]
    assert ServiceType.WellManagementService in tasks.tasks

    service_tasks = tasks.tasks[ServiceType.WellManagementService]
    tasks_by_well = {task["name"]: task for task in service_tasks.request}

    for key_path, expected_value in expected_updates.items():
        well_name, *nested_keys = key_path.split(".")
        assert well_name in tasks_by_well
        task = tasks_by_well[well_name]
        value = task
        for key in nested_keys:
            assert key in value
            value = value[key]
        assert value == expected_value


def test_invalid_problem_definition_type_raises_validation_error():
    with pytest.raises(ValidationError):
        ProblemDispatcherService(problem_definition={"well_placement": "not-a-list"})


def test_empty_problem_definition_raises_validation_error():
    with pytest.raises(ValidationError):
        ProblemDispatcherService(problem_definition={})


def test_handle_iteration_loop_empty_vector_returns_candidates(dict_problem_definition):
    service = ProblemDispatcherService(problem_definition=dict_problem_definition)
    control_vectors = [ControlVector(items={})]
    response = service.process_iteration(control_vectors)
    assert isinstance(response, ProblemDispatcherServiceResponse)
    assert len(response.solution_candidates) == 1


def test_handle_iteration_loop_with_invalid_float_value(dict_problem_definition):
    service = ProblemDispatcherService(problem_definition=dict_problem_definition)
    control_vectors = [ControlVector(items={"well_placement#W1#md": 75.5})]
    response = service.process_iteration(control_vectors)
    assert isinstance(response, ProblemDispatcherServiceResponse)
    assert len(response.solution_candidates) == 1


def test_get_linear_inequalities(dict_problem_definition):
    dict_problem_definition["optimization_parameters"]["linear_inequalities"] = {
        "A": [{"W1.md": 1, "W2.md": 1}],
        "b": [100],
        "sense": ["<="],
    }
    service = ProblemDispatcherService(problem_definition=dict_problem_definition)
    inequalities = service.get_linear_inequalities()
    assert inequalities is not None
    assert "A" in inequalities
    assert "b" in inequalities


def test_validate_total_md_len(dict_problem_definition):
    dict_problem_definition["optimization_parameters"]["total_md_len"] = {
        "lb": 100,
        "ub": 5000,
    }
    service = ProblemDispatcherService(problem_definition=dict_problem_definition)
    params = service._problem_definition.optimization_parameters
    assert params.total_md_len.ub < 5000


def test_process_iteration_exception_handling(dict_problem_definition, monkeypatch):
    def mock_build(*args, **kwargs):
        raise ValueError("Test Exception")

    monkeypatch.setattr(
        "services.problem_dispatcher_service.core.builder.TaskBuilder.build", mock_build
    )
    service = ProblemDispatcherService(problem_definition=dict_problem_definition)
    with pytest.raises(ValueError, match="Test Exception"):
        service.process_iteration()


def test_initialization_failure(dict_problem_definition):
    dict_problem_definition["well_placement"][0]["optimization_constrains"] = "invalid"
    with pytest.raises(ValidationError):
        ProblemDispatcherService(problem_definition=dict_problem_definition)
