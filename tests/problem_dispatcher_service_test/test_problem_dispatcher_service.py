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
                "optimization_constraints": {
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
                "optimization_constraints": {
                    "wellhead": {"x": {"lb": 0, "ub": 100}, "y": {"lb": 10, "ub": 200}},
                    "md": {"lb": 0, "ub": 300},
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


def test_linear_inequalities_nested_attribute_valid(dict_problem_definition):
    dict_problem_definition["optimization_parameters"]["linear_inequalities"] = {
        "A": [{"W1.wellhead.x": 1, "W2.wellhead.x": 1}],
        "b": [200],
        "sense": ["<="],
    }
    service = ProblemDispatcherService(problem_definition=dict_problem_definition)
    inequalities = service.get_linear_inequalities()
    assert inequalities is not None


def test_linear_inequalities_missing_well_constraint_raises():
    pd = {
        "well_placement": [
            {
                "well_name": "W1",
                "initial_state": {
                    "well_type": "IWell",
                    "wellhead": {"x": 0, "y": 0, "z": 0},
                    "md": 200,
                    "perforations": [{"start_md": 100.0, "end_md": 200.0}],
                },
                "optimization_constraints": {"md": {"lb": 0, "ub": 300}},
            },
            {
                "well_name": "W2",
                "initial_state": {
                    "well_type": "IWell",
                    "wellhead": {"x": 10, "y": 10, "z": 0},
                    "md": 200,
                    "perforations": [{"start_md": 100.0, "end_md": 200.0}],
                },
                "optimization_constraints": None,
            },
        ],
        "optimization_parameters": {
            "optimization_strategy": "maximize",
            "linear_inequalities": {
                "A": [{"W1.md": 1, "W2.md": 1}],
                "b": [100],
            },
        },
    }

    with pytest.raises(ValidationError):
        ProblemDispatcherService(problem_definition=pd)


def test_linear_inequalities_missing_variable_in_constraints_raises():
    # W1 has optimization_constraints but missing 'md' entry required by inequalities
    pd = {
        "well_placement": [
            {
                "well_name": "W1",
                "initial_state": {
                    "well_type": "IWell",
                    "wellhead": {"x": 0, "y": 0, "z": 0},
                    "md": 200,
                    "perforations": [{"start_md": 100.0, "end_md": 200.0}],
                },
                "optimization_constraints": {"wellhead": {"x": {"lb": 0, "ub": 100}}},
            }
        ],
        "optimization_parameters": {
            "optimization_strategy": "maximize",
            "linear_inequalities": {"A": [{"W1.md": 1}], "b": [100]},
        },
    }

    with pytest.raises(ValidationError):
        ProblemDispatcherService(problem_definition=pd)


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
    dict_problem_definition["well_placement"][0]["optimization_constraints"] = "invalid"
    with pytest.raises(ValidationError):
        ProblemDispatcherService(problem_definition=dict_problem_definition)


def test_problem_dispatcher_service_initializes_correct_population(
    dict_problem_definition,
):
    service = ProblemDispatcherService(problem_definition=dict_problem_definition)
    assert service._constraints
    response = service.process_iteration()
    assert len(response.solution_candidates) > 0

    # Extract md bounds from the fixture for W1
    w1_md_bounds = dict_problem_definition["well_placement"][0][
        "optimization_constraints"
    ]["md"]
    lb = w1_md_bounds["lb"]
    ub = w1_md_bounds["ub"]

    for candidate in response.solution_candidates:
        # Each candidate should include a WellManagementService task with the ControlVector
        assert ServiceType.WellManagementService in candidate.tasks
        payload = candidate.tasks[ServiceType.WellManagementService]
        control_items = payload.control_vector.items

        assert "well_placement#W1#md" in control_items
        md_val = control_items["well_placement#W1#md"]
        assert lb <= md_val <= ub


def test_linear_inequalities_unknown_well_raises():
    pd = {
        "well_placement": [
            {
                "well_name": "INJ",
                "initial_state": {
                    "well_type": "IWell",
                    "wellhead": {"x": 0, "y": 0, "z": 0},
                    "md": 200,
                    "perforations": [{"start_md": 100.0, "end_md": 200.0}],
                },
                "optimization_constraints": {
                    "wellhead": {
                        "x": {"lb": 0, "ub": 100},
                        "y": {"lb": 0, "ub": 100},
                    },
                    "md": {"lb": 0, "ub": 300},
                },
            },
            {
                "well_name": "PRO",
                "initial_state": {
                    "well_type": "IWell",
                    "wellhead": {"x": 0, "y": 0, "z": 0},
                    "md": 200,
                    "perforations": [{"start_md": 100.0, "end_md": 200.0}],
                },
                "optimization_constraints": {
                    "wellhead": {
                        "x": {"lb": 0, "ub": 100},
                        "y": {"lb": 0, "ub": 100},
                    },
                    "md": {"lb": 0, "ub": 300},
                },
            },
        ],
        "optimization_parameters": {
            "optimization_strategy": "maximize",
            "linear_inequalities": {
                "A": [{"Brent.md": 1.0, "PRO.md": 1.0}, {"INJ.md": 1.0, "PRO.md": 1.0}],
                "b": [2100.0, 5000.0],
                "sense": [">=", "<="],
            },
        },
    }
    with pytest.raises((ValidationError, TypeError)):
        ProblemDispatcherService(problem_definition=pd)


def test_linear_inequalities_A_b_length_mismatch_raises(dict_problem_definition):
    dict_problem_definition["optimization_parameters"]["linear_inequalities"] = {
        "A": [{"W1.md": 1, "W2.md": 1}, {"W1.md": -1, "W2.md": 1}],
        "b": [100],
    }
    with pytest.raises((ValidationError, TypeError)):
        ProblemDispatcherService(problem_definition=dict_problem_definition)


def test_linear_inequalities_invalid_sense_length_raises(dict_problem_definition):
    dict_problem_definition["optimization_parameters"]["linear_inequalities"] = {
        "A": [{"W1.md": 1, "W2.md": 1}],
        "b": [100],
        "sense": ["<=", ">="],
    }
    with pytest.raises((ValidationError, TypeError)):
        ProblemDispatcherService(problem_definition=dict_problem_definition)


def test_linear_inequalities_variable_without_dot_raises(dict_problem_definition):
    dict_problem_definition["optimization_parameters"]["linear_inequalities"] = {
        "A": [{"W1md": 1}],
        "b": [100],
    }
    with pytest.raises((ValidationError, TypeError)):
        ProblemDispatcherService(problem_definition=dict_problem_definition)


def test_linear_inequalities_inconsistent_attribute_suffix_raises(
    dict_problem_definition,
):
    dict_problem_definition["optimization_parameters"]["linear_inequalities"] = {
        "A": [{"W1.md": 1, "W2.wellhead.x": 1}],
        "b": [100],
    }
    with pytest.raises((ValidationError, TypeError)):
        ProblemDispatcherService(problem_definition=dict_problem_definition)


def test_linear_inequalities_nested_missing_leaf_key_raises(dict_problem_definition):
    # Constraints only define wellhead.x/y, but A references wellhead.z
    dict_problem_definition["optimization_parameters"]["linear_inequalities"] = {
        "A": [{"W1.wellhead.z": 1}],
        "b": [10],
    }
    with pytest.raises((ValidationError, TypeError)):
        ProblemDispatcherService(problem_definition=dict_problem_definition)


def test_duplicate_well_names_raises():
    pd = {
        "well_placement": [
            {
                "well_name": "W1",
                "initial_state": {
                    "well_type": "IWell",
                    "wellhead": {"x": 0, "y": 0, "z": 0},
                    "md": 100,
                    "perforations": [{"start_md": 20.0, "end_md": 80.0}],
                },
                "optimization_constraints": {"md": {"lb": 0, "ub": 200}},
            },
            {
                "well_name": "W1",
                "initial_state": {
                    "well_type": "IWell",
                    "wellhead": {"x": 10, "y": 10, "z": 0},
                    "md": 150,
                    "perforations": [{"start_md": 50.0, "end_md": 140.0}],
                },
                "optimization_constraints": {"md": {"lb": 0, "ub": 300}},
            },
        ],
        "optimization_parameters": {"optimization_strategy": "maximize"},
    }
    with pytest.raises((ValidationError, TypeError)):
        ProblemDispatcherService(problem_definition=pd)


def test_variable_bounds_lb_gt_ub_raises(dict_problem_definition):
    dict_problem_definition["well_placement"][0]["optimization_constraints"]["md"] = {
        "lb": 100,
        "ub": 50,
    }
    with pytest.raises((ValidationError, TypeError)):
        ProblemDispatcherService(problem_definition=dict_problem_definition)


def test_linear_inequalities_non_numeric_values_raises(dict_problem_definition):
    dict_problem_definition["optimization_parameters"]["linear_inequalities"] = {
        "A": [{"W1.md": "one"}],
        "b": ["hundred"],
    }
    with pytest.raises((ValidationError, TypeError)):
        ProblemDispatcherService(problem_definition=dict_problem_definition)


def test_linear_inequalities_invalid_sense_symbol_raises(dict_problem_definition):
    dict_problem_definition["optimization_parameters"]["linear_inequalities"] = {
        "A": [{"W1.md": 1}],
        "b": [100],
        "sense": ["!="],
    }
    with pytest.raises(ValidationError):
        ProblemDispatcherService(problem_definition=dict_problem_definition)


@pytest.mark.parametrize(
    "well_config",
    [
        # JWell
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
                "azimuth": {"lb": 0, "ub": 360},
            },
        },
        # SWell
        {
            "well_name": "S1",
            "initial_state": {
                "well_type": "SWell",
                "wellhead": {"x": 0, "y": 0, "z": 0},
                "md_linear1": 400.0,
                "md_curved1": 200.0,
                "dls1": 10.0,
                "md_linear2": 500.0,
                "md_curved2": 300.0,
                "dls2": 20.0,
                "md_linear3": 600.0,
                "azimuth": 180.0,
                "md_step": 30.0,
                "perforations": [{"start_md": 100.0, "end_md": 200.0}],
            },
            "optimization_constraints": {
                "md_linear1": {"lb": 300, "ub": 500},
                "dls1": {"lb": 5, "ub": 15},
            },
        },
        # HWell
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
                "TVD": {"lb": 800, "ub": 1200},
                "md_lateral": {"lb": 1000, "ub": 2000},
            },
        },
    ],
)
def test_problem_dispatcher_supports_extended_well_types(well_config):
    problem_definition = {
        "well_placement": [well_config],
        "optimization_parameters": {"optimization_strategy": "maximize"},
    }
    service = ProblemDispatcherService(problem_definition=problem_definition)
    response = service.process_iteration()

    assert isinstance(response, ProblemDispatcherServiceResponse)
    assert len(response.solution_candidates) > 0

    candidate = response.solution_candidates[0]
    assert ServiceType.WellManagementService in candidate.tasks

    # Verify control vector keys match constraints
    task = candidate.tasks[ServiceType.WellManagementService]
    control_items = task.control_vector.items

    well_name = well_config["well_name"]
    for param in well_config["optimization_constraints"]:
        key = f"well_placement#{well_name}#{param}"
        assert key in control_items


def test_hwell_validation_failure_invalid_geometry():
    # HWell with invalid dimensions (TVD too small for curvature)
    # _CURVATURE_RADIUS approx 430m for default params
    well_config = {
        "well_name": "H1_Invalid",
        "initial_state": {
            "well_type": "HWell",
            "wellhead": {"x": 0, "y": 0, "z": 0},
            "TVD": 100.0,  # Too shallow
            "md_lateral": 1500.0,
            "azimuth": 45.0,
            "md_step": 10.0,
            "perforations": [{"start_md": 100.0, "end_md": 200.0}],
        },
        "optimization_constraints": {"TVD": {"lb": 50, "ub": 150}},
    }
    problem_definition = {
        "well_placement": [well_config],
        "optimization_parameters": {"optimization_strategy": "maximize"},
    }

    with pytest.raises(ValidationError) as excinfo:
        ProblemDispatcherService(problem_definition=problem_definition)

    # The ValueError from HWellModel is wrapped in Pydantic's ValidationError
    assert (
        "Horizontal well true total depth is less than curved well section radius"
        in str(excinfo.value)
    )
