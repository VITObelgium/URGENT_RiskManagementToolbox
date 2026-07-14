from services.problem_dispatcher_service.core.builder import TaskBuilder
from services.solution_updater_service import ControlVector


def test_task_builder_creates_tasks_with_control_vector():
    initial_state = {
        "well_design": {
            "W1": {"name": "W1", "md": 100},
        }
    }
    task_builder = TaskBuilder(initial_state)

    control_vectors = [{"well_design#W1#md": 150}]
    result = task_builder.build(control_vectors)

    assert len(result) == 1
    payload = result[0].tasks["well_design"]
    assert isinstance(payload.control_vector, ControlVector)
    assert payload.request[0]["md"] == 150


def test_task_builder_builds_one_task_per_domain_service():
    initial_state = {
        "well_design": {
            "W1": {"name": "W1", "md": 100},
        },
        "well_counter": {
            "field": {"name": "field", "count": 2},
        },
    }
    task_builder = TaskBuilder(initial_state)

    control_vectors = [{"well_design#W1#md": 150, "well_counter#field#count": 3}]
    result = task_builder.build(control_vectors)

    assert len(result) == 1
    tasks = result[0].tasks
    assert set(tasks) == {"well_design", "well_counter"}
    assert tasks["well_design"].request[0]["md"] == 150
    assert tasks["well_counter"].request[0]["count"] == 3
    # Each task carries the full control vector of its candidate.
    assert tasks["well_design"].control_vector == tasks["well_counter"].control_vector
