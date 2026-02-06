from services.problem_dispatcher_service.core.builder import TaskBuilder
from services.problem_dispatcher_service.core.models import ServiceType
from services.solution_updater_service import ControlVector


def test_task_builder_creates_tasks_with_control_vector():
    class MockHandler:
        def build_service_tasks(self, items):
            return [{"name": "W1", "md": items["md"]}]

    handlers = {"mock_problem": MockHandler()}
    service_type_map = {"mock_problem": ServiceType.WellDesignService}

    initial_state = {"mock_problem": {"md": 100}}
    task_builder = TaskBuilder(initial_state, handlers, service_type_map)

    control_vectors = [{"mock_problem#md": 150}]
    result = task_builder.build(control_vectors)

    assert len(result) == 1
    payload = result[0].tasks[ServiceType.WellDesignService]
    assert isinstance(payload.control_vector, ControlVector)
    assert payload.request[0]["md"] == 150
