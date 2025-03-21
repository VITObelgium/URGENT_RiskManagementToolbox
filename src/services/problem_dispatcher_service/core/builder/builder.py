from typing import Any

from services.problem_dispatcher_service.core.models import (
    ControlVector,
    RequestPayload,
    ServiceType,
    SolutionCandidateServicesTasks,
)
from services.problem_dispatcher_service.core.service import ProblemTypeHandler
from services.problem_dispatcher_service.core.utils import (
    parse_flat_dict_to_nested,
    update_initial_state,
)


class TaskBuilder:
    """
    A builder class responsible for constructing tasks for solution candidates based on
    provided control vectors and a predefined initial state.

    Attributes:
        initial_state (dict[str, Any]): The initial state dictionary, used as the base for updates
            from control vectors.
        handlers (dict[str, ProblemTypeHandler]): Handlers responsible for processing problem types.
        service_type_map (dict[str, ServiceType]): A mapping of problem types to their corresponding
            service types.
    """

    def __init__(
        self,
        initial_state: dict[str, Any],
        handlers: dict[str, ProblemTypeHandler],
        service_type_map: dict[str, ServiceType],
    ):
        """
        Initializes the TaskBuilder with the initial state, handlers, and service type mapping.

        Args:
            initial_state (dict[str, Any]): The initial state dictionary used to build solutions.
            handlers (dict[str, ProblemTypeHandler]): A dictionary mapping problem type strings
                to their respective handlers.
            service_type_map (dict[str, ServiceType]): A dictionary mapping problem type strings
                to their corresponding service types.
        """
        self.initial_state = initial_state
        self.handlers = handlers
        self.service_type_map = service_type_map

    def build(
        self, control_vectors: list[dict[str, float]]
    ) -> list[SolutionCandidateServicesTasks]:
        """
        Constructs a list of `SolutionCandidateServicesTasks` based on the provided control vectors.

        For each control vector:
            - Parses it into a nested structure and updates the initial state accordingly.
            - Constructs a `ControlVector` object and generates a map of service tasks
              for each problem type.
            - Assembles the tasks into a `SolutionCandidateServicesTasks` object.

        Args:
            control_vectors (list[dict[str, float]]): A list of control vectors where each
                control vector is represented as a dictionary of key-value pairs.

        Returns:
            list[SolutionCandidateServicesTasks]: A list of tasks, each representing solutions
            derived from the provided control vectors along with their service type mappings.
        """
        tasks_list: list[SolutionCandidateServicesTasks] = []

        for cv_dict in control_vectors:
            nested_updates = parse_flat_dict_to_nested(cv_dict)
            updated_state = update_initial_state(self.initial_state, nested_updates)
            control_vector = ControlVector(items=cv_dict)

            task_map: dict[ServiceType, RequestPayload] = {}
            for problem_type, solution_items in updated_state.items():
                handler = self.handlers.get(problem_type)
                service_type = self.service_type_map.get(problem_type)
                if handler and service_type:
                    service_requests = handler.build_service_tasks(solution_items)
                    task_map[service_type] = RequestPayload(
                        request=service_requests, control_vector=control_vector
                    )

            tasks_list.append(SolutionCandidateServicesTasks(tasks=task_map))
        return tasks_list
