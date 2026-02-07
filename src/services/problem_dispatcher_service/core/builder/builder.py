from typing import Any

from services.problem_dispatcher_service.core.models import (
    RequestPayload,
    ServiceType,
    SolutionCandidateServicesTasks,
)
from services.problem_dispatcher_service.core.service import ProblemTypeHandler
from services.problem_dispatcher_service.core.utils import (
    parse_flat_dict_to_nested,
    update_initial_state,
)
from services.solution_updater_service import ControlVector


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
        handlers: dict[ServiceType, ProblemTypeHandler],
    ):
        """
        Initializes the TaskBuilder with the initial state, handlers, and service type mapping.

        Args:
            initial_state (dict[str, Any]): The initial state dictionary used to build solutions.
            handlers (dict[str, ProblemTypeHandler]): A dictionary mapping problem type strings
                to their respective handlers.
        """
        self.initial_state: dict[str, Any] = initial_state
        self.handlers: dict[ServiceType, ProblemTypeHandler] = handlers

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
            for service_type, solution_items in updated_state.items():
                service_type = ServiceType(service_type)  # safeguard
                handler = self.handlers.get(service_type)
                if handler and service_type:
                    service_requests = handler.build_service_tasks(solution_items)
                    task_map[service_type] = RequestPayload(
                        request=service_requests, control_vector=control_vector
                    )
                else:
                    raise ValueError(
                        f"Invalid service type: {service_type} or handler missing."
                    )

            tasks_list.append(SolutionCandidateServicesTasks(tasks=task_map))
        return tasks_list
