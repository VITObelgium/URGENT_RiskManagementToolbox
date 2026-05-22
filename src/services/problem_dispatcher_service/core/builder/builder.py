from typing import Any

from services.problem_dispatcher_service.core.models import (
    RequestPayload,
    ServiceType,
    SolutionCandidateServicesTasks,
)
from services.problem_dispatcher_service.core.utils import (
    parse_flat_dict_to_nested,
    update_initial_state,
)
from services.solution_updater_service import ControlVector


class TaskBuilder:
    """Converts control vectors into per-service task payloads.

    Merges each control vector's updates onto a deep copy of ``initial_state``
    and packages the resulting well-state dicts as ``RequestPayload`` objects,
    one per registered ``ServiceType``.
    """

    def __init__(self, initial_state: dict[str, Any]):
        self.initial_state: dict[str, Any] = initial_state

    def build(
        self, control_vectors: list[dict[str, float]]
    ) -> list[SolutionCandidateServicesTasks]:
        """Build solution candidate tasks for each control vector.

        Args:
            control_vectors: Flat key-value dicts produced by the optimizer,
                e.g. ``{"well_design#W1#md": 150.0, ...}``.

        Returns:
            One ``SolutionCandidateServicesTasks`` per control vector, each
            containing a ``RequestPayload`` per service type.
        """
        tasks_list: list[SolutionCandidateServicesTasks] = []

        for cv_dict in control_vectors:
            nested_updates = parse_flat_dict_to_nested(cv_dict)
            updated_state = update_initial_state(self.initial_state, nested_updates)
            control_vector = ControlVector(items=cv_dict)
            task_map: dict[ServiceType, RequestPayload] = {}

            for service_type, solution_items in updated_state.items():
                service_type = ServiceType(service_type)
                task_map[service_type] = RequestPayload(
                    request=list(solution_items.values()),
                    control_vector=control_vector,
                )

            tasks_list.append(SolutionCandidateServicesTasks(tasks=task_map))
        return tasks_list
