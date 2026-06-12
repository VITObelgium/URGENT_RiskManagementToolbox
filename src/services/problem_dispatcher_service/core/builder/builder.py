from typing import Any

from services.problem_dispatcher_service.core.models import (
    RequestPayload,
    SolutionCandidateServicesTasks,
)
from services.problem_dispatcher_service.core.utils import (
    parse_flat_dict_to_nested,
    update_initial_state,
)
from services.shared import DomainServiceName
from services.solution_updater_service import ControlVector


class TaskBuilder:
    """Converts control vectors into per-domain-service task payloads.

    Merges each control vector's updates onto a deep copy of ``initial_state``
    and packages the resulting item-state dicts as ``RequestPayload`` objects,
    one per configured domain service.
    """

    def __init__(self, initial_state: dict[str, Any]):
        self.initial_state: dict[str, Any] = initial_state

    def build(
        self, control_vectors: list[dict[str, float]]
    ) -> list[SolutionCandidateServicesTasks]:
        """Build solution candidate tasks for each control vector.

        Args:
            control_vectors: Flat key-value dicts produced by the optimizer,
                e.g. ``{"well_design#W1#md": 150.0, ...}``. The first key
                segment selects the domain service.

        Returns:
            One ``SolutionCandidateServicesTasks`` per control vector, each
            containing a ``RequestPayload`` per domain service.
        """
        tasks_list: list[SolutionCandidateServicesTasks] = []

        for cv_dict in control_vectors:
            nested_updates = parse_flat_dict_to_nested(cv_dict)
            updated_state = update_initial_state(self.initial_state, nested_updates)
            control_vector = ControlVector(items=cv_dict)
            task_map: dict[DomainServiceName, RequestPayload] = {
                service: RequestPayload(
                    request=list(solution_items.values()),
                    control_vector=control_vector,
                )
                for service, solution_items in updated_state.items()
            }
            tasks_list.append(SolutionCandidateServicesTasks(tasks=task_map))
        return tasks_list
