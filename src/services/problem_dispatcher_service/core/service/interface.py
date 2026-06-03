from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import TypeAdapter


class DomainServiceInterface(ABC):
    """Plugin interface for translating well-design request dicts into simulation payloads.

    The framework owns parameter space construction, PSO, and constraint
    satisfaction. The plugin owns domain translation — what wells get built,
    how their geometry is described, and what a valid initial well state is.

    **What you must implement**

    1. ``get_well_state_adapter(cls) -> TypeAdapter[Any]``

       Return a ``TypeAdapter`` wrapping your domain's well model type.
       The framework calls ``.validate_python(data)`` and ``.dump_python(value)``
       on it to validate and normalise each well's ``initial_state`` dict.
       Called once at orchestrator startup, before ``ProblemDispatcherService``
       is constructed. Example::

           from pydantic import TypeAdapter

           @classmethod
           def get_well_state_adapter(cls) -> TypeAdapter[Any]:
               return TypeAdapter(YourWellModel)

    2. ``process_request(request_dict) -> dict[str, Any]``

       ``request_dict`` is shaped as ``{"models": list[dict]}``.
       Return a JSON-serialisable dict representing the simulation payload.
       The framework treats this dict as an opaque JSON blob — it is
       serialised to gRPC, transmitted to the worker, and written to disk
       as-is. The shape must be whatever your simulator connector expects.

    **Bypass / stub implementation**

    Return a minimal payload dict::

        return {
            "wells": [
                {"name": m["name"], "trajectory": [[0.0, 0.0, 0.0]], "completion": None}
                for m in request_dict["models"]
            ]
        }

    The control vector is attached separately by the orchestrator.
    """

    ServiceName: str

    @classmethod
    @abstractmethod
    def get_well_state_adapter(cls) -> TypeAdapter[Any]:
        """Return a TypeAdapter for validating and normalising a well's initial_state dict.

        Called once at orchestrator startup after the plugin is loaded and
        before ``ProblemDispatcherService`` is constructed. The returned adapter
        must accept dicts with at least a ``"name"`` key.
        """
        ...

    @abstractmethod
    def process_request(self, request_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert a well-design request dict to an opaque simulation payload dict.

        Called once per optimisation candidate. The returned dict is serialised
        to JSON and sent to the simulator worker verbatim — its structure is
        entirely plugin-defined.
        """
        ...
