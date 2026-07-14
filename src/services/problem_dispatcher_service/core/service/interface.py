from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import TypeAdapter


class DomainServiceInterface(ABC):
    """Plugin interface translating per-item state dicts into one simulation payload column.

    A run may configure several domain services at once; each one owns a single
    key ("column") in the payload sent to the simulator, named after the
    service. The framework owns parameter-space construction, optimization, and
    constraint satisfaction. The plugin owns domain translation- what its
    items mean and what its payload column looks like.

    **What you must implement**

    1. ``get_item_state_adapter(cls) -> TypeAdapter[Any]``

       Return a ``TypeAdapter`` wrapping the pydantic model that describes one
       item's ``initial_state`` dict in the run configuration. The framework
       calls ``.validate_python(data)`` and ``.dump_python(value)`` on it to
       validate and normalise each item's state right after the plugin is
       loaded, before any service is constructed. Example::

           from pydantic import TypeAdapter

           @classmethod
           def get_item_state_adapter(cls) -> TypeAdapter[Any]:
               return TypeAdapter(YourItemModel)

    2. ``build_payload(items) -> dict[str, Any]``

       ``items`` is the list of this service's item-state dicts with the
       optimizer's control-vector updates already merged in. Return a
       JSON-serialisable dict- it becomes the ``{ServiceName: payload}``
       column of the simulation case and is transmitted to the worker.
       Its shape is entirely plugin-defined; pick whatever your
       simulator model expects.

    **Trace verification**

    ``trace()`` returns the compatibility tags this service provides
    (default: ``{ServiceName}``). At startup the framework checks every tag
    required by the configured connector and optimizer
    (``required_traces()`` on their interfaces) against the union of all
    domain service traces and fails fast on a mismatch. Override ``trace()``
    when your payload column satisfies additional contracts.
    """

    ServiceName: ClassVar[str]

    @classmethod
    def trace(cls) -> frozenset[str]:
        """Compatibility tags this domain service provides."""
        return frozenset({cls.ServiceName})

    @classmethod
    @abstractmethod
    def get_item_state_adapter(cls) -> TypeAdapter[Any]:
        """Return a TypeAdapter validating and normalising one item's initial_state dict.

        Called once at startup after the plugin is loaded, before any
        service is constructed. The returned adapter must accept dicts with
        at least a ``"name"`` key.
        """
        ...

    @abstractmethod
    def build_payload(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert this service's item states into its simulation payload column.

        Called once per optimisation candidate. The returned dict is placed
        under the service's name in the simulation payload and sent to the
        simulator verbatim — its structure is entirely plugin-defined.
        """
        ...
