from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from services.well_management_service.core.models import (
    SimulationWellModel,
    WellDesignServiceRequest,
    WellDesignServiceResponse,
)

__all__ = [
    "DomainServiceInterface",
    "SimulationWellModel",
    "WellDesignServiceRequest",
    "WellDesignServiceResponse",
]


class DomainServiceInterface(ABC):
    """Plugin interface for translating well-design request dicts into simulation inputs.

    The framework owns parameter space construction, PSO, and constraint
    satisfaction.  The plugin owns domain translation — what wells get built
    and how their geometry is described.

    **What you must implement**

    ``process_request(request_dict) -> WellDesignServiceResponse``

    **What** ``request_dict`` **contains**

    A dict shaped as ``{"models": list[dict]}``, where each inner dict is a
    serialised ``WellModel`` (discriminator key ``"well_type"``: ``"IWell"``,
    ``"JWell"``, ``"SWell"``, or ``"HWell"``).  Parse with::

        request = WellDesignServiceRequest(**request_dict)

    or inspect ``request_dict["models"]`` directly for bypass/stub cases.

    **Bypass / stub implementations**

    Return a minimal but schema-valid ``WellDesignServiceResponse``::

        WellDesignServiceResponse(wells=[
            SimulationWellModel(name=m["name"], trajectory=((0., 0., 0.),), completion=None)
            for m in request_dict["models"]
        ])

    The control vector is attached to the simulation case separately by the
    orchestrator, so geometry can be trivial for benchmark/test plugins.

    **What is NOT your API**

    - ``services.well_management_service.core._well_templates`` — private geometry
      kernel; user plugins must not import from it.
    - ``WellBuilder``, section classes, ``Trajectory`` — internal geometry math.

    All public types needed by a domain plugin are re-exported from this module:
    ``WellDesignServiceRequest``, ``WellDesignServiceResponse``,
    ``SimulationWellModel``.
    """

    ServiceName: str

    @abstractmethod
    def process_request(
        self, request_dict: dict[str, Any]
    ) -> WellDesignServiceResponse:
        """Convert a well-design request to a simulation-ready response.

        Called once per optimisation candidate with a request dict shaped as
        ``{"models": [...]}``.
        """
        ...
