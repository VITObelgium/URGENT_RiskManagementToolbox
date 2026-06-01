"""Built-in domain service plugin.

Dispatches well-design requests to the WMS geometry kernel and returns
simulation-ready ``WellDesignServiceResponse`` objects.

note::

    This plugin imports ``services.well_management_service.core.well_templates``
    (the private geometry kernel) because it *is* the reference implementation.
    User-written domain plugins must **not** import from ``well_templates``.
    Build your own geometry or construct a minimal ``WellDesignServiceResponse``
    from the public types re-exported by ``DomainServiceInterface``.

Bootstrapped via::

    pixi run create-plugin domain BuiltinDomainService --plugin-name builtin
"""

from __future__ import annotations

from typing import Any, assert_never

import services.well_management_service.core.well_templates as _wt
from logger import get_logger
from services.problem_dispatcher_service.core.service.interface import (
    DomainServiceInterface,
    SimulationWellModel,
    WellDesignServiceRequest,
    WellDesignServiceResponse,
)
from services.well_management_service import (
    HWellModel,
    IWellModel,
    JWellModel,
    SWellModel,
)
from urgent_plugins import DomainServicePlugin

logger = get_logger(__name__)


class BuiltinDomainService(DomainServiceInterface):
    """Reference domain service: dispatches each well model to its geometry template."""

    ServiceName: str = "builtin"

    def process_request(
        self, request_dict: dict[str, Any]
    ) -> WellDesignServiceResponse:
        request = WellDesignServiceRequest(**request_dict)
        wells: list[SimulationWellModel] = []
        for model in request.models:
            logger.debug("Building well %r (type=%s)", model.name, type(model).__name__)
            match model:
                case IWellModel():
                    well = _wt.IWellTemplate.from_model(model).build()
                case JWellModel():
                    well = _wt.JWellTemplate.from_model(model).build()
                case SWellModel():
                    well = _wt.SWellTemplate.from_model(model).build()
                case HWellModel():
                    well = _wt.HWellTemplate.from_model(model).build()
                case _ as unreachable:
                    assert_never(unreachable)
            wells.append(SimulationWellModel.from_well(well))
        return WellDesignServiceResponse(wells=wells)


plugin = DomainServicePlugin(
    name=BuiltinDomainService.ServiceName,
    implementation=BuiltinDomainService,
)
