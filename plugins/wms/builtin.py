"""Built-in WMS plugin — delegates to WellDesignHandler and WellDesignService.

Bootstrapped via:
    pixi run create-plugin wms BuiltinWms --plugin-name builtin
"""

from __future__ import annotations

from typing import Any

from services.problem_dispatcher_service.core.utils import DEFAULT_SEPARATOR
from services.problem_dispatcher_service.core.service.handlers import (
    ProblemTypeHandler,
    WellDesignHandler,
)
from services.shared import Boundaries
from services.well_management_service import WellDesignService
from services.well_management_service.core.models import WellDesignServiceResponse
from urgent_plugins import WellManagementPlugin

_handler = WellDesignHandler()


class BuiltinWms(ProblemTypeHandler):
    WmsName: str = "builtin"

    def build_initial_state(self, items: list[Any]) -> dict[str, Any]:
        return _handler.build_initial_state(items)

    def build_full_key_boundaries(
        self, items: list[Any], separator: str = DEFAULT_SEPARATOR
    ) -> dict[str, Boundaries]:
        return _handler.build_full_key_boundaries(items, separator)

    def build_service_tasks(
        self, solution_items: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return _handler.build_service_tasks(solution_items)


def build_wells(request: list[dict]) -> WellDesignServiceResponse:
    return WellDesignService.process_request({"models": request})


plugin = WellManagementPlugin(
    name=BuiltinWms.WmsName,
    handler=BuiltinWms(),
    build_wells=build_wells,
)
