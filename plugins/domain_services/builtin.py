"""Built-in domain service plugin — converts well models to simulation geometry."""

from __future__ import annotations

from services.problem_dispatcher_service.core.service.interface import (
    DomainServiceInterface,
)
from services.well_management_service import WellDesignService, WellModel
from services.well_management_service.core.models import WellDesignServiceResponse
from urgent_plugins import DomainServicePlugin


class BuiltinDomainService(DomainServiceInterface):
    ServiceName: str = "builtin"

    def build(self, wells: list[WellModel]) -> WellDesignServiceResponse:
        """Delegate geometry construction to the built-in WellDesignService."""
        return WellDesignService.process_request(
            {"models": [w.model_dump() for w in wells]}
        )


plugin = DomainServicePlugin(
    name=BuiltinDomainService.ServiceName,
    implementation=BuiltinDomainService,
)
