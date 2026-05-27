"""Built-in domain service plugin: delegates well design to WellDesignService."""

from __future__ import annotations

from typing import Any

from services.problem_dispatcher_service.core.service.interface import (
    DomainServiceInterface,
)
from services.well_management_service import WellDesignService
from services.well_management_service.core.models import WellDesignServiceResponse
from urgent_plugins import DomainServicePlugin


class BuiltinDomainService(DomainServiceInterface):
    ServiceName: str = "builtin"

    def process_request(
        self, request_dict: dict[str, Any]
    ) -> WellDesignServiceResponse:
        """Delegate geometry construction to the built-in WellDesignService."""
        return WellDesignService.process_request(request_dict)


plugin = DomainServicePlugin(
    name=BuiltinDomainService.ServiceName,
    implementation=BuiltinDomainService,
)
