"""Demo well_counter domain service plugin — a second payload column."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, TypeAdapter

from services.problem_dispatcher_service.core.service.interface import (
    DomainServiceInterface,
)
from urgent_plugins import DomainServicePlugin


class WellCountModel(BaseModel, extra="forbid"):
    name: str
    count: float = Field(ge=0.0)


class WellCounterDomainService(DomainServiceInterface):
    """Demo domain service: rounds each item's continuous count to an integer."""

    ServiceName = "well_counter"

    @classmethod
    def get_item_state_adapter(cls) -> TypeAdapter[Any]:
        return TypeAdapter(WellCountModel)

    def build_payload(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        adapter = self.get_item_state_adapter()
        counts: dict[str, int] = {}
        for item in items:
            model = adapter.validate_python(item)
            counts[model.name] = int(round(model.count))
        return {"counts": counts}


# Plugin descriptor

plugin = DomainServicePlugin(
    name=WellCounterDomainService.ServiceName,
    implementation=WellCounterDomainService,
)
