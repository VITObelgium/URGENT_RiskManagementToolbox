from typing import Any

from pydantic import TypeAdapter, BaseModel, Field


from services.problem_dispatcher_service.core.service import DomainServiceInterface
from urgent_plugins import DomainServicePlugin


class WellControlModel(BaseModel, extra="forbid"):
    name: str
    flow_chop: float = Field(..., ge=0.0, le=1.0)

class WellControlDomainService(DomainServiceInterface):

    ServiceName = "well_control"

    @classmethod
    def get_item_state_adapter(cls) -> TypeAdapter[Any]:
        return TypeAdapter(WellControlModel)

    def build_payload(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        adapter = self.get_item_state_adapter()
        payload: dict[str, float]  = {}
        for item in items:
            model = adapter.validate_python(item)
            payload[model.name] = model.flow_chop
        return payload


plugin = DomainServicePlugin(
    name=WellControlDomainService.ServiceName,
    implementation=WellControlDomainService,
)
