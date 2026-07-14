"""Benchmark domain service plugin.

Converts benchmark wellhead item states into the flat decision vector consumed
by the synthetic benchmark models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, FiniteFloat, TypeAdapter, model_validator

from services.problem_dispatcher_service.core.service.interface import (
    DomainServiceInterface,
)
from urgent_plugins import DomainServicePlugin


class BenchmarkPositionModel(BaseModel, extra="forbid"):
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat = 0.0


class BenchmarkWellModel(BaseModel, extra="forbid"):
    name: str
    wellhead: BenchmarkPositionModel


class BenchmarkWellDesignRequest(BaseModel, extra="forbid"):
    models: list[BenchmarkWellModel] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_names(self) -> "BenchmarkWellDesignRequest":
        names = [model.name for model in self.models]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Benchmark well names must be unique: {duplicates}")
        return self


class BenchmarkWellDesignDomainService(DomainServiceInterface):
    """Builds a benchmark decision vector from each item's wellhead x/y."""

    ServiceName = "benchmark_well_design"

    @classmethod
    def trace(cls) -> frozenset[str]:
        return frozenset({cls.ServiceName, "well_design"})

    @classmethod
    def get_item_state_adapter(cls) -> TypeAdapter[Any]:
        return TypeAdapter(BenchmarkWellModel)

    def build_payload(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        request = BenchmarkWellDesignRequest(models=items)
        decision_vector: list[float] = []
        wellheads: dict[str, dict[str, float]] = {}

        for model in request.models:
            wellhead = model.wellhead
            decision_vector.extend([float(wellhead.x), float(wellhead.y)])
            wellheads[model.name] = wellhead.model_dump()

        return {"x": decision_vector, "wellheads": wellheads}


plugin = DomainServicePlugin(
    name=BenchmarkWellDesignDomainService.ServiceName,
    implementation=BenchmarkWellDesignDomainService,
)
