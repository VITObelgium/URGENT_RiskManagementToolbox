from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SimulationCase(BaseModel, extra="forbid"):
    """One simulation candidate.

    ``payload`` is the opaque simulation input: one key per configured domain
    service (the service's "column"), with plugin-defined values.
    """

    payload: dict[str, Any]
    control_vector: dict[str, float]
    results: dict[str, float]


class SimulationServiceRequest(BaseModel, extra="forbid"):
    simulation_cases: list[SimulationCase]
    connector: str


class SimulationServiceResponse(BaseModel, extra="forbid"):
    simulation_cases: list[SimulationCase]
