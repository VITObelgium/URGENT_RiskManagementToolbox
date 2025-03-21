from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field

from services.simulation_service.core.models.shared_from_well_management import (
    WellManagementServiceResult,
)


class SimulationResults(BaseModel, extra="forbid"):
    """
    NOTES:
        please make sure that SimulationResults class implementation is aligned with:
            - SimulationResultType from common.py
            - SimulationResults from common.py
    """

    Heat: float | Sequence[float] | Sequence[Sequence[float] | float]


class SimulationCase(BaseModel, extra="forbid"):
    wells: WellManagementServiceResult
    control_vector: dict[str, float]
    results: SimulationResults | None = Field(default=None)


class SimulationServiceRequest(BaseModel, extra="forbid"):
    simulation_cases: list[SimulationCase]


class SimulationServiceResponse(BaseModel, extra="forbid"):
    simulation_cases: list[SimulationCase]
