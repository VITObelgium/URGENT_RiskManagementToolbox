from __future__ import annotations

from pydantic import BaseModel

from services.well_management_service import WellDesignServiceResponse


class SimulationCase(BaseModel, extra="forbid"):
    wells: WellDesignServiceResponse
    control_vector: dict[str, float]
    results: dict[str, float]
    use_fem: bool = False
    rebuild_fem_cache: bool = False


class SimulationServiceRequest(BaseModel, extra="forbid"):
    simulation_cases: list[SimulationCase]


class SimulationServiceResponse(BaseModel, extra="forbid"):
    simulation_cases: list[SimulationCase]
