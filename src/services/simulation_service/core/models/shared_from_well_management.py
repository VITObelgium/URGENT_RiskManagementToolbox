from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SimulationWellPerforationModel(BaseModel, extra="forbid"):
    range: tuple[float, float]
    points: tuple[tuple[float, float, float], ...]

    @model_validator(mode="after")
    def validate_perforation_start_and_end(self) -> SimulationWellPerforationModel:
        if self.range[0] >= self.range[1]:
            raise ValueError(
                f"Invalid range: start ({self.range[0]}) must be less than end ({self.range[1]})"
            )
        return self


class SimulationWellCompletionModel(BaseModel, extra="forbid"):
    perforations: tuple[SimulationWellPerforationModel, ...]


class SimulationWellModel(BaseModel, extra="forbid"):
    name: str
    trajectory: tuple[tuple[float, float, float], ...]
    completion: SimulationWellCompletionModel | None = Field(default=None)


class WellManagementServiceResult(BaseModel, extra="forbid"):
    wells: list[SimulationWellModel]

    @model_validator(mode="after")
    def ensure_unique_names(self) -> WellManagementServiceResult:
        seen_names = set()
        for well in self.wells:
            if well.name in seen_names:
                raise ValueError("Wells names must be unique.")
            seen_names.add(well.name)
        return self
