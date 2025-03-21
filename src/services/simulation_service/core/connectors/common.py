from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Sequence, TypedDict

type WellName = str
type GridCell = tuple[int, int, int]
type Point = tuple[float, float, float]


class PerforationSchema(TypedDict):
    range: tuple[float, float]
    points: list[Point]


class CompletionSchema(TypedDict):
    perforations: list[PerforationSchema]


class WellSchema(TypedDict):
    name: str
    trajectory: list[Point]
    completion: CompletionSchema | None


class WellManagementServiceResultSchema(TypedDict):
    wells: list[WellSchema]


def extract_well_with_perforations_points(
    well_management_service_result: WellManagementServiceResultSchema,
) -> dict[WellName, tuple[Point, ...]]:
    results: dict[WellName, tuple[Point, ...]] = {}
    for well in well_management_service_result["wells"]:
        perforation_points: list[Point] = []
        well_name = well["name"]
        completion = well["completion"]
        if completion:
            for p in completion["perforations"]:
                perforation_points.extend(p["points"])
        results[well_name] = tuple(perforation_points)
    return results


class SimulationResultType(StrEnum):
    """
    NOTES:
        please make sure that implementation of SimulationResultType is aligned with:
            - SimulationResults from user.py
    """

    Heat = "Heat"


type SimulationResults = dict[
    SimulationResultType, float | Sequence[float] | Sequence[Sequence[float] | float]
]

type SerializedJson = str


class ConnectorInterface(ABC):
    @staticmethod
    @abstractmethod
    def run(config: SerializedJson) -> SimulationResults: ...
