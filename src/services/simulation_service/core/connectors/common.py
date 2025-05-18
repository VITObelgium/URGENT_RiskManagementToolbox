"""
NOTE:
This module must be aligned with python 3.10 syntax, as open-darts whl requires it.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, TypeAlias, TypedDict

WellName: TypeAlias = str
GridCell: TypeAlias = tuple[int, int, int]
Point: TypeAlias = tuple[float, float, float]
SerializedJson: TypeAlias = str


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


class SimulationResultType(str, Enum):
    """
    NOTES:
        please make sure that implementation of SimulationResultType is aligned with:
            - SimulationResults from user.py
    """

    Heat = "Heat"


SimulationResults: TypeAlias = dict[
    SimulationResultType, float | Sequence[float] | Sequence[Sequence[float] | float]
]


class SimulationStatus(Enum):
    SUCCESS = 0
    FAILED = 1
    TIMEOUT = 2


class ConnectorInterface(ABC):
    @staticmethod
    @abstractmethod
    def run(config: SerializedJson) -> tuple[SimulationStatus, SimulationResults]: ...
