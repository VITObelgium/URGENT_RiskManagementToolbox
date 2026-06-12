from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import TypedDict

type WellName = str
type GridCell = tuple[int, int, int]
type Point = tuple[float, float, float]
type JsonPath = str


class PerforationSchema(TypedDict):
    name: str
    range: tuple[float, float]
    points: list[Point]


class CompletionSchema(TypedDict):
    perforations: list[PerforationSchema]


class WellSchema(TypedDict):
    name: str
    trajectory: list[Point]
    completion: CompletionSchema | None


class WellDesignServiceResultSchema(TypedDict):
    wells: list[WellSchema]


def extract_well_with_perforations_points(
    well_design_service_result: WellDesignServiceResultSchema,
) -> dict[WellName, tuple[Point, ...]]:
    results: dict[WellName, tuple[Point, ...]] = {}
    for well in well_design_service_result["wells"]:
        perforation_points: list[Point] = []
        try:
            well_name = well["name"]
            completion = well["completion"]
        except KeyError as e:
            raise KeyError(f"Well does not have a field: {e}")
        if completion:
            for p in completion["perforations"]:
                perforation_points.extend(p["points"])
        results[well_name] = tuple(perforation_points)
    return results


type SimulationResults = dict[str, float]


class SimulationStatus(Enum):
    SUCCESS = 0
    FAILED = 1
    TIMEOUT = 2
    EXCEPTION = 3


class ConnectorInterface(ABC):
    @classmethod
    def required_traces(cls) -> frozenset[str]:
        """Trace tags that must be provided by the configured domain services.

        Used by the startup plugin-trace verification: every tag returned here
        must appear in the union of the domain services' ``trace()`` sets,
        otherwise the run fails fast. Default: no requirements.
        """
        return frozenset()

    @abstractmethod
    def run(
        self,
        config_path: JsonPath,
        user_cost_function_with_default_values: SimulationResults,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]: ...
