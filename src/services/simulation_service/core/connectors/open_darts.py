import json
import re
import subprocess
import sys
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Protocol, Sequence, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from .common import (
    ConnectorInterface,
    GridCell,
    Point,
    SerializedJson,
    SimulationResults,
    SimulationResultType,
    WellManagementServiceResultSchema,
    WellName,
    extract_well_with_perforations_points,
)  # the relative import, because connectors will be copied to worker, this will prevent import error


class _StructDiscretizerProtocol(Protocol):
    """
    Struct Discretizer Protocol

    required properties:
    centroids_all_cells
    """

    centroids_all_cells: npt.NDArray[np.float64]


class _GlobalData(TypedDict):
    """
    Global Data Protocol (dictionary)
    required keys:
    dx
    dy
    dz
    """

    dx: npt.NDArray[np.float64]
    dy: npt.NDArray[np.float64]
    dz: npt.NDArray[np.float64]


class _StructReservoirProtocol(Protocol):
    """
    Struct Reservoir Protocol
    required properties:
    nx
    ny
    nz,
    discretizer
    global_data

    required methods:
    find_cell_index
    """

    nx: int
    ny: int
    nz: int
    discretizer: _StructDiscretizerProtocol
    global_data: _GlobalData


def open_darts_input_configuration_injector(func: Callable[..., None]) -> Any:
    """
    The decorator should be used on top of the main function which trigger Open-Darts simulation
    Decorator allow to inject configuration json directly from the command line

    Usage: python3 main.py (or other file name with open-darts simulation) 'configuration.json'

    example:

    @open_darts_input_configuration_injector
    def main(configuration_content):
        ...

    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        if len(sys.argv) < 2:
            print("Usage: python main.py 'configuration.json (TBD)'")
            sys.exit(1)
        json_config_str = sys.argv[1]
        try:
            config_data = json.loads(json_config_str)
            func(config_data, *args, **kwargs)
        except json.JSONDecodeError:
            print("Invalid JSON input.")
            sys.exit(1)

    return wrapper


class OpenDartsConnector(ConnectorInterface):
    """
    Connector class design for OpenDarts simulator
    """

    MsgTemplate = "OpenDartsConnector: Type:{0}, Value:{1}"

    @staticmethod
    def run(
        config: SerializedJson,
    ) -> SimulationResults:
        """
        Start reservoir simulator with given config and model in main.py
        Simulation is starting in separate process, output is capturing from the stdout
        and will be searching against template message to get back results parameters.

        On simulation file side, user must use OpenDartsConnector.broadcast_result method to channel the
        proper results to stdout

        If process terminate with non-zero error code then SimulationResults will be empty (?TBC)

        """
        process = subprocess.run(
            ["python3", "main.py", config],
            capture_output=True,
            text=True,  # main.py must be present on worker side
        )

        if process.returncode != 0:
            raise RuntimeError(f"Error: {process.stderr}")

        process_stdout = process.stdout
        broadcast_results = OpenDartsConnector._get_broadcast_results(process_stdout)

        return broadcast_results

    @staticmethod
    def _get_broadcast_results(
        stdout: str,
    ) -> SimulationResults:
        template = OpenDartsConnector.MsgTemplate
        re_key = r"(\w+)"  # str
        re_value = r"\[?([\d.,\s]+)\]?"  # float or sequence of floats
        pattern = template.format(re_key, re_value)
        matches = re.findall(pattern, stdout)

        results = defaultdict(list)
        value: float | Sequence[float]
        for key, raw_value in matches:
            raw_value = raw_value.strip()
            if any(x in raw_value for x in [",", " "]):
                value = [float(v) for v in re.split(r"[,\s]+", raw_value) if v]
            else:
                value = float(raw_value)
            results[key].append(value)

        # Simplify output if single values per key
        return {k: v[0] if len(v) == 1 else v for k, v in results.items()}

    @staticmethod
    def broadcast_result(
        key: SimulationResultType, value: float | Sequence[float]
    ) -> None:
        """
        Use for broadcast given simulation results to stdout
        key - SimulationResultsType
        value - float or sequence of floats
        """
        broadcast_template = OpenDartsConnector.MsgTemplate.format(key, value)
        print(broadcast_template)

    @staticmethod
    def get_well_connection_cells(
        well_management_service_result: WellManagementServiceResultSchema,
        struct_reservoir: _StructReservoirProtocol,
    ) -> dict[WellName, tuple[GridCell, ...]]:
        """
        Retrieve the connection cells for a given well, identifying which reservoir grid cells
        correspond to the well's perforations.

        This method maps well perforations onto the reservoir's discretized cells, potentially
        using spatial querying (e.g., through a KD-tree). It ensures that connections between
        wells and reservoir cells are accurately identified considering reservoir geometry,
        discretization, and perforation coordinates.

        Args:
            [Include parameters here, for example:]
            well_name (str): Identifier or name of the well to locate perforation-related cells.
            perforations (List[Tuple[float, float, float]]): List of perforation coordinates (X, Y, Z) associated with the well.

        Returns:
            List[int]: List of cell indices within the reservoir grid corresponding to the provided perforations.

        Raises:
            ValueError: If provided perforation coordinates are invalid or outside the reservoir bounds.
            RuntimeError: If internal spatial querying fails due to invalid KD-tree or spatial data setup.


        Notes:
            - Ensure that the internal KD-tree or spatial data used for find operations is initialized and updated accurately.
            - Coordinate systems (units and orientation) must be consistent with reservoir modeling conventions.

        """

        result: dict[WellName, tuple[GridCell, ...]] = {}
        wells_with_perforations_points: dict[WellName, tuple[Point, ...]] = (
            extract_well_with_perforations_points(well_management_service_result)
        )

        cell_connector = _CellConnector(struct_reservoir.discretizer)

        for well_name, perforations_points in wells_with_perforations_points.items():
            filtered_perforations_points = (
                OpenDartsConnector._filter_perforations_inside_reservoir(
                    struct_reservoir, perforations_points
                )
            )
            global_perforation_idx: list[int] = [
                cell_connector.find_cell_index(p) for p in filtered_perforations_points
            ]
            unique_global_perforation_idx = list(set(global_perforation_idx))
            unique_perforation_grid_cells = [
                OpenDartsConnector._global_idx_to_grid_cell(idx, struct_reservoir)
                for idx in unique_global_perforation_idx
            ]
            result[well_name] = tuple(unique_perforation_grid_cells)

        return result

    @staticmethod
    def _global_idx_to_grid_cell(
        idx: int, struct_reservoir: _StructReservoirProtocol
    ) -> GridCell:
        nx = struct_reservoir.nx
        ny = struct_reservoir.ny
        nz = struct_reservoir.nz

        x = idx % nx
        y = (idx // nx) % ny
        z = (idx // (nx * ny)) % nz
        return x + 1, y + 1, z + 1

    @staticmethod
    def _filter_perforations_inside_reservoir(
        struct_reservoir: _StructReservoirProtocol,
        well_perforations_points: tuple[Point, ...],
    ) -> tuple[Point, ...]:
        dx = struct_reservoir.global_data["dx"]
        dy = struct_reservoir.global_data["dy"]
        dz = struct_reservoir.global_data["dz"]

        cell1, cell2 = struct_reservoir.discretizer.centroids_all_cells[[0, -1]]
        bounds_min = cell1 - 0.5 * np.array([dx[0, 0, 0], dy[0, 0, 0], dz[0, 0, 0]])
        bounds_max = cell2 + 0.5 * np.array(
            [dx[-1, -1, -1], dy[-1, -1, -1], dz[-1, -1, -1]]
        )

        well_perforations_points_ar = np.asarray(well_perforations_points)
        in_bounds = np.all(
            (well_perforations_points_ar >= bounds_min)
            & (well_perforations_points_ar <= bounds_max),
            axis=1,
        )

        filtered_perforations_points = well_perforations_points_ar[in_bounds]

        return tuple(filtered_perforations_points)


class _CellConnector:
    """
    Notes:
    - This class explicitly depends on 'scipy' for KD-tree spatial queries. Ensure 'scipy' is installed
      through the 'open-darts' dependency specification
    """

    def __init__(self, discretizer: _StructDiscretizerProtocol) -> None:
        self._kd_tree = KDTree(discretizer.centroids_all_cells)

    def find_cell_index(self, coord: Point) -> int:
        _, idx = self._kd_tree.query(coord)
        return idx
