"""
NOTE:
This module must be aligned with python 3.10 syntax, as open-darts whl requires it.
"""

import json
import os
import re
import sys
import threading
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Protocol, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from logger import get_logger

from .common import (
    ConnectorInterface,
    GridCell,
    JsonPath,
    Point,
    SimulationResults,
    SimulationResultType,
    SimulationStatus,
    WellManagementServiceResultSchema,
    WellName,
    extract_well_with_perforations_points,
)  # the relative import, because connectors will be copied to worker, this will prevent import error
from .conn_utils.managed_subprocess import ManagedSubprocess
from .runner import SubprocessRunner, ThreadRunner

logger = get_logger("threading-worker", filename=__name__)


class _StructDiscretizerProtocol(Protocol):
    """
    Struct Discretizer Protocol
    """

    len_cell_xdir: npt.NDArray[np.float64]
    len_cell_ydir: npt.NDArray[np.float64]
    len_cell_zdir: npt.NDArray[np.float64]


class _GlobalData(TypedDict):
    """
    Global Data Protocol (dictionary)
    Attributes:
        dx: NumPy array of grid cell dimensions in the x-direction.
        dy: NumPy array of grid cell dimensions in the y-direction.
        dz: NumPy array of grid cell dimensions in the z-direction.
        start_z: The starting depth reference for the z-axis.
    """

    dx: npt.NDArray[np.float64]
    dy: npt.NDArray[np.float64]
    dz: npt.NDArray[np.float64]
    start_z: float


class _StructReservoirProtocol(Protocol):
    """Protocol for a structured reservoir model.

    This protocol defines the expected attributes for a structured reservoir.

    Attributes:
        nx (int): Number of grid cells in the x-direction.
        ny (int): Number of grid cells in the y-direction.
        nz (int): Number of grid cells in the z-direction.
        discretizer (_StructDiscretizerProtocol): Grid discretizer.
        global_data (_GlobalData): Global data for the grid.
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

    Usage: python3 main.py (or other file name with open-darts simulation) <path-to-configuration.json>

    example:

    @open_darts_input_configuration_injector
    def main(configuration_content):
        ...

    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        if len(sys.argv) < 2:
            logger.info("Usage: python main.py <path-to-configuration.json>")
            sys.exit(1)
        json_config_str = sys.argv[1]
        if os.path.isfile(json_config_str):
            try:
                with open(json_config_str, "r") as f:
                    config = json.load(f)
                if isinstance(config, str):
                    config = json.loads(config)
            except Exception:
                logger.error("Failed to read configuration file.")
                sys.exit(1)
        else:
            # For legacy reason, allowing passing json string directly
            try:
                config = json.loads(json_config_str)
            except json.JSONDecodeError:
                logger.error("Invalid JSON input.")
                sys.exit(1)

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                sys.exit(1)

        if not isinstance(config, (dict, list)):
            logger.error(f"Invalid JSON input, got:{type(config).__name__}")
            sys.exit(1)
        if isinstance(config, dict):
            if not all(isinstance(k, str) for k in config.keys()):
                logger.error(f"Invalid JSON input:{config}")
                sys.exit(1)

        func(config, *args, **kwargs)

    return wrapper


class OpenDartsConnector(ConnectorInterface):
    """
    Connector class design for OpenDarts simulator
    """

    MsgTemplate = "OpenDartsConnector: Type:{0}, Value:{1}"

    @staticmethod
    def run(
        config_path: JsonPath,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]:
        # Choose runner implementation via environment. Default uses subprocess runner.
        runner_mode = os.environ.get("OPEN_DARTS_RUNNER", "thread").lower()

        subprocess_runner = SubprocessRunner(
            managed_subprocess_factory=lambda *a, **k: ManagedSubprocess(*a, **k),
            broadcast_results_parser=OpenDartsConnector._get_broadcast_results,
            repo_root_getter=OpenDartsConnector._repo_root,
            worker_id_getter=OpenDartsConnector._current_worker_id,
        )

        if runner_mode == "thread":
            thread_runner = ThreadRunner(subprocess_runner)
            return thread_runner.run(config_path, stop)

        return subprocess_runner.run(config_path, stop)

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
            key: SimulationResultsType
            value: float or sequence of floats
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

        Returns:
            list[int]: List of cell indices within the reservoir grid corresponding to the provided perforations.

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

        cell_connector = _CellConnector(struct_reservoir)

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

        centroids = _calculate_centroids(struct_reservoir)
        cell1, cell2 = centroids[[0, -1]]
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

    @staticmethod
    @lru_cache(maxsize=1)
    def _repo_root() -> Path:
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "orchestration_files").exists():
                return parent
        raise RuntimeError("Failed to locate repository root directory.")

    @staticmethod
    def _current_worker_id() -> str | None:
        try:
            name = threading.current_thread().name
            if name.startswith("worker-"):
                return name.split("-", 1)[1]
        except Exception:
            raise RuntimeError("Failed to parse current thread name.")
        return None


class _CellConnector:
    """
    Notes:
    - This class explicitly depends on 'scipy' for KD-tree spatial queries. Ensure 'scipy' is installed
      through the 'open-darts' dependency specification
    """

    def __init__(
        self,
        struct_reservoir: _StructReservoirProtocol,
    ) -> None:
        centroids = _calculate_centroids(struct_reservoir)
        self._kd_tree = KDTree(centroids)

    def find_cell_index(self, coord: Point) -> int:
        _, idx = self._kd_tree.query(coord)
        return idx


def _calculate_centroids(struct_reservoir: _StructReservoirProtocol) -> npt.NDArray:
    """
    Calculates the centroids of grid cells within a 3D structured reservoir. Centroids are computed
    based on the structured reservoir's dimensions and its discretization parameters in the x, y,
    and z directions. The centroids represent the geometric centers of each grid cell in terms of
    their x, y, and z coordinates.

    Args:
        struct_reservoir: A data structure implementing the _StructReservoirProtocol. It contains
            reservoir configuration including dimensions (nx, ny, nz), the starting depth along the
            z-direction ("start_z"), and length of each cell in the x, y, and z directions (accessible
            via its discretizer attribute).

    Returns:
        npt.NDArray: A 2D NumPy array of shape (nx * ny * nz, 3), where each row represents the [x, y, z]
        coordinates of a cell's centroid in column-major order.
    """
    nx = struct_reservoir.nx
    ny = struct_reservoir.ny
    nz = struct_reservoir.nz

    start_z = struct_reservoir.global_data["start_z"]

    len_cell_zdir = struct_reservoir.discretizer.len_cell_zdir
    len_cell_ydir = struct_reservoir.discretizer.len_cell_ydir
    len_cell_xdir = struct_reservoir.discretizer.len_cell_xdir

    centroids_all_cells = np.zeros((nx, ny, nz, 3))  # 3 - for x,y,z coordinates
    # fill z-coordinates using DZ
    centroids_all_cells[:, :, 0, 2] = (
        start_z + len_cell_zdir[:, :, 0] * 0.5
    )  # nx*ny array of current layer's depths
    if nz > 1:
        d_cumsum = len_cell_zdir.cumsum(axis=2)
        centroids_all_cells[:, :, 1:, 2] = (
            start_z + (d_cumsum[:, :, :-1] + d_cumsum[:, :, 1:]) * 0.5
        )

    # fill y-coordinates using DY
    centroids_all_cells[:, 0, :, 1] = len_cell_ydir[:, 0, :] * 0.5  # nx*nz array
    if ny > 1:
        d_cumsum = len_cell_ydir.cumsum(axis=1)
        centroids_all_cells[:, 1:, :, 1] = (
            d_cumsum[:, :-1, :] + d_cumsum[:, 1:, :]
        ) * 0.5

    # fill x-coordinates using DX
    centroids_all_cells[0, :, :, 0] = len_cell_xdir[0, :, :] * 0.5  # ny*nz array
    if nx > 1:
        d_cumsum = len_cell_xdir.cumsum(axis=0)
        centroids_all_cells[1:, :, :, 0] = (
            d_cumsum[:-1, :, :] + d_cumsum[1:, :, :]
        ) * 0.5

    return np.reshape(centroids_all_cells, (nx * ny * nz, 3), order="F")
