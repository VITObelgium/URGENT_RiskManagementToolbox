"""
NOTE:
This module must be aligned with python 3.10 syntax, as open-darts whl requires it.
"""

import json
import re
import subprocess
import sys
import threading
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from logger import get_logger, stream_reader

from .common import (
    ConnectorInterface,
    GridCell,
    Point,
    SerializedJson,
    SimulationResults,
    SimulationResultType,
    SimulationStatus,
    WellManagementServiceResultSchema,
    WellName,
    extract_well_with_perforations_points,
)  # the relative import, because connectors will be copied to worker, this will prevent import error
from .conn_utils.managed_subprocess import ManagedSubprocess

logger = get_logger(__name__)


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

    Usage: python3 main.py (or other file name with open-darts simulation) 'configuration.json'

    example:

    @open_darts_input_configuration_injector
    def main(configuration_content):
        ...

    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        if len(sys.argv) < 2:
            logger.info("Usage: python main.py 'configuration.json (TBD)'")
            sys.exit(1)
            return
        json_config_str = sys.argv[1]
        try:
            config = json.loads(json_config_str)
        except json.JSONDecodeError:
            logger.error("Invalid JSON input.")
            sys.exit(1)
            return

        if not isinstance(config, dict):
            logger.error("Invalid JSON input.")
            sys.exit(1)
            return
        if not all(isinstance(k, str) for k in config.keys()):
            logger.error("Invalid JSON input.")
            sys.exit(1)
            return

        func(config, *args, **kwargs)

    return wrapper


class OpenDartsConnector(ConnectorInterface):
    """
    Connector class design for OpenDarts simulator
    """

    MsgTemplate = "OpenDartsConnector: Type:{0}, Value:{1}"

    @staticmethod
    def run(
        config: SerializedJson,
        stop: threading.Event | None = None,
    ) -> tuple[SimulationStatus, SimulationResults]:
        """
        Start reservoir simulator with given config and model in main.py
        Simulation is starting in separate process, output is capturing from the stdout
        and will be searching against template message to get back results parameters.

        On simulation file side, user must use OpenDartsConnector.broadcast_result method to channel the
        proper results to stdout

        If process terminate with non-zero error code then SimulationResults will be empty (?TBC)

        """

        default_failed_value: float = float(
            -1e3
        )  # for maximization problem it should be big negative number
        default_failed_results: SimulationResults = {
            k: default_failed_value for k in SimulationResultType
        }

        # Best-effort cleanup of potentially corrupted DARTS point-data caches.
        try:
            for p in Path.cwd().glob("obl_point_data_*.pkl"):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            # If cleanup fails we still try to proceed.
            logger.warning(
                "Failed to scan/remove old 'obl_point_data_*.pkl' caches; proceeding anyway."
            )

        # Determine Python interpreter to use for OpenDarts subprocess.
        # Priority: .venv_darts in parent dir > "python3" from PATH
        venv_candidate = Path.cwd().parent / ".venv_darts" / "bin" / "python"
        if venv_candidate.exists():
            selected = str(venv_candidate)
            reason = f"auto-detected {venv_candidate}"
        else:
            selected = "python3"
            reason = "default fallback"

        command = [selected, "-u", "main.py", config]
        logger.info(
            "Launching OpenDarts subprocess using interpreter (%s): %s",
            reason,
            selected,
        )

        manager = ManagedSubprocess(
            command_args=command,
            stream_reader_func=stream_reader,
            logger_info_func=logger.info,
            logger_error_func=logger.error,
        )

        try:
            with manager as process:
                timeout_duration = 15 * 60
                try:
                    waited = 0.0
                    poll_step = 0.25
                    while True:
                        if stop is not None and stop.is_set():
                            logger.warning(
                                "Stop requested; terminating OpenDarts subprocess."
                            )
                            if process.poll() is None:
                                process.terminate()
                                try:
                                    process.wait(timeout=3)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                    process.wait()
                            return SimulationStatus.FAILED, default_failed_results
                        try:
                            process.wait(timeout=poll_step)
                            break
                        except subprocess.TimeoutExpired:
                            waited += poll_step
                            if waited >= timeout_duration:
                                raise subprocess.TimeoutExpired(
                                    process.args, timeout_duration
                                )
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"Subprocess timed out after {timeout_duration} seconds. Terminating."
                    )
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            logger.warning(
                                "Subprocess did not terminate gracefully. Killing."
                            )
                            process.kill()
                            process.wait()
                    if manager.stdout_thread and manager.stdout_thread.is_alive():
                        manager.stdout_thread.join(timeout=2)
                    if manager.stderr_thread and manager.stderr_thread.is_alive():
                        manager.stderr_thread.join(timeout=2)
                    return SimulationStatus.TIMEOUT, default_failed_results

                if process.returncode != 0:
                    stderr_tail = "\n".join(manager.stderr_lines[-100:])
                    stdout_tail = "\n".join(manager.stdout_lines[-100:])
                    if process.returncode == -9:
                        logger.error(
                            "OpenDarts subprocess was killed (rc=-9). This is often due to OOM kill. "
                            "Consider lowering worker_count, reducing model size, or limiting threads. "
                            "Stdout tail:\n%s\nStderr tail:\n%s",
                            stdout_tail,
                            stderr_tail,
                        )
                    else:
                        logger.error(
                            "OpenDarts subprocess failed rc=%s. Stdout tail:\n%s\nStderr tail:\n%s",
                            process.returncode,
                            stdout_tail,
                            stderr_tail,
                        )
                        if (
                            "BlockingIOError" in stderr_tail
                            and "h5py" in stderr_tail
                            and "Unable to synchronously create file" in stderr_tail
                        ):
                            logger.error(
                                "Detected HDF5 file locking error from h5py. The worker sets HDF5_USE_FILE_LOCKING=FALSE, "
                                "but if the error persists, ensure each worker runs in an isolated directory and no other process "
                                "holds the same HDF5 file open."
                            )
                    return SimulationStatus.FAILED, default_failed_results

                full_stdout = "\n".join(manager.stdout_lines)
                broadcast_results = OpenDartsConnector._get_broadcast_results(
                    full_stdout
                )
                return SimulationStatus.SUCCESS, broadcast_results

        except FileNotFoundError:
            logger.exception(
                f"Failed to start subprocess. Command '{' '.join(command)}' not found."
            )
            return SimulationStatus.FAILED, default_failed_results
        except Exception as e:
            logger.exception(
                f"An error occurred while running the simulation subprocess: {e}"
            )
            return SimulationStatus.FAILED, default_failed_results

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
