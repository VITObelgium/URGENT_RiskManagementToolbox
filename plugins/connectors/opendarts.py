"""Built-in OpenDARTS connector plugin.

Loaded by the urgent_plugins loader (orchestrator context) to register the
descriptor in the plugin registry, and staged into the worker subprocess
directory so user simulation scripts can call
:meth:`OpenDartsConnector.broadcast_result` and other helpers.

Both contexts have ``src/`` on ``PYTHONPATH`` (orchestrator: always; worker:
injected via ``PYTHONPATH`` env var by :func:`_build_env`), so absolute
imports are used throughout.

Bootstrapped via::

    pixi run create-plugin connector OpenDartsConnector --plugin-name opendarts
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, Protocol, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from logger import get_logger

from services.simulation_service.core.connectors.common import (
    GridCell,
    JsonPath,
    Point,
    SimulationResults,
    WellName,
    extract_well_with_perforations_points,
)
from services.simulation_service.core.connectors.runner import (
    SubprocessConnectorInterface,
)

logger = get_logger("threading-worker", filename=__name__)


class _StructDiscretizerProtocol(Protocol):
    """Struct Discretizer Protocol."""

    len_cell_xdir: npt.NDArray[np.float64]
    len_cell_ydir: npt.NDArray[np.float64]
    len_cell_zdir: npt.NDArray[np.float64]


class _GlobalData(TypedDict):
    """Global Data Protocol (dictionary).

    Attributes:
        dx: Cell sizes in the x-direction.
        dy: Cell sizes in the y-direction.
        dz: Cell sizes in the z-direction.
        start_z: Starting depth reference for the z-axis.
    """

    dx: npt.NDArray[np.float64]
    dy: npt.NDArray[np.float64]
    dz: npt.NDArray[np.float64]
    start_z: float


class _StructReservoirProtocol(Protocol):
    """Protocol for a structured reservoir model.

    Attributes:
        nx (int): Cells in the x-direction.
        ny (int): Cells in the y-direction.
        nz (int): Cells in the z-direction.
        discretizer (_StructDiscretizerProtocol): Grid discretizer.
        global_data (_GlobalData): Global data for the grid.
    """

    nx: int
    ny: int
    nz: int
    discretizer: _StructDiscretizerProtocol
    global_data: _GlobalData


def open_darts_input_configuration_injector(func: Callable[..., None]) -> Any:
    """Inject a JSON configuration into the user simulation's main function.

    Usage::

        @open_darts_input_configuration_injector
        def main(configuration_content): ...

    The decorated function receives the parsed configuration as its first
    positional argument when the worker subprocess invokes
    ``python main.py <path-to-configuration.json>``.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        if len(sys.argv) < 2:
            logger.info("Usage: python main.py <path-to-configuration.json>")
            sys.exit(1)
        json_config_str = sys.argv[1]
        if os.path.isfile(json_config_str):
            try:
                with open(json_config_str) as f:
                    config = json.load(f)
                if isinstance(config, str):
                    config = json.loads(config)
            except Exception as e:
                logger.error(f"Failed to read configuration file: {e}.")
                sys.exit(1)
        else:
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


class OpenDartsConnector(SubprocessConnectorInterface):
    """Connector for the OpenDARTS reservoir simulator."""

    ConnectorName = "opendarts"
    MsgTemplate = "OpenDartsConnector: Type:{0}, Value:{1}"

    @classmethod
    def required_traces(cls) -> frozenset[str]:
        """The bundled helpers assume a well_design payload column is present."""
        return frozenset({"well_design"})

    @classmethod
    def build_command(cls, config_path: JsonPath) -> list[str]:
        runner_mode = os.environ.get("RUNNER_MODE", "thread").lower()
        if runner_mode == "docker":
            return [sys.executable, "-u", "main.py", config_path]
        return ["pixi", "run", "-e", "worker", "python", "-u", "main.py", config_path]

    @classmethod
    def parse_results(cls, work_dir: Path | None, stdout: str) -> SimulationResults:  # noqa: ARG002
        template = cls.MsgTemplate
        re_key = r"(\w+)"
        re_value = r"([+-]?\d+(?:\.\d+)?)"
        pattern = template.format(re_key, re_value)
        matches = re.findall(pattern, stdout)
        results: dict[str, float] = {}
        for key, raw_value in matches:
            results[key] = float(raw_value.strip())
        return results

    @classmethod
    def pre_run(cls, work_dir: Path | None) -> None:
        """Remove stale OpenDARTS OBL point cache files before each run."""
        scan_dir = work_dir if work_dir is not None else Path.cwd()
        try:
            for p in scan_dir.glob("obl_point_data_*.pkl"):
                try:
                    p.unlink(missing_ok=True)
                except OSError as e:
                    logger.debug(f"Could not remove cache file {p}: {e}")
        except OSError as e:
            logger.warning(
                f"Failed to scan/remove old 'obl_point_data_*.pkl' caches; proceeding anyway. Error: {e}"
            )

    @staticmethod
    def broadcast_result(key: str, value: float) -> None:
        """Emit a simulation result line for the parent worker to capture."""
        broadcast_template = OpenDartsConnector.MsgTemplate.format(key, float(value))
        print(broadcast_template)

    @staticmethod
    def get_well_connection_cells(
        domain_services_result: dict[str, Any],
        struct_reservoir: _StructReservoirProtocol,
    ) -> dict[WellName, tuple[GridCell, ...]]:
        """Map well perforation points to reservoir grid cells.

        Helper used by OpenDARTS-specific user models. Not part of the generic
        :class:`ConnectorInterface` contract.
        """
        result: dict[WellName, tuple[GridCell, ...]] = {}
        well_design_result = domain_services_result.get("well_design")
        if not well_design_result:
            return {}
        wells_with_perforations_points: dict[WellName, tuple[Point, ...]] = (
            extract_well_with_perforations_points(well_design_result)
        )

        cell_connector = _CellConnector(struct_reservoir)

        for well_name, perforations_points in wells_with_perforations_points.items():
            if not perforations_points:
                logger.warning(
                    f"Well {well_name} has no perforations points, skipping."
                )
                continue

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
    """KD-tree wrapper for mapping points to reservoir cells.

    Depends on ``scipy.spatial`` from the OpenDARTS wheel.
    """

    def __init__(self, struct_reservoir: _StructReservoirProtocol) -> None:
        centroids = _calculate_centroids(struct_reservoir)
        self._kd_tree = KDTree(centroids)

    def find_cell_index(self, coord: Point) -> int:
        _, idx = self._kd_tree.query(coord)
        return int(idx)


def _calculate_centroids(struct_reservoir: _StructReservoirProtocol) -> npt.NDArray:
    """Compute centroid coordinates for every cell of a structured reservoir."""
    nx = struct_reservoir.nx
    ny = struct_reservoir.ny
    nz = struct_reservoir.nz

    start_z = struct_reservoir.global_data["start_z"]

    len_cell_zdir = struct_reservoir.discretizer.len_cell_zdir
    len_cell_ydir = struct_reservoir.discretizer.len_cell_ydir
    len_cell_xdir = struct_reservoir.discretizer.len_cell_xdir

    centroids_all_cells = np.zeros((nx, ny, nz, 3))
    centroids_all_cells[:, :, 0, 2] = start_z + len_cell_zdir[:, :, 0] * 0.5
    if nz > 1:
        d_cumsum = len_cell_zdir.cumsum(axis=2)
        centroids_all_cells[:, :, 1:, 2] = (
            start_z + (d_cumsum[:, :, :-1] + d_cumsum[:, :, 1:]) * 0.5
        )

    centroids_all_cells[:, 0, :, 1] = len_cell_ydir[:, 0, :] * 0.5
    if ny > 1:
        d_cumsum = len_cell_ydir.cumsum(axis=1)
        centroids_all_cells[:, 1:, :, 1] = (
            d_cumsum[:, :-1, :] + d_cumsum[:, 1:, :]
        ) * 0.5

    centroids_all_cells[0, :, :, 0] = len_cell_xdir[0, :, :] * 0.5
    if nx > 1:
        d_cumsum = len_cell_xdir.cumsum(axis=0)
        centroids_all_cells[1:, :, :, 0] = (
            d_cumsum[:-1, :, :] + d_cumsum[1:, :, :]
        ) * 0.5

    return np.reshape(centroids_all_cells, (nx * ny * nz, 3), order="F")


# Plugin descriptor — only built when ``urgent_plugins`` is importable.
try:
    from urgent_plugins import ConnectorPlugin

    plugin = ConnectorPlugin(
        name=OpenDartsConnector.ConnectorName,
        implementation=OpenDartsConnector,
    )
except ImportError:
    pass
