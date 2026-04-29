from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from services.problem_dispatcher_service.core.models import ProblemDispatcherDefinition


def checkpoint_filename(run_id: str) -> str:
    return f"checkpoint_{run_id}.npz"


def _encode(s: str) -> npt.NDArray[np.uint8]:
    return np.frombuffer(s.encode("utf-8"), dtype=np.uint8)


def _decode(arr: npt.NDArray[np.uint8]) -> str:
    return bytes(arr).decode("utf-8")


def compute_file_hash(file_path: str | Path, chunk_size: int = 8192) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as file:
        chunk = file.read(chunk_size)
        while chunk:
            sha256_hash.update(chunk)
            chunk = file.read(chunk_size)
    return sha256_hash.hexdigest()


def save_checkpoint(
    checkpoint_path: Path,
    config: ProblemDispatcherDefinition,
    state: dict[str, Any],
    model_hash: str,
) -> None:
    """Write optimizer state to checkpoint_path atomically.

    checkpoint_path must be the full .npz destination (e.g. log/checkpoint_abc123.npz).
    state must be the dict returned by SolutionUpdaterService.get_checkpoint_state().
    """
    pso = state["pso_state"]
    mapper = state["mapper_state"]
    lc = state["loop_controller_state"]

    last_best = lc["last_best"]
    lc_last_best_arr = np.array(
        [np.nan if last_best is None else last_best], dtype=np.float64
    )

    arrays = {
        "config_json": _encode(config.model_dump_json()),
        "model_hash": _encode(model_hash),
        "next_positions": state["next_positions"],
        # PSO
        "pso_particles_best_positions": pso["particles_best_positions"],
        "pso_particles_best_results": pso["particles_best_results"],
        "pso_global_best_position": pso["global_best_position"],
        "pso_global_best_result": pso["global_best_result"],
        "pso_velocities": pso["velocities"],
        "pso_external_archive_positions": pso["external_archive_positions"],
        "pso_external_archive_results": pso["external_archive_results"],
        "pso_rng_state": _encode(pso["rng_state_json"]),
        # Mapper
        "mapper_cv_mapping": _encode(json.dumps(mapper["control_vector_mapping"])),
        "mapper_results_mapping": _encode(json.dumps(mapper["results_mapping"])),
        "mapper_population_size": np.array([mapper["population_size"]], dtype=np.int64),
        # Loop controller
        "lc_current_generation": np.array([lc["current_generation"]], dtype=np.int64),
        "lc_stall_left": np.array([lc["stall_left"]], dtype=np.int64),
        "lc_last_best": lc_last_best_arr,
        "lc_pareto_min_history": lc["pareto_min_history"],
        "lc_pareto_mean_history": lc["pareto_mean_history"],
    }

    tmp_path = checkpoint_path.with_suffix(".tmp")
    np.savez(tmp_path, **arrays)
    # np.savez appends .npz when not present
    tmp_npz = tmp_path.with_suffix(".tmp.npz")
    tmp_npz.replace(checkpoint_path)


def load_checkpoint(checkpoint_file: Path) -> dict[str, Any]:
    """Load optimizer state from an .npz checkpoint file.

    Returns a dict with keys:
        config         – ProblemDispatcherDefinition
        pso_state      – dict for PSOEngine.restore_checkpoint_state
        mapper_state   – dict for _Mapper.restore_checkpoint_state
        loop_controller_state – dict for loop controller restore
        next_positions – ndarray of particle positions for the next generation
    """
    data = np.load(checkpoint_file)

    config_dict = json.loads(_decode(data["config_json"]))
    config = ProblemDispatcherDefinition.model_validate(config_dict)

    last_best_val = float(data["lc_last_best"][0])
    last_best: float | None = None if np.isnan(last_best_val) else last_best_val

    model_hash = _decode(data["model_hash"]) if "model_hash" in data else None

    return {
        "config": config,
        "model_hash": model_hash,
        "pso_state": {
            "particles_best_positions": data["pso_particles_best_positions"],
            "particles_best_results": data["pso_particles_best_results"],
            "global_best_position": data["pso_global_best_position"],
            "global_best_result": data["pso_global_best_result"],
            "velocities": data["pso_velocities"],
            "external_archive_positions": data["pso_external_archive_positions"],
            "external_archive_results": data["pso_external_archive_results"],
            "rng_state_json": _decode(data["pso_rng_state"]),
        },
        "mapper_state": {
            "control_vector_mapping": json.loads(_decode(data["mapper_cv_mapping"])),
            "results_mapping": json.loads(_decode(data["mapper_results_mapping"])),
            "population_size": int(data["mapper_population_size"][0]),
        },
        "loop_controller_state": {
            "current_generation": int(data["lc_current_generation"][0]),
            "stall_left": int(data["lc_stall_left"][0]),
            "last_best": last_best,
            "pareto_min_history": data["lc_pareto_min_history"],
            "pareto_mean_history": data["lc_pareto_mean_history"],
        },
        "next_positions": data["next_positions"],
    }
