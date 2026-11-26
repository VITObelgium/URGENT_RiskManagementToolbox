"""RMT worker entry point for one minewater candidate.

Run as: python -u main.py candidate.json

The worker JSON contains warm_wells and cold_wells (lists of well definitions),
global_demand (path relative to the model directory, with required warm_series
and cold_series names), month_for_day (one string key per simulated day), and
monthly_shares (month key -> EPANET node ID -> decoded share).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from epanet_input import WellControl, read_global_demand, write_candidate_input
from heat import HeatEnergy, calculate_heat_energy

TEMPLATE_NAME = "optimization-simplified.inp"
REPORT_NAME = "report.rpt"
OBJECTIVE_NAME = "net_heat_gj"
LOG_TAIL_BYTES = 4096


def evaluate_candidate(payload_path: Path) -> HeatEnergy:

    model_dir = Path(__file__).resolve().parent
    with payload_path.open(encoding="utf-8-sig") as stream:
        candidate = json.load(stream)
    if not isinstance(candidate, dict):
        raise ValueError("Expected a candidate JSON object.")

    warm_wells = [_well_control(well) for well in candidate["warm_wells"]]
    cold_wells = [_well_control(well) for well in candidate["cold_wells"]]
    demand_config = candidate["global_demand"]
    missing_series = {"warm_series", "cold_series"} - demand_config.keys()
    if missing_series:
        fields = ", ".join(sorted(missing_series))
        raise ValueError(f"Missing required fields in global_demand: {fields}.")
    demand_path = model_dir / demand_config["path"]
    daily_demand = read_global_demand(
        demand_path,
        warm_series=demand_config["warm_series"],
        cold_series=demand_config["cold_series"],
    )
    executable = shutil.which("runepanet")
    if executable is None:
        raise FileNotFoundError("runepanet was not found on the worker's PATH.")

    with TemporaryDirectory(prefix="minewater-") as directory:
        run_dir = Path(directory)
        candidate_inp = write_candidate_input(
            model_dir / TEMPLATE_NAME,
            run_dir / "candidate.inp",
            warm_wells=warm_wells,
            cold_wells=cold_wells,
            daily_demand=daily_demand,
            month_for_day=candidate["month_for_day"],
            monthly_shares=candidate["monthly_shares"],
        )
        _run_epanet(Path(executable).resolve(), candidate_inp, run_dir)
        return calculate_heat_energy(
            candidate_inp,
            [well.node_id for well in warm_wells],
            [well.node_id for well in cold_wells],
            run_directory=run_dir,
        )


def _well_control(well: dict) -> WellControl:
    return WellControl(
        node_id=well["node_id"],
        flow_pattern_id=well["flow_pattern_id"],
        max_production_rate_m3_h=well["max_production_rate_m3_h"],
        max_injection_rate_m3_h=well["max_injection_rate_m3_h"],
    )


def _run_epanet(executable: Path, candidate_inp: Path, run_dir: Path) -> None:
    log_path = run_dir / "runepanet.log"
    timeout_seconds = float(os.environ.get("WORKER_SIMULATION_TIMEOUT_SECONDS", "900"))
    with log_path.open("wb") as log:
        try:
            process = subprocess.run(
                [str(executable), candidate_inp.name, REPORT_NAME],
                cwd=run_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"runepanet exceeded the {timeout_seconds:g} s worker timeout."
            ) from exc

    with log_path.open("rb") as log:
        log.seek(max(0, log_path.stat().st_size - LOG_TAIL_BYTES))
        last_output = log.read().decode("utf-8", errors="replace").splitlines()

    report = run_dir / REPORT_NAME
    if process.returncode or not report.is_file() or report.stat().st_size == 0:
        details = "\n".join(line.strip() for line in last_output if line.strip())
        raise RuntimeError(
            f"runepanet failed or did not write a report (exit code {process.returncode})."
            f"\n{details}"
        )
    for line in last_output:
        if "EPANET ran with warnings" in line:
            print(line.strip(), file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_json", type=Path, help="JSON payload from the worker")
    args = parser.parse_args()
    try:
        heat = evaluate_candidate(args.candidate_json)
        if not math.isfinite(heat.net_heat_gj):
            raise ValueError("The net heat objective is not finite.")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Minewater simulation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Gross heat: {heat.gross_heat_gj:.6f} GJ")
    print(f"Gross cooling: {heat.gross_cooling_gj:.6f} GJ")
    print(f"EpanetConnector: Type:{OBJECTIVE_NAME}, Value:{heat.net_heat_gj:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
