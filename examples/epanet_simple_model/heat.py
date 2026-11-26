from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from itertools import chain, pairwise
from math import fsum, isfinite
from pathlib import Path

# Water density (kg/m³) times specific heat (kJ/kg/K), converted to GJ/m³/K.
WATER_HEAT_CAPACITY_GJ_M3_K = 1_000.0 * 4.18 / 1_000_000.0
NODE_RESULTS = re.compile(r"^\s*Node Results at (\d+:\d\d:\d\d) hrs:")
LINK_RESULTS = re.compile(r"^\s*Link Results\b")


@dataclass(frozen=True)
class HeatEnergy:
    gross_heat_gj: float
    gross_cooling_gj: float

    @property
    def net_heat_gj(self) -> float:
        return self.gross_heat_gj - self.gross_cooling_gj


@dataclass(frozen=True)
class WellResult:
    rate_m3_h: float
    temperature_c: float


@dataclass(frozen=True)
class EpanetInputs:
    duration_s: int
    report_step_s: int
    pattern_step_s: int
    pattern_start_s: int
    sources: dict[str, tuple[str, float, str | None]]
    patterns: dict[str, list[float]]

    def injection_temperature(self, node: str, elapsed_s: int) -> float:
        if node not in self.sources:
            raise ValueError(f"Injecting well {node!r} has no [SOURCES] entry.")
        source_type, temperature_c, pattern_id = self.sources[node]
        if source_type != "CONCEN":
            raise ValueError(
                f"Well {node!r} has source type {source_type}; expected CONCEN."
            )
        if pattern_id is None:
            return temperature_c
        if not self.patterns.get(pattern_id):
            raise ValueError(f"Well {node!r} uses missing source pattern {pattern_id!r}.")
        values = self.patterns[pattern_id]
        index = (elapsed_s + self.pattern_start_s) // self.pattern_step_s
        return temperature_c * values[index % len(values)]


def calculate_heat_energy(
    inp_path: str | Path,
    warm_well_ids: Collection[str],
    cold_well_ids: Collection[str],
    *,
    run_directory: str | Path = ".",
) -> HeatEnergy:
    """Score report.rpt in run_directory using the INP used for that candidate.

    Pass the EPANET node IDs for each well group. Each group needs at least one
    well. Reported demands determine operating mode and well flow rates, with
    positive demand indicating production. Each row represents the start of its
    reporting interval, so the final timestamp adds no volume or energy.
    """
    warm = tuple(warm_well_ids)
    cold = tuple(cold_well_ids)
    nodes = set(warm + cold)
    if not warm or not cold:
        raise ValueError("At least one warm well and one cold well are required.")
    if len(nodes) != len(warm) + len(cold):
        raise ValueError("Well node IDs must be unique within and across groups.")

    model = _read_epanet_inputs(Path(inp_path), nodes)
    report_path = Path(run_directory) / "report.rpt"
    heating, cooling = [], []
    samples = _read_report(report_path, nodes)
    first = next(samples, None)
    if first is None or first[0] != 0:
        raise ValueError(f"{report_path}: expected node results at simulation time zero.")

    last_time = 0
    for (start_s, results), (end_s, _) in pairwise(chain((first,), samples)):
        if end_s - start_s != model.report_step_s or end_s > model.duration_s:
            raise ValueError(
                f"{report_path}: unexpected interval from {start_s} to {end_s} seconds."
            )
        last_time = end_s
        warm_producers, warm_injectors = _split_well_roles(warm, results, start_s)
        cold_producers, cold_injectors = _split_well_roles(cold, results, start_s)
        if (warm_producers and cold_producers) or (warm_injectors and cold_injectors):
            raise ValueError(f"Warm and cold groups have the same active role at {start_s} s.")
        factor = WATER_HEAT_CAPACITY_GJ_M3_K * (end_s - start_s) / 3600
        if warm_producers:
            return_c = _weighted_injection_temperature(cold_injectors, results, model, start_s)
            heating.append(factor * fsum(
                results[node].rate_m3_h * (results[node].temperature_c - return_c)
                for node in warm_producers
            ))
        if cold_producers:
            return_c = _weighted_injection_temperature(warm_injectors, results, model, start_s)
            cooling.append(factor * fsum(
                results[node].rate_m3_h * (return_c - results[node].temperature_c)
                for node in cold_producers
            ))
    if last_time != model.duration_s:
        raise ValueError(
            f"{report_path}: ends at {last_time} s, expected {model.duration_s} s."
        )
    return HeatEnergy(fsum(heating), fsum(cooling))



def _split_well_roles(
    wells: tuple[str, ...], results: dict[str, WellResult], elapsed_s: int
) -> tuple[list[str], list[str]]:
    producers = [node for node in wells if results[node].rate_m3_h > 0]
    injectors = [node for node in wells if results[node].rate_m3_h < 0]
    if producers and injectors:
        raise ValueError(f"Mixed production and injection within one group at {elapsed_s} s.")
    return producers, injectors


def _weighted_injection_temperature(
    injectors: list[str], results: dict[str, WellResult], model: EpanetInputs, elapsed_s: int
) -> float:
    if not injectors:
        raise ValueError(f"Production at {elapsed_s} s has no active return injectors.")
    total_rate = fsum(-results[node].rate_m3_h for node in injectors)
    return fsum(
        -results[node].rate_m3_h * model.injection_temperature(node, elapsed_s)
        for node in injectors
    ) / total_rate


def _read_epanet_inputs(path: Path, nodes: set[str]) -> EpanetInputs:
    section = ""
    times = {}
    patterns = defaultdict(list)
    sources = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            content = line.partition(";")[0].strip()
            if content.startswith("["):
                section = content.upper()
                continue
            fields = content.split()
            if not fields:
                continue
            if section == "[TIMES]":
                duration = fields[0].lower() == "duration"
                key_length = 1 if duration else 2
                key = " ".join(fields[:key_length]).lower()
                if key in {"duration", "report timestep", "pattern timestep", "pattern start"}:
                    times[key] = _elapsed_seconds(fields[key_length])
            elif section == "[SOURCES]" and fields[0] in nodes:
                node, source_type, temperature, *pattern = fields
                if node in sources:
                    raise ValueError(f"Duplicate [SOURCES] entry for {node!r} in {path}.")
                sources[node] = (
                    source_type.upper(), _number(temperature), pattern[0] if pattern else None
                )
            elif section == "[PATTERNS]":
                patterns[fields[0]].extend(_number(value) for value in fields[1:])

    try:
        duration_s = times["duration"]
        report_step_s = times["report timestep"]
        pattern_step_s = times["pattern timestep"]
    except KeyError as exc:
        raise ValueError(f"{path}: missing {exc.args[0]!r} in [TIMES].") from exc
    pattern_start_s = times.get("pattern start", 0)
    if min(duration_s, report_step_s, pattern_step_s) <= 0 or duration_s % report_step_s:
        raise ValueError(f"{path}: expected positive time settings and a report at Duration.")
    if pattern_step_s % report_step_s or pattern_start_s % report_step_s:
        raise ValueError(f"{path}: pattern boundaries must coincide with report timestamps.")
    return EpanetInputs(duration_s, report_step_s, pattern_step_s, pattern_start_s, sources, dict(patterns))


def _read_report(path: Path, nodes: set[str]) -> Iterator[tuple[int, dict[str, WellResult]]]:
    time_s = None
    rows: dict[str, WellResult] = {}
    columns: tuple[int, int] | None = None
    in_node_table = False
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            match = NODE_RESULTS.match(line)
            if match:
                next_time = _elapsed_seconds(match[1])
                if time_s is not None and next_time != time_s:
                    _check_report_wells(path, rows, nodes, time_s)
                    yield time_s, rows
                    rows = {}
                time_s = next_time
                in_node_table = True
                columns = None
            elif LINK_RESULTS.match(line):
                in_node_table = False
            elif in_node_table and line[2:17].strip() == "Node":
                units = [line[i:i + 10].strip().lower() for i in range(17, len(line), 10)]
                try:
                    columns = units.index("m3/h"), units.index("degc")
                except ValueError as exc:
                    raise ValueError(f"{path}: expected report units m3/h and degC.") from exc
            elif in_node_table and columns is not None:
                fields = line.split(maxsplit=1)
                if not fields or fields[0] not in nodes:
                    continue
                node = fields[0]
                if node in rows:
                    raise ValueError(f"{path}: duplicate results for {node!r} at {time_s} s.")
                offset = 2 + max(15, len(node))
                demand_idx, temperature_idx = columns
                demand_start = offset + 10 * demand_idx
                temperature_start = offset + 10 * temperature_idx
                rows[node] = WellResult(
                    _number(line[demand_start:demand_start + 10]),
                    _number(line[temperature_start:temperature_start + 10]),
                )
    if time_s is not None:
        _check_report_wells(path, rows, nodes, time_s)
        yield time_s, rows


def _check_report_wells(
    path: Path, rows: dict[str, WellResult], nodes: set[str], time_s: int
) -> None:
    missing = nodes - rows.keys()
    if missing:
        raise ValueError(f"{path}: missing well results at {time_s} s: {', '.join(sorted(missing))}.")


def _elapsed_seconds(value: str) -> int:
    if ":" not in value:
        return round(_number(value) * 3600)
    parts = [int(part) for part in value.split(":")]
    hours, minutes, seconds = parts + [0] * (3 - len(parts))
    return 3600 * hours + 60 * minutes + seconds


def _number(value: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"Expected a finite number, got {value!r}.")
    return number
