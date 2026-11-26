from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isclose, isfinite
from pathlib import Path

SECONDS_PER_DAY = 86_400
RATE_TOLERANCE_M3_H = 1e-8

Month = Hashable
MonthlyShares = Mapping[Month, Mapping[str, float]]


@dataclass(frozen=True)
class WellControl:
    node_id: str
    flow_pattern_id: str
    max_production_rate_m3_h: float
    max_injection_rate_m3_h: float


@dataclass(frozen=True)
class DailyDemand:
    warm_m3_h: Sequence[float]
    cold_m3_h: Sequence[float]


def read_global_demand(
    path: str | Path,
    *,
    warm_series: str,
    cold_series: str,
) -> DailyDemand:
    """Read the two signed daily series from the external demand text file."""
    for field, name in (("warm_series", warm_series), ("cold_series", cold_series)):
        if (
            not isinstance(name, str)
            or not name
            or any(character.isspace() for character in name)
            or ";" in name
        ):
            raise ValueError(f"{field} must be a single nonempty demand keyword.")
    if warm_series == cold_series:
        raise ValueError("Warm and cold demand must have different series names.")
    series: dict[str, list[float]] = {warm_series: [], cold_series: []}
    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = _fields(line)
            if not fields:
                continue
            name, *values = fields
            if name not in series:
                raise ValueError(
                    f"{path}:{line_number}: unexpected demand keyword {name!r}; "
                    f"expected {warm_series!r} or {cold_series!r}."
                )
            if not values:
                raise ValueError(f"{path}:{line_number}: no rate values for {name!r}.")
            try:
                series[name].extend(_number(value) for value in values)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc

    for name, values in series.items():
        if not values:
            raise ValueError(f"{path}: missing values for {name!r}.")
    return DailyDemand(series[warm_series], series[cold_series])


def write_candidate_input(
    template_path: str | Path,
    output_path: str | Path,
    *,
    warm_wells: Sequence[WellControl],
    cold_wells: Sequence[WellControl],
    daily_demand: DailyDemand,
    month_for_day: Sequence[Month],
    monthly_shares: MonthlyShares,
) -> Path:
    """Write only the candidate's well flow patterns and return its input path.

    The dispatcher supplies one month key per day and one mapping from EPANET
    node ID to decoded share per month. The same keys index both inputs. Rate
    units are m³/h with positive demand for production and negative demand for
    injection. Preserve a distinct output path for every candidate.
    """
    template_path, output_path = Path(template_path), Path(output_path)
    if template_path.resolve() == output_path.resolve():
        raise ValueError("The candidate output path must differ from the template path.")

    wells = (*warm_wells, *cold_wells)
    if not warm_wells or not cold_wells:
        raise ValueError("At least one warm well and one cold well are required.")
    if len({well.node_id for well in wells}) != len(wells):
        raise ValueError("Controlled well node IDs must be unique.")
    if len({well.flow_pattern_id for well in wells}) != len(wells):
        raise ValueError("Each controlled well needs a dedicated flow pattern.")
    for well in wells:
        caps = (well.max_production_rate_m3_h, well.max_injection_rate_m3_h)
        if any(not isfinite(cap) or cap < 0 for cap in caps):
            raise ValueError(f"Invalid rate cap for well {well.node_id!r}.")

    with template_path.open(encoding="utf-8", newline="") as stream:
        lines = stream.readlines()
    sections = _index_sections(lines)
    days, demand_multiplier = _daily_model_settings(sections)
    for label, count in (
        ("Warm demand", len(daily_demand.warm_m3_h)),
        ("Cold demand", len(daily_demand.cold_m3_h)),
        ("Day-to-month mapping", len(month_for_day)),
    ):
        if count != days:
            raise ValueError(f"{label}: expected {days} values, found {count}.")
    for day, (warm, cold) in enumerate(
        zip(daily_demand.warm_m3_h, daily_demand.cold_m3_h), start=1
    ):
        if not isfinite(warm) or not isfinite(cold):
            raise ValueError(f"Day {day}: demand rates must be finite.")
        if (warm > 0 and cold > 0) or (warm < 0 and cold < 0):
            raise ValueError(f"Day {day}: both groups have the same active role.")

    rates_per_multiplier = _well_bases(sections, wells, demand_multiplier)
    expected_nodes = {well.node_id for well in wells}
    month_keys = set(month_for_day)
    if set(monthly_shares) != month_keys:
        raise ValueError("Monthly shares must cover exactly the months in month_for_day.")
    for month, shares in monthly_shares.items():
        if shares.keys() != expected_nodes:
            raise ValueError(f"Month {month!r}: expected one share for each controlled well.")
        if any(not isfinite(share) or not 0 <= share <= 1 for share in shares.values()):
            raise ValueError(f"Month {month!r}: all well shares must be between 0 and 1.")
        for label, group in (("warm", warm_wells), ("cold", cold_wells)):
            total = fsum(shares[well.node_id] for well in group)
            if not isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"Month {month!r}: {label} shares sum to {total}, expected 1.")
            
    replacement_patterns: dict[str, list[float]] = {}
    for group, totals in (
        (warm_wells, daily_demand.warm_m3_h),
        (cold_wells, daily_demand.cold_m3_h),
    ):
        for well in group:
            values = []
            for day, (month, total) in enumerate(zip(month_for_day, totals), start=1):
                rate = total * monthly_shares[month][well.node_id]
                cap = (
                    well.max_production_rate_m3_h if rate >= 0
                    else well.max_injection_rate_m3_h
                )
                if abs(rate) > cap + RATE_TOLERANCE_M3_H:
                    role = "production" if rate >= 0 else "injection"
                    raise ValueError(
                        f"Day {day}, month {month!r}: {well.node_id!r} {role} rate "
                        f"{abs(rate):g} m³/h exceeds its cap of {cap:g} m³/h."
                    )
                values.append(_number(rate / rates_per_multiplier[well.node_id]))
            # At Duration, EPANET reads one more pattern value for the final
            # report row. Hold the final daily rate there to prevent wraparound.
            replacement_patterns[well.flow_pattern_id] = [*values, values[-1]]

    new_lines = _replace_patterns(lines, sections, replacement_patterns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        stream.writelines(new_lines)
    return output_path


def _index_sections(lines: Sequence[str]) -> dict[str, list[tuple[int, list[str]]]]:
    sections: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    section = ""
    for index, line in enumerate(lines):
        fields = _fields(line.removeprefix("\ufeff"))
        if not fields:
            continue
        if fields[0].startswith("["):
            section = fields[0].upper()
        else:
            sections[section].append((index, fields))
    return sections


def _daily_model_settings(
    sections: Mapping[str, list[tuple[int, list[str]]]],
) -> tuple[int, float]:
    times = {}
    for _, fields in sections.get("[TIMES]", []):
        key = "DURATION" if fields[0].upper() == "DURATION" else " ".join(fields[:2]).upper()
        if key in {"DURATION", "PATTERN TIMESTEP", "PATTERN START"}:
            times[key] = _seconds(fields[-1])

    duration, timestep = times.get("DURATION", 0), times.get("PATTERN TIMESTEP")
    if timestep != SECONDS_PER_DAY or times.get("PATTERN START", 0) != 0:
        raise ValueError("Daily demand requires a 24-hour pattern timestep and zero pattern start.")
    if duration <= 0 or duration % SECONDS_PER_DAY:
        raise ValueError("EPANET Duration must be a positive whole number of days.")

    options = [fields for _, fields in sections.get("[OPTIONS]", [])]
    units = next((fields[1].upper() for fields in options if fields[0].upper() == "UNITS"), None)
    if units != "CMH":
        raise ValueError("Expected EPANET flow units CMH (m³/h).")
    multiplier = next(
        (
            _number(fields[2]) for fields in options
            if " ".join(fields[:2]).upper() == "DEMAND MULTIPLIER"
        ),
        1.0,
    )
    if multiplier <= 0:
        raise ValueError("EPANET Demand Multiplier must be positive.")
    return int(duration // SECONDS_PER_DAY), multiplier


def _well_bases(
    sections: Mapping[str, list[tuple[int, list[str]]]],
    wells: Sequence[WellControl],
    multiplier: float,
) -> dict[str, float]:
    junctions = [fields for _, fields in sections.get("[JUNCTIONS]", [])]
    by_node = {fields[0]: fields for fields in junctions}
    controlled_nodes = {well.node_id for well in wells}
    assignments = {well.flow_pattern_id: well.node_id for well in wells}
    known_patterns = {fields[0] for _, fields in sections.get("[PATTERNS]", [])}
    shared = {
        fields[3] for fields in junctions
        if len(fields) > 3 and fields[3] in assignments and assignments[fields[3]] != fields[0]
    }
    for section, pattern_column in (
        ("[RESERVOIRS]", 2), ("[SOURCES]", 3), ("[DEMANDS]", 2),
    ):
        for _, fields in sections.get(section, []):
            if section == "[DEMANDS]" and fields[0] in controlled_nodes:
                raise ValueError(f"Controlled well {fields[0]!r} has a [DEMANDS] entry.")
            if len(fields) > pattern_column and fields[pattern_column] in assignments:
                shared.add(fields[pattern_column])
    for _, fields in sections.get("[OPTIONS]", []):
        if fields[0].upper() == "PATTERN" and fields[-1] in assignments:
            shared.add(fields[-1])
    for _, fields in sections.get("[ENERGY]", []):
        if len(fields) >= 3 and fields[-2].upper() == "PATTERN" and fields[-1] in assignments:
            shared.add(fields[-1])
    if shared:
        raise ValueError(f"Controlled flow patterns also used elsewhere: {', '.join(sorted(shared))}.")

    bases = {}
    for well in wells:
        fields = by_node.get(well.node_id, [])
        if len(fields) < 4 or fields[3] != well.flow_pattern_id:
            raise ValueError(
                f"Junction {well.node_id!r} must use pattern {well.flow_pattern_id!r}."
            )
        if well.flow_pattern_id not in known_patterns:
            raise ValueError(f"Missing [PATTERNS] entry for {well.flow_pattern_id!r}.")
        base = _number(_number(fields[2]) * multiplier)
        if base == 0:
            raise ValueError(f"Controlled well {well.node_id!r} has zero base demand.")
        bases[well.node_id] = base
    return bases


def _replace_patterns(
    lines: Sequence[str],
    sections: Mapping[str, list[tuple[int, list[str]]]],
    patterns: Mapping[str, Sequence[float]],
) -> list[str]:
    controlled_rows = {
        index: fields[0]
        for index, fields in sections["[PATTERNS]"]
        if fields[0] in patterns
    }
    new_lines = []
    written = set()
    for index, line in enumerate(lines):
        name = controlled_rows.get(index)
        if name is None:
            new_lines.append(line)
        elif name not in written:
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            values = patterns[name]
            for start in range(0, len(values), 12):
                chunk = " ".join(f"{value:.17g}" for value in values[start:start + 12])
                new_lines.append(f"{name} {chunk}{newline}")
            written.add(name)
    return new_lines


def _fields(line: str) -> list[str]:
    return line.partition(";")[0].split()


def _seconds(value: str) -> float:
    parts = [_number(part) for part in value.split(":")]
    hours, minutes, seconds = parts + [0] * (3 - len(parts))
    return hours * 3600 + minutes * 60 + seconds


def _number(value: str | float) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"Expected a finite number, got {value!r}.")
    return number