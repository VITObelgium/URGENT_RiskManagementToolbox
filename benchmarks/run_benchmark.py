import numpy as np
from interfaces.common import risk_management

benchmarks = {
    # "rastrigin": {
    #     "type": "single",
    #     "global_optimum": 0.0,
    #     "optimal_control_vector": [0.0, 0.0, 0.0, 0.0],
    #     "input_file": "rastrigin_benchmark.json",
    #     "model": "rastrigin_benchmark.zip",
    # },
    "zdt1": {
        "type": "multi",
        "input_file": "zdt1_benchmark.json",
        "model": "zdt1_benchmark.zip",
    },
}


def _get_control_vector(nested_cv: dict) -> np.ndarray:
    """Extract a flat coordinate array from a single nested control vector dict."""
    well_design = nested_cv["well_design"]
    coords_flat = []
    for _, well_data in well_design.items():
        coords_flat.append(well_data["wellhead"]["x"])
        coords_flat.append(well_data["wellhead"]["y"])
    return np.array(coords_flat, dtype=float)


def _get_control_vectors(result) -> list[np.ndarray]:
    """Return a list of flat control vectors.

    Single-objective: result[1] is a single nested dict  → returns [cv].
    Multi-objective:  result[1] is a list of nested dicts → returns [cv, ...].
    """
    raw = result[1]
    if isinstance(raw, list):
        return [_get_control_vector(cv) for cv in raw]
    return [_get_control_vector(raw)]


def _validate_zdt1_point(
    f1: float, f2: float, x: np.ndarray, tol: float = 1e-1
) -> None:
    """Validate that a single (f1, f2, x) triplet is on or near the ZDT1 Pareto front.

    Uses a relaxed tolerance (1e-2) because a PSO approximation will not hit the
    analytical front exactly — tight tolerances like 1e-9 would always fail.
    """
    if np.any(x < 0.0) or np.any(x > 1.0):
        raise AssertionError(f"ZDT1 variables must lie in [0,1], got x={x}.")

    # Analytical Pareto front: f2 = 1 - sqrt(f1), achieved when x[1:] == 0.
    expected_f2 = 1.0 - np.sqrt(f1)
    if not np.isclose(f2, expected_f2, rtol=tol, atol=tol):
        raise AssertionError(
            f"Point not on ZDT1 Pareto front: f1={f1:.6f}, f2={f2:.6f}, "
            f"expected f2≈{expected_f2:.6f} (tol={tol})."
        )


def _validate_zdt1_front(
    objectives: np.ndarray,
    control_vectors: list[np.ndarray],
    tol: float = 1e-2,
) -> None:
    """Validate the full Pareto front approximation for ZDT1.

    objectives:      shape (n_pareto, 2) — f1 and f2 for each solution.
    control_vectors: list of n_pareto flat arrays, one per Pareto solution.
    tol:             tolerance for proximity to the analytical Pareto front.

    Checks:
      1. objectives has 2 columns (ZDT1 is bi-objective).
      2. f1 values are in [0, 1].
      3. Each solution is approximately on the f2 = 1 - sqrt(f1) front.
      4. The front has reasonable spread (f1 range covers at least 50% of [0,1]).
    """
    if objectives.ndim != 2 or objectives.shape[1] != 2:
        raise AssertionError(
            f"Expected objectives shape (n_pareto, 2), got {objectives.shape}."
        )

    n_pareto = objectives.shape[0]
    if n_pareto != len(control_vectors):
        raise AssertionError(
            f"Mismatch: {n_pareto} objective rows but {len(control_vectors)} control vectors."
        )

    f1_vals = objectives[:, 0]
    f2_vals = objectives[:, 1]

    if np.any(f1_vals < 0.0) or np.any(f1_vals > 1.0):
        raise AssertionError(
            f"ZDT1 f1 values must lie in [0,1], got min={f1_vals.min():.4f} max={f1_vals.max():.4f}."
        )

    # Validate each point individually.
    failures = []
    for i, (f1, f2, x) in enumerate(zip(f1_vals, f2_vals, control_vectors)):
        try:
            _validate_zdt1_point(float(f1), float(f2), x, tol=tol)
        except AssertionError as e:
            failures.append(f"  solution {i}: {e}")

    if failures:
        raise AssertionError(
            f"{len(failures)}/{n_pareto} Pareto solutions failed ZDT1 front check:\n"
            + "\n".join(failures)
        )

    # Check spread: f1 range should cover a reasonable portion of [0, 1].
    f1_range = float(f1_vals.max() - f1_vals.min())
    if f1_range < 0.5:
        raise AssertionError(
            f"ZDT1 Pareto front has poor spread: f1 range = {f1_range:.4f} (expected >= 0.5)."
        )

    print(
        f"  ZDT1 front OK: {n_pareto} solutions, "
        f"f1 in [{f1_vals.min():.4f}, {f1_vals.max():.4f}], "
        f"f2 in [{f2_vals.min():.4f}, {f2_vals.max():.4f}]"
    )


if __name__ == "__main__":
    for k, v in benchmarks.items():
        print("==========================")
        print(f"Running benchmark {k}")
        print("==========================")
        try:
            r = risk_management(
                config_file=v["input_file"],
                model_file=v["model"],
            )

            # r[0]: ndarray shape (1,) single-obj or (n_pareto, n_obj) multi-obj
            # r[1]: single nested dict or list of nested dicts
            control_vectors = _get_control_vectors(r)

            if v["type"] == "single":
                global_optimum = v["global_optimum"]
                optimal_control_vector = v["optimal_control_vector"]
                # Single-objective: r[0] is shape (1,)
                assert np.isclose(
                    float(r[0][0]), global_optimum, rtol=1e-9, atol=1e-9
                ), f"Expected global optimum {global_optimum}, got {float(r[0][0]):.9f}"
                assert np.allclose(
                    control_vectors[0], optimal_control_vector, rtol=1e-9, atol=1e-9
                ), (
                    f"Control vector mismatch: {control_vectors[0]} vs {optimal_control_vector}"
                )
                print(f"  Single-objective OK: result={float(r[0][0]):.6f}")

            elif v["type"] == "multi":
                # Multi-objective: r[0] is shape (n_pareto, n_obj)
                objectives = np.atleast_2d(np.asarray(r[0], dtype=float))
                _validate_zdt1_front(objectives, control_vectors)

            else:
                raise ValueError(f"Unknown benchmark type: {v['type']}")

            print(f"Benchmark {k} PASSED.")

        except Exception as e:
            print(f"Benchmark {k} FAILED: {e}")
