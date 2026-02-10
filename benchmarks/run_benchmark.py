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


def _get_control_vector(result):
    well_design = result[1]["well_design"]
    coords_flat = []
    for _, well_data in well_design.items():
        coords_flat.append(well_data["wellhead"]["x"])
        coords_flat.append(well_data["wellhead"]["y"])
    return np.array(coords_flat, dtype=float)


def _validate_zdt1(objectives, control_vector, tol=1e-9):
    """
    objectives: iterable (f1, f2)
    control_vector: decision vector x
    """

    f1, f2 = objectives
    x = control_vector

    # ZDT1 requires x in [0,1]
    if np.any(x < 0.0) or np.any(x > 1.0):
        raise AssertionError("ZDT1 variables must lie in [0,1].")

    # Pareto-optimal structure: x2...xn == 0
    if not np.allclose(x[1:], 0.0, rtol=tol, atol=tol):
        raise AssertionError("ZDT1 Pareto-optimal condition violated: x[1:] != 0.")

    # Analytical Pareto front: f2 = 1 - sqrt(f1)
    expected_f2 = 1.0 - np.sqrt(f1)

    if not np.isclose(f2, expected_f2, rtol=tol, atol=tol):
        raise AssertionError("Point is not on ZDT1 Pareto front.")


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

            cv = _get_control_vector(r)

            if v["type"] == "single":
                global_optimum = v["global_optimum"]
                optimal_control_vector = v["optimal_control_vector"]

                assert np.isclose(r[0], global_optimum, rtol=1e-9, atol=1e-9)
                assert np.allclose(cv, optimal_control_vector, rtol=1e-9, atol=1e-9)

            elif v["type"] == "multi":
                objectives = np.asarray(r[0], dtype=float)
                assert len(objectives) == 2
                _validate_zdt1(objectives, cv)

            else:
                raise ValueError(f"Unknown benchmark type: {v['type']}")

        except Exception as e:
            print(f"Benchmark {k} failed: {e}")
