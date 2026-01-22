import numpy as np

from interfaces.common import risk_management

benchmarks = {
    "rastrigin": {
        "global_optimum": 0.0,
        "optimal_control_vector": [0.0, 0.0, 0.0, 0.0],
        "input_file": "rastrigin_benchmark.json",
        "model": "rastrigin_benchmark.zip",
    }
}


def _get_control_vector(result):
    well_placement = result[1]["well_placement"]
    coords_flat = []
    for well_name, well_data in well_placement.items():
        coords_flat.append(well_data["wellhead"]["x"])
        coords_flat.append(well_data["wellhead"]["y"])
    return np.array(coords_flat)


if __name__ == "__main__":
    for k, v in benchmarks.items():
        global_optimum = v["global_optimum"]
        optimal_control_vector = v["optimal_control_vector"]
        input_file = v["input_file"]
        model = v["model"]

        print("==========================")
        print(f"Running benchmark {k}")
        print("==========================")
        r = risk_management(config_file=input_file, model_file=model)

        assert np.isclose(r[0], global_optimum, rtol=1e-9, atol=1e-9)
        cv = _get_control_vector(r)
        assert np.allclose(cv, optimal_control_vector, rtol=1e-9, atol=1e-9)
