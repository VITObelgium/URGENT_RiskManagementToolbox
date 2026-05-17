###############
## Production model run
###############

import os
import shutil

# import other libraries
import numpy as np
import pandas as pd

# import RMT modules
# from connectors.common import SimulationResultType
from connectors.opendarts import (
    OpenDartsConnector,
    open_darts_input_configuration_injector,
)

# import DARTS libraries
from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.engines import set_num_threads

# import helper functions
import helpers.helper_heatproduction as func_heat
import helpers.helper_modelling as func

# import DARTS model
from model_PROD import ProductionModel


@open_darts_input_configuration_injector
def run_darts(well_data) -> None:
    set_num_threads(1)
    # output file location
    out_root = os.path.join(os.getcwd(), "output_PROD")
    vtk_dir = os.path.join(out_root, "vtk_PROD")
    os.makedirs(
        vtk_dir, exist_ok=True
    )  # creates both output_PROD and vtk_PROD if needed

    # input file location
    mesh_file = "input/mesh_with_properties.vtu"
    restart_file = "input/initial_condition.csv"

    # well names
    PROD = "PROD"
    INJ = "INJ"

    # Well control parameters
    Qinj = 600.0  # ton/day
    Tinj = 66 + 273.15  # K    - reference temperature (injection temperature)
    Qprod = Qinj  # ton/day   -- assuming balanced doublet for simplicity, but it can be different

    # model run
    m = ProductionModel(
        well_data=well_data,
        mesh_file=mesh_file,
        restart_file=restart_file,
        Qinj=Qinj,
        Tinj=Tinj,
        Qprod=Qprod,
    )
    m.init(
        verbose=True,
        output_folder=out_root,
    )

    # Simulation time
    Dtimes = [365] * 10    # 10 years period with fault reactivation check every year
    # Dtimes = [365 / 4] * 40  # total of approx 10 years with outputs every quarter year
    # m.params.max_ts = 365.0
    m.params.max_ts = 365.0 / 4

    # Geomechanics initialization
    stress_file = "input/stress_state.csv"
    stress_df = pd.read_csv(stress_file, sep=",")
    # fault_file = "input/faults.csv"
    # fault_df = pd.read_csv(fault_file, sep=",")
    fault_df = func.build_fault_df_from_reservoir(m.reservoir)
    depth_reservoir = m.reservoir.global_data["depth"]
    print(f"Fault dataframe built with {len(fault_df)} fault elements based on reservoir mesh and fault ID mapping.")

    initcond_df = pd.read_csv(
        restart_file, sep=",", index_col=0
    ) 

    fault_stress_df = func.stress_fault_df(
        fault_df, depth_reservoir, initcond_df, stress_df=stress_df
    )

    mu_crit = 0.6  # critical friction coefficient for fault reactivation
    flow_rate_chop = 0.7  # flow rate reduction if fault reactivation occurs
    t_cum = 0.0  # cumulative time tracker for reporting

    for i, t in enumerate(Dtimes):
        m.run(days=t, restart_dt=0, verbose=True)
        # m.output_to_vtk(
        #     ith_step=i + 1,
        #     output_directory=vtk_dir,
        #     output_properties=["temperature"],
        # )  # pressure/enthalpy are primary vars

        # Getting primary variables
        P = np.array(m.physics.engine.X[0::2], copy=False)  # pressure in bar
        H = np.array(m.physics.engine.X[1::2], copy=False)  # enthalpy in kJ/kmol
        T = _Backward1_T_Ph_vec(P / 10, H / 18.015)  # temperature in K
        solution_df = pd.DataFrame({"P": P, "H": H, "T": T})
        # solution_df.to_csv(
        #     os.path.join(out_root, f"solution_PROD_{i + 1}.csv"), sep=","
        # )

        # Computing stress state on faults
        mu_vec = m.compute_analytical_stress_vect(
            fault_stress_df, stress_df=stress_df, solution_df=solution_df
        )
        n_cells = m.reservoir.mesh.n_res_blocks
        mu_full = np.full(n_cells, np.nan, dtype=float)
        fault_ids = fault_stress_df["ID"].to_numpy(dtype=int)
        mu_full[fault_ids] = mu_vec

        func.output_darts_vtk_with_cell_prop(model=m, ith_step=i + 1, vtk_dir=vtk_dir, cell_prop=mu_full, field_name="mu_fault", output_properties=("temperature",),)

        # Fault reactivation check and well control adjustment
        Max_mu = mu_vec.max()
        t_cum = t_cum + t
        print(f"Time {t_cum} days: Max friction coefficient on faults = {Max_mu:.3f}")

        # Check failure criteria
        if Max_mu >= mu_crit:
            print(
                f"FAULT REACTIVATION DETECTED at time {t_cum} days with max mu={Max_mu:.3f} >= mu_crit={mu_crit:.3f}"
            )
            Qinj = flow_rate_chop * Qinj
            m.Qinj = Qinj
            m.Qprod = Qinj  # for balanced operation
            print(f"New injection rate: {Qinj:.2f} ton/day")
            m.set_well_controls()

    # Get and writting well vectors
    td = pd.DataFrame.from_dict(m.physics.engine.time_data)
    address = out_root + os.sep + "well_data_volumetric_mass_control.xlsx"
    writer = pd.ExcelWriter(address)
    td.to_excel(excel_writer=writer, sheet_name="Sheet1")
    writer.close()

    ## Cumulative heat production (MWy)
    Heat = func_heat.cumulative_heat(
        td, PROD, INJ
    )  # cumulative heat for a specific doublet: requires accurate well names for producers and injectors
    address = out_root + os.sep + "indicators.txt"
    Indicators = pd.DataFrame(
        {"Heat[MWy]": [Heat]}
    )  # We can store here different indicators
    np.savetxt(address, Indicators.values, fmt="%.1f", header="Heat[MWy]")

    OpenDartsConnector.broadcast_result("Heat", Heat)

    # move darts created pvd to vtk folder for better organization
    src = os.path.join(os.getcwd(), "solution.pvd")
    dst = os.path.join(vtk_dir, "solution.pvd")

    if os.path.exists(src):
        shutil.move(src, dst)
        func.fix_pvd_paths(dst)


if __name__ == "__main__":
    run_darts()
