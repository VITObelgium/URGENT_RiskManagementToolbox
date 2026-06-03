###############
## Production model run
###############

import os
import shutil
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# import other libraries
import numpy as np
import pandas as pd

# import RMT modules
# from connectors.common import SimulationResultType
from connectors.open_darts import (
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

    # disable multi-threaded runs
    set_num_threads(1)

    # output file location
    out_root = os.path.join(os.getcwd(), "output_PROD")
    vtk_dir = os.path.join(out_root, "vtk_PROD")
    os.makedirs(
        vtk_dir, exist_ok=True
    )  # creates both output_PROD and vtk_PROD if needed

    for stale_vtk in Path(vtk_dir).glob("production_*.vtu"):
        stale_vtk.unlink()
    for stale_pvd in Path(vtk_dir).glob("production*.pvd"):
        stale_pvd.unlink()

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
    )

    friction_method = m.friction_estimation_method
    use_fem = friction_method == "analytical" # 'fem' or 'analytical' 
    print(f"Friction estimation method: {friction_method}")

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

    # Extract initial principal stresses for VTK output (only once)
    SV_init = fault_stress_df["SV"].values / 1e5  # Convert Pa to bar
    SH_init = fault_stress_df["SH"].values / 1e5
    Sh_init = fault_stress_df["Sh"].values / 1e5

    if use_fem:
        print("Initializing FEM geomechanics module...")
        cache_dir = os.environ.get(
            "FEM_CACHE_DIR",
            "/home/pogacnij/DEVELOPER/URGENT_RiskManagementToolbox/log/fem_cache",
        )
        m.init_fem_geomech(
            fault_df=fault_stress_df,
            P0=initcond_df["P"].values,
            T0=initcond_df["T"].values,
            E=9e9,
            nu=0.3,
            alpha_b=1.0,
            alpha_T=1e-5,
            rho=1300.0,
            solver_params={"rtol": 1e-7, "maxiter": 1500, "warm_start": True},
            cache_dir=cache_dir,
            rebuild_cache=True,
        )
        print("FEM geomechanics initialization completed.")

    mu_crit = 0.35  # critical friction coefficient for fault reactivation
    flow_rate_chop = 0.7  # flow rate reduction if fault reactivation occurs
    t_cum = 0.0  # cumulative time tracker for reporting

    nx = len(m.hex_conn)
    init_P = initcond_df["P"].values[:nx]
    init_T = initcond_df["T"].values[:nx]
    n_cells = m.reservoir.mesh.n_res_blocks
    fault_ids = fault_stress_df["ID"].to_numpy(dtype=int)
    p0_fault_bar = fault_stress_df["P0"].values / 1e5

    # Initial effective principal stresses from total-stress profile and initial pore pressure.
    SV0_eff_init = SV_init - p0_fault_bar
    SH0_eff_init = SH_init - p0_fault_bar
    Sh0_eff_init = Sh_init - p0_fault_bar

    # Initial principal stress state exported as constant VTK fields [bar].
    sv0_full = np.full(n_cells, np.nan, dtype=float)
    shmax0_full = np.full(n_cells, np.nan, dtype=float)
    shmin0_full = np.full(n_cells, np.nan, dtype=float)
    sv0_eff_full = np.full(n_cells, np.nan, dtype=float)
    shmax0_eff_full = np.full(n_cells, np.nan, dtype=float)
    shmin0_eff_full = np.full(n_cells, np.nan, dtype=float)
    sv0_full[fault_ids] = SV_init
    shmax0_full[fault_ids] = SH_init
    shmin0_full[fault_ids] = Sh_init
    sv0_eff_full[fault_ids] = SV0_eff_init
    shmax0_eff_full[fault_ids] = SH0_eff_init
    shmin0_eff_full[fault_ids] = Sh0_eff_init

    for i, t in enumerate(Dtimes):
        m.run(days=t, restart_dt=0, verbose=True)

        # Getting primary variables
        P = np.array(m.physics.engine.X[0::2], copy=False)  # pressure in bar
        H = np.array(m.physics.engine.X[1::2], copy=False)  # enthalpy in kJ/kmol
        T = _Backward1_T_Ph_vec(P / 10, H / 18.015)  # temperature in K
        solution_df = pd.DataFrame({"P": P, "H": H, "T": T})
        # solution_df.to_csv(
        #     os.path.join(out_root, f"solution_PROD_{i + 1}.csv"), sep=","
        # )

        if use_fem:
            print("Solving FEM geomechanics...")
            u = m.fem_geo.solve(P, T)
            (
                mu_vec,
                principal_stresses,
                mu_tangent,
                principal_theta_deg,
                principal_azimuth_deg,
                sh_azimuth_rotation_deg,
            ) = m.fem_geo.compute_mu(P, T, u=u, return_orientation=True)

            dP_full = np.full(n_cells, np.nan, dtype=float)
            dT_full = np.full(n_cells, np.nan, dtype=float)
            dP_full[:nx] = P[:nx] - init_P
            dT_full[:nx] = T[:nx] - init_T

            mu_full = np.full(n_cells, np.nan, dtype=float)
            mu_tan_full = np.full(n_cells, np.nan, dtype=float)
            sv_eff_full = np.full(n_cells, np.nan, dtype=float)
            shmax_eff_full = np.full(n_cells, np.nan, dtype=float)
            shmin_eff_full = np.full(n_cells, np.nan, dtype=float)
            sh_az_full = np.full(n_cells, np.nan, dtype=float)

            mu_full[fault_ids] = mu_vec
            mu_tan_full[fault_ids] = mu_tangent
            sv_eff_full[fault_ids] = principal_stresses[:, 0]
            shmax_eff_full[fault_ids] = principal_stresses[:, 1]
            shmin_eff_full[fault_ids] = principal_stresses[:, 2]
            sh_az_full[fault_ids] = principal_azimuth_deg[:, 2]

            func.output_darts_vtk_with_cell_prop(
                model=m,
                ith_step=i + 1,
                vtk_dir=vtk_dir,
                custom_cell_props={
                    "dP": dP_full,
                    "dT": dT_full,
                    "mu_vec": mu_full,
                    "mu_tan": mu_tan_full,
                    "SV_eff": sv_eff_full,
                    "SH_eff": shmax_eff_full,
                    "Sh_eff": shmin_eff_full,
                    "Sh_azimuth_deg": sh_az_full,
                    "SV0": sv0_full,
                    "SH0": shmax0_full,
                    "Sh0": shmin0_full,
                    "SV0_eff": sv0_eff_full,
                    "SH0_eff": shmax0_eff_full,
                    "Sh0_eff": shmin0_eff_full,
                },
                output_properties=("temperature",),
            )
        else:
            mu_vec, principal_stresses, mu_tangent = m.compute_analytical_stress_vect(
                fault_stress_df, stress_df=stress_df, solution_df=solution_df
            )

            dP_full = np.full(n_cells, np.nan, dtype=float)
            dT_full = np.full(n_cells, np.nan, dtype=float)
            dP_full[:nx] = P[:nx] - init_P
            dT_full[:nx] = T[:nx] - init_T

            mu_full = np.full(n_cells, np.nan, dtype=float)
            mu_tan_full = np.full(n_cells, np.nan, dtype=float)
            sv_eff_full = np.full(n_cells, np.nan, dtype=float)
            shmax_eff_full = np.full(n_cells, np.nan, dtype=float)
            shmin_eff_full = np.full(n_cells, np.nan, dtype=float)

            mu_full[fault_ids] = mu_vec
            mu_tan_full[fault_ids] = mu_tangent
            sv_eff_full[fault_ids] = principal_stresses[:, 0]
            shmax_eff_full[fault_ids] = principal_stresses[:, 1]
            shmin_eff_full[fault_ids] = principal_stresses[:, 2]

            func.output_darts_vtk_with_cell_prop(
                model=m,
                ith_step=i + 1,
                vtk_dir=vtk_dir,
                custom_cell_props={
                    "dP": dP_full,
                    "dT": dT_full,
                    "mu_vec": mu_full,
                    "mu_tan": mu_tan_full,
                    "SV_eff": sv_eff_full,
                    "SH_eff": shmax_eff_full,
                    "Sh_eff": shmin_eff_full,
                    "SV0": sv0_full,
                    "SH0": shmax0_full,
                    "Sh0": shmin0_full,
                    "SV0_eff": sv0_eff_full,
                    "SH0_eff": shmax0_eff_full,
                    "Sh0_eff": shmin0_eff_full,
                },
                output_properties=("temperature",),
            )

        # Fault reactivation check and well control adjustment
        Max_mu = mu_vec.max()
        t_cum = t_cum + t
        if use_fem:
            print(f"Time {t_cum} days: Max displacement = {u.max():.3f}")
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

    # # Get and writting well vectors
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
