# Auto-generated from notebook code cells #1 and #5
# Production model imported from model_PROD.py
# Source notebook: case_setup.ipynb

## Code cell 1
# import DARTS libraries
#from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from darts.engines import redirect_darts_output
from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec

redirect_darts_output("run_urgent.log")

# import other libraries
import os

import numpy as np
import pandas as pd
import psutil
from model_PROD import ProductionModel

import reservoir_modelling.helpers.helper_modelling as func


def print_mem(label=""):
    process = psutil.Process()
    mem_bytes = process.memory_info().rss
    mem_gb = mem_bytes / (1024 ** 3)
    print(f"[MEM] {label}: RSS = {mem_gb:.3f} GB")

## Code cell 5
# Production scenario run

# output file location
out_root = os.path.join(os.getcwd(), "output_PROD")
vtk_dir = os.path.join(out_root, "vtk_PROD")
os.makedirs(vtk_dir, exist_ok=True)  # creates both output_PROD and vtk_PROD if needed

# input file location
mesh_file = "geology/MESH_with_properties.vtu"
restart_file = "output_NS/solution_NS.csv"

# Simulation time
Dtimes = [1 * 365, 1 * 365, 1 * 365, 1 * 365, 1 * 365]

# Well control parameters
Qinj = 600.0          # ton/day
Tinj = 66 + 273.15     # K    - reference temperature (injection temperature)

# model run
m = ProductionModel(mesh_file=mesh_file, restart_file=restart_file, Qinj=Qinj, Tinj=Tinj)
m.init(verbose=True, output_folder=out_root)

#m.set_well_controls(Q_inj=Qinj, T_inj=Tinj)

m.params.max_ts = 365.

# Geomechanics initialization
stressState_file = "geomechanics/StressState.csv"
stress_df = pd.read_csv(stressState_file, sep=',')
fault_file = "geomechanics/faults.csv"
fault_df = pd.read_csv(fault_file, sep=',')
depth_reservoir = m.reservoir.global_data["depth"]

initcond_df = pd.read_csv(restart_file, sep=',', index_col=0)       # restart file contains only P and H with cell ID as index
initcond_df['T'] = _Backward1_T_Ph_vec(initcond_df['P'].to_numpy()/10, initcond_df['H'].to_numpy()/18.015)

fault_stress_df = func.stress_fault_df(fault_df, depth_reservoir, initcond_df, stress_df=stress_df)

# Extract initial stresses for VTK output (only once)
idx = fault_df["ID"].to_numpy(dtype=int).ravel()
SV_init = fault_stress_df['SV'].values / 1e5  # Convert Pa to bar
SH_init = fault_stress_df['SH'].values / 1e5
Sh_init = fault_stress_df['Sh'].values / 1e5

use_fem = True
if use_fem:  
    print_mem('In geomechanics init')
    m.init_fem_geomech(
        fault_df=fault_stress_df,
        P0=initcond_df['P'].values,
        T0=initcond_df['T'].values,
        E=9e9,
        nu=0.25,
        alpha_b=1.0,
        alpha_T=1e-5,
        rho=1300.0,
        solver_params={"rtol": 1e-7, "maxiter": 1500, "warm_start": True},
    )
print('Setting up VTK')
m.initialize_vtk_data(P=initcond_df['P'].values, T=initcond_df['T'].values, F=fault_df['ID'].values)  # update cell_data for vtk output with initial conditions

# Store initial stresses to VTK (these don't change during simulation)
m.cell_data["S0_11"][0][idx] = SV_init
m.cell_data["S0_22"][0][idx] = SH_init
m.cell_data["S0_33"][0][idx] = Sh_init

m.write_vtk(0, vtk_dir)
mu_crit = 0.6           # critical friction coefficient for fault reactivation
flow_rate_chop = 0.7    # flow rate reduction if fault reactivation occurs
t_cum = 0.0             # cumulative time tracker for reporting

print_mem('before time looping')
nx = len(m.hex_conn)
init_P = initcond_df['P'].values[:nx]
init_T = initcond_df['T'].values[:nx]
for i,t in enumerate(Dtimes):
    m.run(days=t, restart_dt=0,verbose=True)  
    m.output_to_vtk(ith_step=i+1,output_directory=vtk_dir,
                    output_properties=["temperature"], )  # pressure/enthalpy are primary vars
    
    # Getting primary variables
    X = np.asarray(m.physics.engine.X)
    P = X[0::2]      # pressure in bar
    H = X[1::2]      # enthalpy in kJ/kmol
    T = _Backward1_T_Ph_vec(P/10, H/18.015)                 # temperature in K
    out_csv = os.path.join(out_root, f"solution_PROD_{i+1}.csv")
    out_arr = np.column_stack((np.arange(P.size), P, H, T))
    np.savetxt(out_csv, out_arr, delimiter=",", header=",P,H,T", comments="")

    # Computing stress state on faults
    if use_fem:
        print_mem('inside use_fem (mu_vec)')
        u = m.fem_geo.solve(P, T)
        print_mem('done with solve_u_free')
        print('maximum displacement: ',u.max())
        mu_vec, principal_stresses, mu_tangent = m.fem_geo.compute_mu(P,T,u=u)
    else:
        solution_df = pd.DataFrame({'P':P,'H':H,'T':T})
        mu_vec, principal_stresses, mu_tangent = m.compute_analytical_stress_vect(fault_stress_df, stress_df=stress_df, solution_df=solution_df)
    
    # Update VTK cell data for fault cells
    m.cell_data["mu_vec"][0][idx] = mu_vec
    m.cell_data["sigma_11"][0][idx] = principal_stresses[:, 0]  # Max principal stress
    m.cell_data["sigma_22"][0][idx] = principal_stresses[:, 1]  # Intermediate principal stress
    m.cell_data["sigma_33"][0][idx] = principal_stresses[:, 2]  # Min principal stress
    m.cell_data["mu_tangent"][0][idx] = mu_tangent

    m.cell_data["dP"][0][:] = P[:nx] - init_P  # update dP for vtk output
    m.cell_data["dT"][0][:] = T[:nx] - init_T  # update dT for vtk output
    m.write_vtk(i+1, filename=vtk_dir+os.sep+'production_'+str(i+1))  # write vtk with updated mu_vec for visualization
    Max_mu = mu_vec.max()
    t_cum = t_cum + t
    print(f"Time {t_cum} days: Max friction coefficient on faults = {Max_mu:.3f}")

    # Check failure criteria
    if Max_mu >= mu_crit:
        print(f"FAULT REACTIVATION DETECTED at time {t} days with max mu={Max_mu:.3f} >= mu_crit={mu_crit:.3f}")
        Qinj = flow_rate_chop * Qinj
        m.Qinj = Qinj
        print(f"New injection rate: {Qinj:.2f} ton/day")
        m.set_well_controls()

print("Production model run complete.")