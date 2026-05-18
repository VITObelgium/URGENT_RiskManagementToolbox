# Auto-generated from notebook code cells #1 and #4
# Source notebook: case_setup.ipynb

## Code cell 1
# import DARTS libraries

from darts.engines import redirect_darts_output
from darts.models.darts_model import DartsModel
from darts.physics.geothermal.physics import Geothermal

#from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from darts.physics.geothermal.property_container import PropertyContainer
from darts.reservoirs.struct_reservoir import StructReservoir
from iapws import IAPWS95

redirect_darts_output("run_urgent.log")

# import other libraries
import meshio
import numpy as np
import pandas as pd
from fem_geomechanics import FEMGeomechanics
from iapws.iapws95 import IAPWS95

import reservoir_modelling.helpers.helper_modelling as func

## Code cell 4
# Production scenario model setup

class ProductionModel(DartsModel):

    def __init__(self, n_points=128, mesh_file='', restart_file='', Qinj=0.0, Tinj=66+273.15):
        
        super().__init__()
                
    # ---- enforce required inputs ----

        if not mesh_file:
            print("[ERROR] For production scenario, mesh_file (.vtu) is required.")
            raise ValueError("For production scenario, mesh_file (.vtu) is required.")       

        if not restart_file:
            print("[ERROR] For production scenario, restart_file (.csv) is required.")
            raise ValueError("For production scenario, restart_file (.csv) is required.")

        self.mesh_file = mesh_file
        self.restart_file = restart_file

        self.Qinj = Qinj
        self.Tinj = Tinj

        self.timer.node["initialization"].start()

        # reservoir grid and properties
        self.set_reservoir()
        
        # definition of the physics (geothermal)
        self.set_physics(n_points=n_points)

        # initial conditions
        df_inc = pd.read_csv(self.restart_file,sep=',')
        self.df_inc = df_inc
        
        self.initial_values = {self.physics.vars[0]:self.df_inc['P'].values,
                               self.physics.vars[1]:self.df_inc['H'].values}

        print('Done with model initializing.')  


        # timestep parameters
        self.params.first_ts = 1e-3
        self.params.mult_ts = 2
        self.params.max_ts = 365        # days

        # nonlinear and linear solver tolerance
        self.params.tolerance_newton = 1e-2
        self.params.tolerance_linear = 1e-4

        self.timer.node['initialization'].stop()

    ## Structured reservoir setup
    def set_reservoir(self):
        self.mesh = meshio.read(self.mesh_file)
        blk = 0                                 # hexahedron cell block index
        pts = self.mesh.points
        self.pts = pts

        # Structured axes from input mesh
        x_edges = np.unique(pts[:, 0]); x_edges.sort()
        y_edges = np.unique(pts[:, 1]); y_edges.sort()
        z_edges = np.unique(pts[:, 2]); z_edges.sort()

        dx_raw = np.diff(x_edges)
        dy_raw = np.diff(y_edges)
        dz_raw = np.diff(z_edges)

        nx, ny, nz = len(dx_raw), len(dy_raw), len(dz_raw)

        DX3 = dx_raw[:, None, None] * np.ones((nx, ny, nz))
        DY3 = np.ones((nx, ny, nz)) * dy_raw[None, :, None]
        DZ3 = np.ones((nx, ny, nz)) * dz_raw[None, None, :]

        dx_full = DX3.ravel(order="F")
        dy_full = DY3.ravel(order="F")
        dz_full = DZ3.ravel(order="F")

        x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
        z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])

        # geometry for initial conditions (centroids)
        hex_conn = self.mesh.cells[blk].data                               
        cell_pts = self.pts[hex_conn]                        
        centroids = cell_pts.mean(axis=1)
        self.centroids = centroids
        self.hex_conn = hex_conn

        depth_raw = np.asarray(centroids[:, 2], dtype=float)
        self.ztop = float(depth_raw.min()) # define "top" as shallowest centroid (smallest z) - DARTS follows right-hand coordinate system with z positive downwards!

        # Check if mesh is pure structured cartesian grid
        n_cells = centroids.shape[0]
        if n_cells != nx * ny * nz: 
            raise ValueError(
                f"VTU has {n_cells} hex cells but nx*ny*nz = {nx*ny*nz}. "
                "Mesh is not a pure structured cartesian grid." )
        
        # read properties from mesh cell data
        #op_num = np.asarray(self.mesh.cell_data["num"][blk], dtype=float)
        permx_raw  = np.asarray(self.mesh.cell_data["permx"][blk], dtype=float)
        permy_raw  = np.asarray(self.mesh.cell_data["permy"][blk], dtype=float)
        permz_raw  = np.asarray(self.mesh.cell_data["permz"][blk], dtype=float)
        poro_raw   = np.asarray(self.mesh.cell_data["pore"][blk],  dtype=float)
        hcap_raw   = np.asarray(self.mesh.cell_data["hcap"][blk],  dtype=float)        
        rcond_raw  = np.asarray(self.mesh.cell_data["cond"][blk], dtype=float) * 24 * 3600 / 1000 # Unit conversion for cond * 24*3600/1000 [[kJ/(m-d-K)]]
        
        # property mapping from vtu to DARTS structure grid
        i = np.searchsorted(x_cent, self.centroids[:, 0]); i = np.clip(i, 0, nx-1)
        j = np.searchsorted(y_cent, self.centroids[:, 1]); j = np.clip(j, 0, ny-1)
        k = np.searchsorted(z_cent, self.centroids[:, 2]); k = np.clip(k, 0, nz-1)
        flat = i + nx * (j + ny * k)

        depth = np.empty(nx * ny * nz, dtype=float)
        permx = np.empty(nx * ny * nz, dtype=float)
        permy = np.empty(nx * ny * nz, dtype=float)
        permz = np.empty(nx * ny * nz, dtype=float)
        poro  = np.empty(nx * ny * nz, dtype=float)
        hcap  = np.empty(nx * ny * nz, dtype=float)
        rcond = np.empty(nx * ny * nz, dtype=float)

        depth[flat] = depth_raw
        permx[flat] = permx_raw
        permy[flat] = permy_raw
        permz[flat] = permz_raw
        poro[flat]  = poro_raw
        hcap[flat]  = hcap_raw
        rcond[flat] = rcond_raw

        self.reservoir = StructReservoir(
            self.timer, nx, ny, nz, dx_full, dy_full, dz_full,
            permx=permx, permy=permy, permz=permz, poro=poro, depth=depth,
            rcond=rcond, hcap=hcap,
            cache=False, )
        self.reservoir.global_data["depth"] = depth

        # init_reservoir does self.discretize() as well but also stores the mesh as self.reservoir.mesh
        # self.reservoir.discretize()
        self.reservoir.init_reservoir(verbose=True)    

        # set boundary volumes to large value to simulate no-flow boundaries on top and bottom sides
        self.reservoir.boundary_volumes["xy_minus"] = 1e25
        self.reservoir.boundary_volumes["xy_plus"] = 1e25
        return
    
    ## Setup physics (geothermal)    
    def set_physics(self, n_points, verbose: bool = False):
        # create pre-defined physics for geothermal
        self.property_container = PropertyContainer()
        self.property_container.output_props = {'temperature': lambda: self.property_container.temperature}
        self.physics = Geothermal(self.timer, n_points, min_p=1, max_p=500, min_e=1000, max_e=20000, cache=True)
        self.physics.add_property_region(self.property_container)

        return    
    
    # custom vtk output
    def initialize_vtk_data(self,P=None, T=None, F=None):
        numele = len(self.hex_conn)
        
        self.point_data = {}
        self.point_data['Random'] = np.random.rand(len(self.pts)) # just placeholder if point data required (displacements eventually)
        
        self.cell_data = {}
        self.cell_data['Pressure'] = [P[:len(self.hex_conn)]]  # pressure is primary var 0
        self.cell_data['Temperature'] = [T[:len(self.hex_conn)]]  # temperature from property container
        self.cell_data['dP'] = [np.zeros(numele)]
        self.cell_data['dT'] = [np.zeros(numele)]
        self.cell_data['mu_vec'] = [np.zeros(numele)]  # for reporting reactivation criteria
        self.cell_data['sigma_11'] = [np.zeros(numele)]  # diagonal stress component 1
        self.cell_data['sigma_22'] = [np.zeros(numele)]  # diagonal stress component 2
        self.cell_data['sigma_33'] = [np.zeros(numele)]  # diagonal stress component 3
        self.cell_data['S0_11'] = [np.zeros(numele)]  # initial diagonal stress component 1
        self.cell_data['S0_22'] = [np.zeros(numele)]  # initial diagonal stress component 2
        self.cell_data['S0_33'] = [np.zeros(numele)]  # initial diagonal stress component 3
        self.cell_data['mu_tangent'] = [np.zeros(numele)]  # tangent modulus based on stress state
        F = np.asarray(F).ravel().astype(int)   # make it 1D int indices
        arr = np.zeros(numele, dtype=np.int8)
        arr[F] = 1
        self.cell_data["FaultID"] = [arr]

    def write_vtk(self, step, filename):
        mesh = meshio.Mesh(self.pts,
                           [("hexahedron",self.hex_conn)],
                           point_data=self.point_data,
                           cell_data=self.cell_data
                           )
        mesh.write(filename+'.vtu')
        return
        
    # ---- additional method for production scenario setup and geomechanics ----
    def init_fem_geomech(
        self,
        fault_df,
        P0,
        T0,
        E,
        nu,
        alpha_b,
        alpha_T,
        rho,
        bc_params=None,
        solver_params=None,
    ):
        """
        Initialize FEM geomechanics backend.

        Parameters can be scalars or 1D arrays of length = #hex cells.
        - E: Young's modulus
        - nu: Poisson ratio
        - alpha_b: Biot coefficient
        - alpha_T: coefficient of thermal expansion (CTE)
        - rho: density for body forces (e.g. gravity)
        """
        if not hasattr(self, "hex_conn") or not hasattr(self, "pts"):
            raise RuntimeError("init_fem_geomech() called before set_reservoir(): self.pts/self.hex_conn not found")

        n_cells = self.hex_conn.shape[0]

        def as_cell_array(x, name):
            arr = np.asarray(x, dtype=float)
            if arr.ndim == 0:
                out = np.empty(n_cells, dtype=float)
                out.fill(float(arr))
                return out
            if arr.ndim == 1 and arr.size == n_cells:
                return np.ascontiguousarray(arr)
            raise ValueError(f"{name} must be a scalar or a 1D array of length {n_cells}. Got shape {arr.shape}")

        E_arr = as_cell_array(E, "E")
        nu_arr = as_cell_array(nu, "nu")
        alpha_b_arr = as_cell_array(alpha_b, "alpha_b")
        alpha_T_arr = as_cell_array(alpha_T, "alpha_T")
        rho_arr = as_cell_array(rho, "rho")

        P0 = np.ascontiguousarray(np.asarray(P0, dtype=float).reshape(-1)[:n_cells])
        T0 = np.ascontiguousarray(np.asarray(T0, dtype=float).reshape(-1)[:n_cells])
        if P0.size != n_cells:
            raise ValueError(f"P0 must contain at least {n_cells} values, got {P0.size}")
        if T0.size != n_cells:
            raise ValueError(f"T0 must contain at least {n_cells} values, got {T0.size}")

        merged_solver_params = {"rtol": 1e-7, "maxiter": 1500, "warm_start": True}
        if solver_params:
            merged_solver_params.update(solver_params)

        self.fem_geo = FEMGeomechanics(
            pts=self.pts,
            hex_conn=self.hex_conn,
            fault_df=fault_df,
            P0=P0,
            T0=T0,
            E=E_arr,
            nu=nu_arr,
            alpha_b=alpha_b_arr,
            alpha_T=alpha_T_arr,
            rho=rho_arr,
            bc_params=bc_params or {},
            solver_params=merged_solver_params,
        )
        self.fem_geo.setup()

    ## Setup wells and perforations
    def set_wells(self):

        #iw = [9, 14]
        #jw = [12, 4]
        iw = [34,48]
        jw = [41,11]

        # add injector well
        self.reservoir.add_well("MOL-GT-02")
        for k in range(8, 17):
            self.reservoir.add_perforation("MOL-GT-02", (iw[1], jw[1], k+1))  # DARTS uses 1-based indexing for perforations

        # add producer well
        self.reservoir.add_well("MOL-GT-01")
        for k in range(8, 17):
            self.reservoir.add_perforation("MOL-GT-01", (iw[0], jw[0], k+1))  # DARTS uses 1-based indexing for perforations


    ## Setup well injection and production controls    
    def set_well_controls(self):
        # if Q_inj is not None:
        #     self.Qinj = float(Q_inj)
        # if T_inj is not None:
        #     self.Tinj = float(T_inj)
        injection_rate = self.Qinj  # ton/day
        press = 321.0           # bar  - reference pressure 
        temp = self.Tinj            # K    - reference temperature (injection temperature) 

        # Mass to volumetric rate conversion
        rho = IAPWS95(P=press/10.0, T=temp).rho  # density in kg/m3
        vol_rate = injection_rate * 1000.0 / rho # m3/day

        for w in self.reservoir.wells:
            if w.name == "MOL-GT-02":  # injector - volumetric rate at constant injection temperature
                w.control = self.physics.new_rate_water_inj(vol_rate, temp)
                print(f"[ProductionModel] Well {w.name} set to injection rate {vol_rate:.2f} m3/day at T={temp:.2f} K")

            elif w.name == "MOL-GT-01":  # producer - volumetric rate 
                w.control = self.physics.new_rate_water_prod(vol_rate)
                print(f"[ProductionModel] Well {w.name} set to production rate {vol_rate:.2f} m3/day")

    @staticmethod
    def compute_mu_tangent(principal_stresses):
        """
        Compute tangent slope from origin to Mohr circle.
        principal_stresses: (n_elements, 3) array in bar, sorted [max, mid, min]
        Returns: mu_tangent array (n_elements,)
        """
        all_positive = np.all(principal_stresses > 0, axis=1)
        mu_tangent = np.ones(len(principal_stresses))
        if np.any(all_positive):
            sigma_max = principal_stresses[all_positive, 0]
            sigma_min = principal_stresses[all_positive, 2]
            mu_tangent[all_positive] = (sigma_max - sigma_min) / (2.0 * np.sqrt(sigma_max * sigma_min))
        return mu_tangent
    
    ## Analytical stress computation method
    def compute_analytical_stress_vect(self, dff, stress_df='', solution_df='',):
        # orientation of MIN PRINCIPAL STRESS (CLOCKWISE FROM NORTH)
        orientation_degrees = 80.0
        orientation_rad = orientation_degrees * np.pi / 180.

        dff['P'] = solution_df.loc[dff['ID'], 'P'].values * 1e5
        dff['T'] = solution_df.loc[dff['ID'], 'T'].values
        dff['dP'] = dff['P'] - dff['P0']
        dff['dT'] = dff['T'] - dff['T0']

        # Vectorized stress computation in cells's fault option 
        ## inputs transformed to matrix and vector for every cell
        SV, SH, Sh = func.stress_initialization(stress_df, dff)  # Size number of faults elements
        normal = func.normals(dff)  # could be out
        p_sigma = func.principal_stress_tensor(SV, SH, Sh)  # could be out Size number of faults elements
        p_faults = dff['P'].values  # pressure in fault blocks
        pressure = func.principal_stress_tensor(p_faults, p_faults, p_faults)  # we used the same function that before
        alpha, E, v = 1.e-5, 40.e9, 0.3  # thermal exp. [1/C], Young's M. [pas] and poisson's r [--].
        dS_T = func.dS_T(alpha, E, v, dff)  # Following dS_T = alpha*E/(1.-v)*dT
        eigenvec = func.eigenvec(orientation_rad, len(dff['ID']))
        eigenvec_t = np.transpose(eigenvec, axes=(0, 2, 1))

        # computing effective stress for all cells
        effective_p_sigma = p_sigma - pressure + dS_T

        effective_stress = np.einsum('ijk,ikl->ijl', effective_p_sigma, eigenvec)
        effective_stress = np.einsum('ijk,ikl->ijl', eigenvec_t, effective_stress)
        # computing traction vector on the fault cells
        Tv = np.einsum('ijk,ik->ij', effective_stress, normal)
        Sn_f = np.einsum('ij,ij->i', Tv, normal)
        Tv_mag = np.einsum('ij,ij->i', Tv, Tv)
        Tau_f = np.sqrt(Tv_mag - (np.einsum('ij,ij->i', Tv, normal)) ** 2)
        mu_vec = Tau_f / Sn_f  # value for reporting reactivation criteria

        # Extract diagonal terms of the effective stress tensor (principal stress components in bar)
        # These are already in principal coordinate system, sorted as [SV, SH, Sh]
        principal_stresses = np.diagonal(effective_p_sigma, axis1=1, axis2=2) / 1e5  # (n_elements, 3)
        
        # Compute mu_tangent
        mu_tangent = self.compute_mu_tangent(principal_stresses)
        
        return mu_vec, principal_stresses, mu_tangent