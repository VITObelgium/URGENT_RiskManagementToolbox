###############
## Production model setup
###############

# import RMT connector for DARTS
from connectors.open_darts import OpenDartsConnector

# import DARTS libraries
from darts.engines import redirect_darts_output
from darts.models.darts_model import DartsModel
from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer
from darts.reservoirs.struct_reservoir import StructReservoir

redirect_darts_output("run_urgent.log")

# import other libraries
import helpers.helper_modelling as func
import meshio
import numpy as np
import pandas as pd
from iapws import IAPWS95

# Create Production Model class, inheriting from DartsModel


class ProductionModel(DartsModel):
    def __init__(
        self,
        well_data,
        n_points=1000,
        mesh_file="",
        restart_file="",
        Qinj=0.0,
        Tinj=66 + 273.15,
        Qprod=None,
    ):

        super().__init__()

        # ---- enforce required inputs ----

        if not mesh_file:
            print("[ERROR] For production scenario, mesh_file (.vtu) is required.")
            raise ValueError("For production scenario, mesh_file (.vtu) is required.")

        if not restart_file:
            print("[ERROR] For production scenario, restart_file (.csv) is required.")
            raise ValueError(
                "For production scenario, restart_file (.csv) is required."
            )

        self._well_data = well_data
        self.mesh_file = mesh_file
        self.restart_file = restart_file

        if Qprod is None:
            Qprod = Qinj  # Default set to balanced operation
        self.Qinj = Qinj
        self.Tinj = Tinj
        self.Qprod = Qprod
        self.timer.node["initialization"].start()

        # reservoir grid and properties
        self.set_reservoir()

        # definition of the physics (geothermal)
        self.set_physics(n_points=n_points)

        # initial conditions
        df_inc = pd.read_csv(self.restart_file, sep=",")
        self.df_inc = df_inc

        self.initial_values = {
            self.physics.vars[0]: self.df_inc["P"].values,
            self.physics.vars[1]: self.df_inc["H"].values,
        }

        print("Done with model initializing.")

        # timestep parameters
        self.params.first_ts = 1e-3
        self.params.mult_ts = 2
        self.params.max_ts = 365  # days

        # nonlinear and linear solver tolerance
        self.params.tolerance_newton = 1e-2
        self.params.tolerance_linear = 1e-4

        self.timer.node["initialization"].stop()

    ## Structured reservoir setup
    def set_reservoir(self):
        mesh = meshio.read(self.mesh_file)
        blk = 0  # hexahedron cell block index
        pts = mesh.points

        # Structured axes from input mesh
        x_edges = np.unique(pts[:, 0])
        x_edges.sort()
        y_edges = np.unique(pts[:, 1])
        y_edges.sort()
        z_edges = np.unique(pts[:, 2])
        z_edges.sort()

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
        hex_conn = mesh.cells[blk].data
        cell_pts = pts[hex_conn]
        centroids = cell_pts.mean(axis=1)
        self.centroids = centroids

        depth_raw = np.asarray(centroids[:, 2], dtype=float)
        self.ztop = float(
            depth_raw.min()
        )  # define "top" as shallowest centroid (smallest z) - DARTS follows right-hand coordinate system with z positive downwards!

        # Check if mesh is pure structured cartesian grid
        n_cells = centroids.shape[0]
        if n_cells != nx * ny * nz:
            raise ValueError(
                f"VTU has {n_cells} hex cells but nx*ny*nz = {nx * ny * nz}. "
                "Mesh is not a pure structured cartesian grid."
            )

        # read properties from mesh cell data
        # op_num = np.asarray(mesh.cell_data["num"][blk], dtype=float)
        permx_raw = np.asarray(mesh.cell_data["permx"][blk], dtype=float)
        permy_raw = np.asarray(mesh.cell_data["permy"][blk], dtype=float)
        permz_raw = np.asarray(mesh.cell_data["permz"][blk], dtype=float)
        poro_raw = np.asarray(mesh.cell_data["pore"][blk], dtype=float)
        hcap_raw = np.asarray(mesh.cell_data["hcap"][blk], dtype=float)
        rcond_raw = (
            np.asarray(mesh.cell_data["cond"][blk], dtype=float) * 24 * 3600 / 1000
        )  # Unit conversion for cond * 24*3600/1000 [[kJ/(m-d-K)]]
        rocknum_raw = np.asarray(mesh.cell_data["num"][blk], dtype=int)

        # property mapping from vtu to DARTS structure grid
        i = np.searchsorted(x_cent, self.centroids[:, 0])
        i = np.clip(i, 0, nx - 1)
        j = np.searchsorted(y_cent, self.centroids[:, 1])
        j = np.clip(j, 0, ny - 1)
        k = np.searchsorted(z_cent, self.centroids[:, 2])
        k = np.clip(k, 0, nz - 1)
        flat = i + nx * (j + ny * k)

        depth = np.empty(nx * ny * nz, dtype=float)
        permx = np.empty(nx * ny * nz, dtype=float)
        permy = np.empty(nx * ny * nz, dtype=float)
        permz = np.empty(nx * ny * nz, dtype=float)
        poro = np.empty(nx * ny * nz, dtype=float)
        hcap = np.empty(nx * ny * nz, dtype=float)
        rcond = np.empty(nx * ny * nz, dtype=float)
        rocknum = np.empty(nx * ny * nz, dtype=int)

        depth[flat] = depth_raw
        permx[flat] = permx_raw
        permy[flat] = permy_raw
        permz[flat] = permz_raw
        poro[flat] = poro_raw
        hcap[flat] = hcap_raw
        rcond[flat] = rcond_raw
        rocknum[flat] = rocknum_raw

        self.reservoir = StructReservoir(
            self.timer,
            nx,
            ny,
            nz,
            dx_full,
            dy_full,
            dz_full,
            permx=permx,
            permy=permy,
            permz=permz,
            poro=poro,
            depth=depth,
            rcond=rcond,
            hcap=hcap,
            cache=False,
        )
        self.reservoir.global_data["depth"] = depth
        self.reservoir.global_data["start_z"] = self.ztop
        self.reservoir.global_data["rocknum"] = rocknum

        # init_reservoir does self.discretize() as well but also stores the mesh as self.reservoir.mesh
        self.reservoir.init_reservoir(verbose=True)

        # set boundary volumes to large value to simulate no-flow boundaries on top and bottom sides
        # self.reservoir.boundary_volumes["xy_minus"] = 1e25
        # self.reservoir.boundary_volumes["xy_plus"] = 1e25
        self.reservoir.boundary_volumes["yz_minus"] = 1e25  # open flow boundary at x=0
        self.reservoir.boundary_volumes["yz_plus"] = (
            1e25  # open flow boundary at x=xmax
        )
        return

    ## Setup physics (geothermal)
    def set_physics(self, n_points, verbose: bool = False):
        # create pre-defined physics for geothermal
        self.property_container = PropertyContainer()
        self.property_container.output_props = {
            "temperature": lambda: self.property_container.temperature
        }
        self.physics = Geothermal(
            self.timer,
            n_points,
            min_p=1,
            max_p=500,
            min_e=1000,
            max_e=20000,
            cache=False,
        )
        self.physics.add_property_region(self.property_container)

        return

    ## Setup wells and perforations
    def set_wells(self):

        wells = OpenDartsConnector.get_well_connection_cells(
            self._well_data, self.reservoir
        )

        for well_name, cells in wells.items():
            self.reservoir.add_well(well_name)
            for cell in cells:
                i, j, k = cell
                self.reservoir.add_perforation(
                    well_name, cell_index=(i, j, k), multi_segment=False, verbose=True
                )

    ## Setup well injection and production controls
    def set_well_controls(self):
        injection_rate = self.Qinj  # ton/day
        production_rate = self.Qprod  # ton/day
        p_avg = 321.5  # bar  - reference pressure (average reservoir pressure - initial value)
        T_avg = 417  # K    - reference temperature (average reservoir temperature - initial value)
        T_inj = self.Tinj  # K    - injection temperature
        # Mass to volumetric rate conversion
        rho_inj = IAPWS95(P=p_avg / 10.0, T=T_inj).rho  # density in kg/m3
        vol_rate_inj = injection_rate * 1000.0 / rho_inj  # m3/day
        rho_prod = IAPWS95(P=p_avg / 10.0, T=T_avg).rho  # density in kg/m3
        vol_rate_prod = production_rate * 1000.0 / rho_prod  # m3/day

        for w in self.reservoir.wells:
            if (
                w.name == "INJ"
            ):  # injector - volumetric rate at constant injection temperature
                w.control = self.physics.new_rate_water_inj(vol_rate_inj, T_inj)
                print(
                    f"[ProductionModel] Well {w.name} set to injection rate {vol_rate_inj:.2f} m3/day at T={T_inj:.2f} K"
                )

            elif w.name == "PROD":  # producer - volumetric rate
                w.control = self.physics.new_rate_water_prod(vol_rate_prod)
                print(
                    f"[ProductionModel] Well {w.name} set to production rate {vol_rate_prod:.2f} m3/day"
                )

    ## Analytical stress computation method
    def compute_analytical_stress_vect(
        self,
        dff,
        stress_df="",
        solution_df="",
    ):
        # orientation of MIN PRINCIPAL STRESS (CLOCKWISE FROM NORTH)
        orientation_degrees = 120.0
        orientation_rad = orientation_degrees * np.pi / 180.0

        dff["P"] = solution_df.loc[dff["ID"], "P"].values * 1e5
        dff["T"] = solution_df.loc[dff["ID"], "T"].values
        dff["dP"] = dff["P"] - dff["P0"]
        dff["dT"] = dff["T"] - dff["T0"]

        # Vectorized stress computation in cells's fault option
        ## inputs transformed to matrix and vector for every cell
        SV, SH, Sh = func.stress_initialization(
            stress_df, dff
        )  # Size number of faults elements
        normal = func.normals(dff)  # could be out
        p_sigma = func.principal_stress_tensor(
            SV, SH, Sh
        )  # could be out Size number of faults elements
        p_faults = dff["P"].values  # pressure in fault blocks
        pressure = func.principal_stress_tensor(
            p_faults, p_faults, p_faults
        )  # we used the same function that before
        alpha, E, v = (
            1.0e-5,
            40.0e9,
            0.3,
        )  # thermal exp. [1/C], Young's M. [pas] and poisson's r [--].
        dS_T = func.dS_T(alpha, E, v, dff)  # Following dS_T = alpha*E/(1.-v)*dT
        eigenvec = func.eigenvec(orientation_rad, len(dff["ID"]))
        eigenvec_t = np.transpose(eigenvec, axes=(0, 2, 1))

        # computing effective stress for all cells
        effective_p_sigma = p_sigma - pressure + dS_T
        effective_stress = np.einsum("ijk,ikl->ijl", effective_p_sigma, eigenvec)
        effective_stress = np.einsum("ijk,ikl->ijl", eigenvec_t, effective_stress)

        # computing traction vector on the fault cells
        Tv = np.einsum("ijk,ik->ij", effective_stress, normal)
        Sn_f = np.einsum("ij,ij->i", Tv, normal)
        Tv_mag = np.einsum("ij,ij->i", Tv, Tv)
        Tau_f = np.sqrt(Tv_mag - (np.einsum("ij,ij->i", Tv, normal)) ** 2)
        mu_vec = Tau_f / Sn_f  # value for reporting reactivation criteria

        return mu_vec
