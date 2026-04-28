# fem_geomechanics.py
from __future__ import annotations

import gc

import numpy as np
from fem_helpers.boundaries import BC_dict, SetDirichletBCs, get_boundary_nodelists
from fem_helpers.element_hex8 import (
    GetElementForces,
    GetElementStiffness,
)
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg

# global variables 
_I6 = np.array([1, 1, 1, 0, 0, 0], dtype=float)

_S = np.array([
    [-1, -1, -1],  # n0
    [+1, -1, -1],  # n1
    [+1, +1, -1],  # n2
    [-1, +1, -1],  # n3
    [-1, -1, +1],  # n4
    [+1, -1, +1],  # n5
    [+1, +1, +1],  # n6
    [-1, +1, +1],  # n7
], dtype=float)

def B_hex8_center(hx: float, hy: float, hz: float) -> np.ndarray:
    dNdx = _S[:, 0] / (4.0 * hx)
    dNdy = _S[:, 1] / (4.0 * hy)
    dNdz = _S[:, 2] / (4.0 * hz)

    B = np.zeros((6, 24), dtype=float)
    for a in range(8):
        ia = 3 * a
        B[0, ia + 0] = dNdx[a]
        B[1, ia + 1] = dNdy[a]
        B[2, ia + 2] = dNdz[a]
        B[3, ia + 0] = dNdy[a]
        B[3, ia + 1] = dNdx[a]
        B[4, ia + 1] = dNdz[a]
        B[4, ia + 2] = dNdy[a]
        B[5, ia + 0] = dNdz[a]
        B[5, ia + 2] = dNdx[a]
    return B

class FEMGeomechanics:
    def __init__(
        self,
        *,
        pts: np.ndarray,
        hex_conn: np.ndarray,
        fault_df,
        P0: np.ndarray,
        T0: np.ndarray,
        E: np.ndarray,
        nu: np.ndarray,
        alpha_b: np.ndarray,
        alpha_T: np.ndarray,
        rho: np.ndarray,
        bc_params: dict | None = None,
        solver_params: dict | None = None,
    ):
        self.ndof = 3  # dofs per node (ux, uy, uz)
        self.pts = np.asarray(pts, dtype=float)
        self.hex_conn = np.asarray(hex_conn, dtype=int)

        self.fault_df = fault_df  # keep as dataframe; we’ll pull arrays in setup()

        self.P0 = np.asarray(P0, dtype=float)
        self.T0 = np.asarray(T0, dtype=float)

        self.E = np.asarray(E, dtype=float)
        self.nu = np.asarray(nu, dtype=float)
        self.alpha_b = np.asarray(alpha_b, dtype=float)
        self.alpha_T = np.asarray(alpha_T, dtype=float)
        self.rho = np.asarray(rho, dtype=float)

        self.bc_params = bc_params or {}
        self.solver_params = solver_params or {}

        # Will be filled in setup()
        self.sizes = None            # (numele,3) hx,hy,hz
        self.K = None                # global stiffness (sparse)
        self.Fp = None               # force operator for dP
        self.Ft = None               # force operator for dT
        self.free_dofs = None
        self.fixed_dofs = None
        self.A_ff = None             # reduced stiffness
        self.M = None                # preconditioner (e.g. LinearOperator)

        self.fault_e = None
        self.fault_normals = None
        self.fault_edofs = None
        self.fault_B = None
        self.fault_D = None
        self.fault_alpha_T = None

        # Solve cache/settings
        self._u_free_prev = None
        self.cg_rtol = float(self.solver_params.get("rtol", 1e-7))
        self.cg_maxiter = int(self.solver_params.get("maxiter", 1500))
        self.cg_warm_start = bool(self.solver_params.get("warm_start", True))

        # Orientation-dependent initial stress cache
        self._sigma_init_cache = {}

    def setup(self):
        """One-time setup: geometry caches, FE operators, BC reduction, fault caches."""
        self.numelem = self.hex_conn.shape[0]
        nnodes = self.pts.shape[0]
        self.numeqns = nnodes*self.ndof

        self.hex_conn_fe = reorder_hex8_cartesian(self.pts, self.hex_conn)  # ensure canonical node ordering for FE shape functions
        # element sizes (hx,hy,hz) for all elements, using reordered connectivity
        cell_pts = self.pts[self.hex_conn_fe]          # (numelem, 8, 3)
        mins = cell_pts.min(axis=1)                    # (numelem, 3)
        maxs = cell_pts.max(axis=1)                    # (numelem, 3)
        self.sizes = maxs - mins                       # (numelem, 3) -> hx,hy,hz

        # basic validation
        if self.P0.shape != (self.numelem,):
            raise ValueError(f"P0 must be shape ({self.numelem},), got {self.P0.shape}")
        if self.T0.shape != (self.numelem,):
            raise ValueError(f"T0 must be shape ({self.numelem},), got {self.T0.shape}")

        for name, arr in [("E", self.E), ("nu", self.nu), ("alpha_b", self.alpha_b), ("alpha_T", self.alpha_T)]:
            if arr.shape != (self.numelem,):
                raise ValueError(f"{name} must be shape ({self.numelem},), got {arr.shape}")

        # cache fault element ids and normals (fault_df has ID,nx,ny,nz)
        self.fault_e = self.fault_df["ID"].to_numpy(dtype=int)
        self.fault_normals = self.fault_df[["nx", "ny", "nz"]].to_numpy(dtype=float)
        
        # Extract initial stresses if available in fault_df
        if 'SV' in self.fault_df.columns:
            self.fault_SV = self.fault_df["SV"].to_numpy(dtype=float)
            self.fault_SH = self.fault_df["SH"].to_numpy(dtype=float)
            self.fault_Sh = self.fault_df["Sh"].to_numpy(dtype=float)
        else:
            # If not available, set to None
            self.fault_SV = None
            self.fault_SH = None
            self.fault_Sh = None

        # fault element dof indices (nfault,24)
        dof_offsets = np.array([0, 1, 2], dtype=np.int32)
        fault_nodes = self.hex_conn_fe[self.fault_e].astype(np.int32, copy=False)   # (nfault, 8)

        self.fault_edofs = (3 * fault_nodes[..., None] + dof_offsets).reshape(len(self.fault_e), 24)

        # Precompute B matrices and elasticity matrices for fault elements only.
        n_faults = len(self.fault_e)
        if n_faults > 0:
            hx = self.sizes[self.fault_e, 0]
            hy = self.sizes[self.fault_e, 1]
            hz = self.sizes[self.fault_e, 2]

            dNdx = _S[:, 0][None, :] / (4.0 * hx[:, None])
            dNdy = _S[:, 1][None, :] / (4.0 * hy[:, None])
            dNdz = _S[:, 2][None, :] / (4.0 * hz[:, None])

            self.fault_B = np.zeros((n_faults, 6, 24), dtype=float)
            for a in range(8):
                ia = 3 * a
                self.fault_B[:, 0, ia + 0] = dNdx[:, a]
                self.fault_B[:, 1, ia + 1] = dNdy[:, a]
                self.fault_B[:, 2, ia + 2] = dNdz[:, a]
                self.fault_B[:, 3, ia + 0] = dNdy[:, a]
                self.fault_B[:, 3, ia + 1] = dNdx[:, a]
                self.fault_B[:, 4, ia + 1] = dNdz[:, a]
                self.fault_B[:, 4, ia + 2] = dNdy[:, a]
                self.fault_B[:, 5, ia + 0] = dNdz[:, a]
                self.fault_B[:, 5, ia + 2] = dNdx[:, a]

            Ef = self.E[self.fault_e]
            nuf = self.nu[self.fault_e]
            lam = Ef * nuf / ((1.0 + nuf) * (1.0 - 2.0 * nuf))
            mu = Ef / (2.0 * (1.0 + nuf))
            self.fault_D = np.zeros((n_faults, 6, 6), dtype=float)
            self.fault_D[:, 0, 0] = lam + 2.0 * mu
            self.fault_D[:, 1, 1] = lam + 2.0 * mu
            self.fault_D[:, 2, 2] = lam + 2.0 * mu
            self.fault_D[:, 0, 1] = lam
            self.fault_D[:, 0, 2] = lam
            self.fault_D[:, 1, 0] = lam
            self.fault_D[:, 1, 2] = lam
            self.fault_D[:, 2, 0] = lam
            self.fault_D[:, 2, 1] = lam
            self.fault_D[:, 3, 3] = mu
            self.fault_D[:, 4, 4] = mu
            self.fault_D[:, 5, 5] = mu

            self.fault_alpha_T = self.alpha_T[self.fault_e]

        # BC setup: identify boundary nodes and construct dof lists for Dirichlet BCs
        self.boundary_nodes = get_boundary_nodelists(self.pts)
        
        BC_u_bot = BC_dict(bc_type='dirchlet',variable=2,
                nodelist=self.boundary_nodes['z_max'],
                values=[0])
        BC_u_left = BC_dict(bc_type='dirchlet',variable=0,
                nodelist=self.boundary_nodes['x_min'],
                values=[0])
        BC_u_back = BC_dict(bc_type='dirchlet',variable=1,
                nodelist=self.boundary_nodes['y_min'],
                values=[0])  
        BC_list = [BC_u_bot, BC_u_left, BC_u_back]
        dofids, dofvals = SetDirichletBCs(BC_list,time_step_number=0,ndof=self.ndof)

        # Construct global stiffness matrix K, force operators Fp and Ft
        cell_nnz_K = 576          # 24*24
        cell_nnz_F = 24           # 24
        K_nnz = self.numelem * cell_nnz_K
        F_nnz = self.numelem * cell_nnz_F

        k_rows = np.empty(K_nnz, dtype=np.int32)
        k_cols = np.empty(K_nnz, dtype=np.int32)
        k_data = np.empty(K_nnz, dtype=np.float64)

        fp_rows = np.empty(F_nnz, dtype=np.int32)
        fp_cols = np.empty(F_nnz, dtype=np.int32)
        fp_data = np.empty(F_nnz, dtype=np.float64)

        ft_rows = np.empty(F_nnz, dtype=np.int32)
        ft_cols = np.empty(F_nnz, dtype=np.int32)
        ft_data = np.empty(F_nnz, dtype=np.float64)

        k_ptr = 0
        f_ptr = 0

        dof_offsets = np.arange(self.ndof, dtype=np.int32)  # [0,1,2]

        for e in range(self.numelem):
            nodes = self.hex_conn_fe[e].astype(np.int32, copy=False)

            Xe = self.pts[nodes, 0]
            Ye = self.pts[nodes, 1]
            Ze = self.pts[nodes, 2]

            ke = GetElementStiffness(Xe, Ye, Ze, self.E[e], self.nu[e], self.ndof)
            fpe, fte, _ = GetElementForces(Xe, Ye, Ze,1.0, 1.0,self.E[e], self.nu[e],self.alpha_b[e], self.alpha_T[e],self.rho[e], self.ndof)

            # element dofs (24,)
            dofs = (self.ndof * nodes[:, None] + dof_offsets[None, :]).reshape(24)

            # stiffness triplets (576)
            ii, jj = np.meshgrid(dofs, dofs, indexing="ij")
            k_rows[k_ptr:k_ptr + cell_nnz_K] = ii.ravel()
            k_cols[k_ptr:k_ptr + cell_nnz_K] = jj.ravel()
            k_data[k_ptr:k_ptr + cell_nnz_K] = ke.ravel()
            k_ptr += cell_nnz_K

            # force operator triplets (24) -> column is element id e
            fp_rows[f_ptr:f_ptr + cell_nnz_F] = dofs
            fp_cols[f_ptr:f_ptr + cell_nnz_F] = e
            fp_data[f_ptr:f_ptr + cell_nnz_F] = fpe

            ft_rows[f_ptr:f_ptr + cell_nnz_F] = dofs
            ft_cols[f_ptr:f_ptr + cell_nnz_F] = e
            ft_data[f_ptr:f_ptr + cell_nnz_F] = fte

            f_ptr += cell_nnz_F

        A = coo_matrix((k_data, (k_rows, k_cols)), shape=(self.numeqns, self.numeqns)).tocsr()
        Fp = coo_matrix((fp_data, (fp_rows, fp_cols)), shape=(self.numeqns, self.numelem)).tocsr()
        Ft = coo_matrix((ft_data, (ft_rows, ft_cols)), shape=(self.numeqns, self.numelem)).tocsr()

        # free big assembly arrays immediately
        del k_rows, k_cols, k_data, fp_rows, fp_cols, fp_data, ft_rows, ft_cols, ft_data
        gc.collect()
        print('Done with element loop')
        # reduces system size based on boundary conditions
        all_dofs   = np.arange(self.numeqns)
        self.fixed_dofs = np.array(dofids, dtype=int)
        self.free_dofs  = np.setdiff1d(all_dofs, self.fixed_dofs)
        # reduced dof stiffness matrix and force operators
        self.A_ff = A[self.free_dofs][:, self.free_dofs].tocsr()
        self.Fp_free = Fp[self.free_dofs, :]
        self.Ft_free = Ft[self.free_dofs, :]
        del Fp, Ft
        diag = self.A_ff.diagonal()
        M_inv_diag = 1.0/diag
        def M_inv(x):
            return M_inv_diag*x
        self.M = LinearOperator(self.A_ff.shape,matvec=M_inv)
        del A
        gc.collect()
        
        return self

    def _get_sigma_init_global(self, orientation_degrees: float) -> np.ndarray:
        """Return cached initial effective stress tensors on faults in global coordinates."""
        key = float(orientation_degrees)
        cached = self._sigma_init_cache.get(key)
        if cached is not None:
            return cached

        n_faults = len(self.fault_e)
        sigma_init_global = np.zeros((n_faults, 3, 3), dtype=float)
        if self.fault_SV is None or n_faults == 0:
            self._sigma_init_cache[key] = sigma_init_global
            return sigma_init_global

        orientation_rad = np.deg2rad(orientation_degrees)
        sin_or = np.sin(orientation_rad)
        cos_or = np.cos(orientation_rad)
        sin_or_90 = np.sin(orientation_rad + np.pi / 2.0)
        cos_or_90 = np.cos(orientation_rad + np.pi / 2.0)
        R = np.array(
            [[0.0, sin_or, sin_or_90], [0.0, cos_or, cos_or_90], [1.0, 0.0, 0.0]],
            dtype=float,
        )

        P0_fault_pa = self.P0[self.fault_e] * 1e5
        sigma_principal = np.column_stack(
            (
                self.fault_SV - P0_fault_pa,
                self.fault_SH - P0_fault_pa,
                self.fault_Sh - P0_fault_pa,
            )
        )
        for i in range(n_faults):
            sigma_init_global[i] = (R * sigma_principal[i]) @ R.T

        self._sigma_init_cache[key] = sigma_init_global
        return sigma_init_global

    def solve_u(self, P: np.ndarray, T: np.ndarray):
        dP = (P[:self.numelem]-self.P0)*1e5
        dT = (T[:self.numelem]-self.T0)

        rhs_free = self.Fp_free @ dP + self.Ft_free @ dT

        # Solve A_ff * u_free = rhs_free using preconditioned CG
        x0 = self._u_free_prev if self.cg_warm_start else None
        u_free, info = cg(
            self.A_ff,
            rhs_free,
            M=self.M,
            x0=x0,
            maxiter=self.cg_maxiter,
            rtol=self.cg_rtol,
        )
        if info != 0:
            # info>0 => no convergence in maxiter, info<0 => breakdown
            print(f"[FEM] CG no convergence: info={info} (rtol={self.cg_rtol}, maxiter={self.cg_maxiter})")
        self._u_free_prev = u_free
        u = np.zeros(self.numeqns)
        u[self.free_dofs] = u_free
        return u

    def solve(self, P: np.ndarray, T: np.ndarray):
        """Compatibility wrapper for callers expecting fem_geo.solve(...)."""
        return self.solve_u(P, T)
    
    def compute_mu(self, P: np.ndarray, T: np.ndarray, u: np.ndarray | None = None, orientation_degrees: float = 80.0) -> tuple:
        """
        Compute mobilized friction coefficient mu on fault elements using EFFECTIVE stress.

        Optimized version: computes stress tensor once per element.
        
        Returns: (mu_vec, principal_stresses, mu_tangent)
            - mu_vec: friction coefficient for each fault element
            - principal_stresses: (n_faults, 3) array [max, mid, min] in bar
            - mu_tangent: tangent slope from origin to Mohr circle
        """
        # ---- solve if u not provided ----
        if u is None:
            u = self.solve_u(P, T)

        n_faults = len(self.fault_e)
        if n_faults == 0:
            return np.empty(0, dtype=float), np.empty((0, 3), dtype=float), np.empty(0, dtype=float)

        dT_fault = (T[:self.numelem] - self.T0)[self.fault_e]
        ue_fault = u[self.fault_edofs]

        eps = np.einsum("fij,fj->fi", self.fault_B, ue_fault)
        eps_th = self.fault_alpha_T[:, None] * dT_fault[:, None] * _I6[None, :]
        sig_voigt = -np.einsum("fij,fj->fi", self.fault_D, eps - eps_th)

        sigma_init_global = self._get_sigma_init_global(orientation_degrees)
        Sigma_eff = np.empty((n_faults, 3, 3), dtype=float)
        Sigma_eff[:, 0, 0] = sig_voigt[:, 0]
        Sigma_eff[:, 1, 1] = sig_voigt[:, 1]
        Sigma_eff[:, 2, 2] = sig_voigt[:, 2]
        Sigma_eff[:, 0, 1] = sig_voigt[:, 5]
        Sigma_eff[:, 1, 0] = sig_voigt[:, 5]
        Sigma_eff[:, 0, 2] = sig_voigt[:, 4]
        Sigma_eff[:, 2, 0] = sig_voigt[:, 4]
        Sigma_eff[:, 1, 2] = sig_voigt[:, 3]
        Sigma_eff[:, 2, 1] = sig_voigt[:, 3]
        Sigma_eff += sigma_init_global

        n = self.fault_normals
        t = np.einsum("fij,fj->fi", Sigma_eff, n)
        Sn_eff = np.einsum("fi,fi->f", n, t)
        tau_vec = t - Sn_eff[:, None] * n
        Tau = np.linalg.norm(tau_vec, axis=1)
        mu_vec = Tau / np.maximum(1e-12, np.abs(Sn_eff))

        eigvals = np.linalg.eigvalsh(Sigma_eff)
        principal_stresses = eigvals[:, ::-1] / 1e5  # [max, mid, min] in bar
        
        # Compute mu_tangent from principal stresses (vectorized)
        all_positive = np.all(principal_stresses > 0, axis=1)
        mu_tangent = np.ones(n_faults)
        if np.any(all_positive):
            sigma_max = principal_stresses[all_positive, 0]
            sigma_min = principal_stresses[all_positive, 2]
            mu_tangent[all_positive] = (sigma_max - sigma_min) / (2.0 * np.sqrt(sigma_max * sigma_min))

        return mu_vec, principal_stresses, mu_tangent

def reorder_hex8_cartesian(pts: np.ndarray, hex_conn: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    Reorder each hex element's 8 nodes into canonical FE ordering:
      0:(xmin,ymin,zmin), 1:(xmax,ymin,zmin), 2:(xmax,ymax,zmin), 3:(xmin,ymax,zmin),
      4:(xmin,ymin,zmax), 5:(xmax,ymin,zmax), 6:(xmax,ymax,zmax), 7:(xmin,ymax,zmax)

    Assumes axis-aligned bricks.
    """
    conn = np.asarray(hex_conn, dtype=int)
    cell_pts = pts[conn]  # (numele,8,3)

    mins = cell_pts.min(axis=1)  # (numele,3)
    maxs = cell_pts.max(axis=1)

    # For each corner, find the node matching (x?, y?, z?) combination
    # using closeness to min/max in each axis.
    def pick_corner(mask):
        # mask shape: (numele,8) with True where node matches corner
        # Take the first True in each row
        idx = np.argmax(mask, axis=1)
        # Optional sanity: ensure each row has exactly one True
        ok = mask.sum(axis=1) == 1
        if not np.all(ok):
            bad = np.where(~ok)[0][:10]
            raise ValueError(f"Corner match not unique for elements {bad.tolist()} (check tol or non-cartesian cells).")
        return conn[np.arange(conn.shape[0]), idx]

    x = cell_pts[..., 0]; y = cell_pts[..., 1]; z = cell_pts[..., 2]
    xmin = mins[:, 0][:, None]; xmax = maxs[:, 0][:, None]
    ymin = mins[:, 1][:, None]; ymax = maxs[:, 1][:, None]
    zmin = mins[:, 2][:, None]; zmax = maxs[:, 2][:, None]

    is_xmin = np.isclose(x, xmin, atol=tol)
    is_xmax = np.isclose(x, xmax, atol=tol)
    is_ymin = np.isclose(y, ymin, atol=tol)
    is_ymax = np.isclose(y, ymax, atol=tol)
    is_zmin = np.isclose(z, zmin, atol=tol)
    is_zmax = np.isclose(z, zmax, atol=tol)

    n0 = pick_corner(is_xmin & is_ymin & is_zmin)
    n1 = pick_corner(is_xmax & is_ymin & is_zmin)
    n2 = pick_corner(is_xmax & is_ymax & is_zmin)
    n3 = pick_corner(is_xmin & is_ymax & is_zmin)
    n4 = pick_corner(is_xmin & is_ymin & is_zmax)
    n5 = pick_corner(is_xmax & is_ymin & is_zmax)
    n6 = pick_corner(is_xmax & is_ymax & is_zmax)
    n7 = pick_corner(is_xmin & is_ymax & is_zmax)

    return np.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=1)