"""
Utility functions for DARTS model setup.

@author: KURGYIS
"""

import numpy as np

# from iapws.iapws97 import _Region1  #IAPWS97
# from iapws.iapws95 import IAPWS95


def order_segments(seg_conn, seg_ids):
    """
    Order line segments into a continuous polyline.

    Parameters
    ----------
    seg_conn : (N,2) int array
        Segment connectivity using global point indices.
    seg_ids : (N,) int array
        Segment IDs (used to break ties deterministically).

    Returns
    -------
    ordered_pt_idx : (M,) int array
        Ordered point indices forming the polyline.
    ordered_seg_ids : (M-1,) int array
        Segment IDs in traversal order.
    """
    seg_conn = np.asarray(seg_conn, dtype=np.int64)
    seg_ids = np.asarray(seg_ids)

    # build adjacency explicitly
    adj = {}
    for i, (a, b) in enumerate(seg_conn):
        if a not in adj:
            adj[a] = []
        if b not in adj:
            adj[b] = []
        adj[a].append((i, b))
        adj[b].append((i, a))

    degrees = {p: len(adj[p]) for p in adj}
    endpoints = [p for p, d in degrees.items() if d == 1]

    # choose start point
    if len(endpoints) >= 1:
        start = endpoints[0]
    else:
        imin = int(np.argmin(seg_ids))
        start = int(seg_conn[imin, 0])

    visited_seg = set()
    ordered_pts = [start]
    ordered_seg_ids = []

    cur = start

    while True:
        candidates = []
        for si, nxt in adj[cur]:
            if si not in visited_seg:
                candidates.append((si, nxt))

        if not candidates:
            break

        cand_seg_idx = [c[0] for c in candidates]
        best = int(np.argmin(seg_ids[cand_seg_idx]))
        seg_idx, nxt = candidates[best]

        visited_seg.add(seg_idx)
        ordered_seg_ids.append(int(seg_ids[seg_idx]))
        ordered_pts.append(int(nxt))
        cur = int(nxt)

    return (
        np.array(ordered_pts, dtype=np.int64),
        np.array(ordered_seg_ids, dtype=np.int32),
    )


def points_to_cell_ids_structured(points, x_faces, y_faces, z_faces):
    """
    Map xyz points to structured grid cell ids using face coordinates.

    points: (N,3) array
    x_faces: (nx+1,) cell boundary coordinates
    y_faces: (ny+1,)
    z_faces: (nz+1,)

    Returns: (N,) int array of cell_ids, -1 for points outside grid.
    Cell flattening: i + nx*(j + ny*k)
    """
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    nx = len(x_faces) - 1
    ny = len(y_faces) - 1
    nz = len(z_faces) - 1

    i = np.searchsorted(x_faces, x, side="right") - 1
    j = np.searchsorted(y_faces, y, side="right") - 1
    k = np.searchsorted(z_faces, z, side="right") - 1

    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)

    cell_ids = np.full(points.shape[0], -1, dtype=np.int64)
    cell_ids[inside] = i[inside] + nx * (j[inside] + ny * k[inside])
    return cell_ids


def points_to_ijk_structured(points, x_faces, y_faces, z_faces):
    """
    Map xyz points to structured grid indices (i, j, k).

    Returns:
        ijk: (N,3) int array
        inside: (N,) bool array
    """
    points = np.asarray(points, dtype=float)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    nx = len(x_faces) - 1
    ny = len(y_faces) - 1
    nz = len(z_faces) - 1

    i = np.searchsorted(x_faces, x, side="right") - 1
    j = np.searchsorted(y_faces, y, side="right") - 1
    k = np.searchsorted(z_faces, z, side="right") - 1

    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)

    ijk = np.column_stack([i, j, k]).astype(np.int32)
    return ijk, inside


def linear_to_ijk_1based(idx, nx, ny, nz):
    """
    Convert linear cell index (0..nx*ny*nz-1) to (i,j,k) 1-based,
    assuming i fastest, then j, then k.
    """
    idx = int(idx)
    i = (idx % nx) + 1
    j = ((idx // nx) % ny) + 1
    k = (idx // (nx * ny)) + 1
    return (i, j, k)


def points_to_ijk_by_nearest_centroid(points, centroids, nx, ny, nz):
    """
    Map points to nearest centroid cell and return (i,j,k) 1-based list.
    points: (N,3)
    centroids: (ncells,3) from reservoir.discretizer.centroids_all_cells
    """
    points = np.asarray(points, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    ijk = []
    for p in points:
        # nearest centroid (brute force; fine for a few thousand points)
        idx = int(np.argmin(np.sum((centroids - p) ** 2, axis=1)))
        ijk.append(linear_to_ijk_1based(idx, nx, ny, nz))
    return ijk


def segment_midpoints(points):
    """
    points: (N,3) ordered along the well
    Returns: (N-1,3) midpoints of consecutive segments
    """
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 2:
        return np.empty((0, 3), dtype=float)
    return 0.5 * (points[:-1] + points[1:])


def clip_segment_to_aabb(p0, p1, bmin, bmax, eps=1e-12):
    """
    Liang–Barsky style line–box clipping:
    Clip segment p(t)=p0+t*(p1-p0), t in [0,1] to axis-aligned box [bmin,bmax].
    Returns (t_enter, t_exit) if it intersects with non-empty length, else None.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0

    t0, t1 = 0.0, 1.0
    for axis in range(3):
        if abs(d[axis]) < eps:
            # Segment parallel to slab; must be within bounds to intersect
            if p0[axis] < bmin[axis] - eps or p0[axis] > bmax[axis] + eps:
                return None
        else:
            inv = 1.0 / d[axis]
            t_near = (bmin[axis] - p0[axis]) * inv
            t_far = (bmax[axis] - p0[axis]) * inv
            if t_near > t_far:
                t_near, t_far = t_far, t_near
            t0 = max(t0, t_near)
            t1 = min(t1, t_far)
            if t0 > t1:
                return None

    # If intersection is effectively a point (touch), treat as no-length intersection
    if t1 - t0 <= eps:
        return None

    return (t0, t1)


def get_perforation_cells_for_well(
    well_points, centroids, nx, ny, nz, tol=1e-6, allow_touch=False
):
    """
    Structured-grid version.
    - Computes segment entry/exit into domain bbox.
    - Uses midpoint of inside portion for nearest-centroid mapping.
    - Skips out-of-domain segments.

    allow_touch=False:
      if True, a segment that only touches the box may still create perforation (usually undesirable).
    """
    well_points = np.asarray(well_points, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    if well_points.shape[0] < 2:
        return []

    # Domain bbox - works for structured grid
    bmin = centroids.min(axis=0)
    bmax = centroids.max(axis=0)

    seen = set()
    perf = []

    for p0, p1 in zip(well_points[:-1], well_points[1:]):
        clip = clip_segment_to_aabb(p0, p1, bmin, bmax, eps=tol)
        if clip is None:
            continue

        t_enter, t_exit = clip

        if not allow_touch and (t_exit - t_enter) <= tol:
            continue

        # Pick a point guaranteed to lie inside the box
        t_mid = 0.5 * (t_enter + t_exit)
        pmid = p0 + t_mid * (p1 - p0)

        # Nearest centroid (now safe because pmid is inside the box)
        idx = int(np.argmin(np.sum((centroids - pmid) ** 2, axis=1)))
        cell = linear_to_ijk_1based(idx, nx, ny, nz)

        if cell not in seen:
            perf.append(cell)
            seen.add(cell)

    return perf


# TODO: move to helper_geomechanics.py


def stress_initialization(stress_df, dff):

    # get principal stresses for cells in the faults
    SV = np.interp(dff["z"].values, stress_df["D"].values, stress_df["SV"].values)
    Sh = np.interp(dff["z"].values, stress_df["D"].values, stress_df["S3"].values)
    # normal faulting or something else?
    SH = dff["Sh"] + 0.33 * (dff["SV"] - dff["Sh"])

    return SV, SH, Sh


def principal_stress_tensor(SV, SH, Sh):
    # builds an array which contains the principal stress tensor for each block of the faults
    # Initialize the matrix
    tensor = np.zeros(9 * len(SV))  # 9 p components

    # Compute the indices for each group
    indices_sv = np.arange(len(SV)) * 9
    indices_sH = indices_sv + 4
    indices_sh = indices_sv + 8

    # Assign values to the matrix
    tensor[indices_sv] = SV
    tensor[indices_sH] = SH
    tensor[indices_sh] = Sh

    tensor = tensor.reshape(len(SV), 3, 3)

    return tensor


def normals(dff):
    # create array that contain normal vector in the faults
    normal = np.column_stack((dff["nx"].values, dff["ny"].values, dff["nz"].values))

    return normal


def dS_T(alpha, E, v, dff):
    """
    builds thermal stress tensor (below) for every cell
    It contains only stress changes in horizontal stresses
    these changes are considered equal:
    [0  0     0
     0  dS_T  0
     0  0     dS_T]
    alpha: therma expansion coefficient [C^-1]
    E: young's modulus [Pas]
    v: poisson ratio
    """
    dS_T = alpha * E / (1.0 - v) * dff["dT"].values

    matrix = np.zeros(9 * len(dS_T))

    # Create indices for the positions of dT values
    base_indices = np.arange(len(dS_T)) * 9  # Starting points for each block
    offsets = np.array([4, 8])  # Relative offsets
    indices = base_indices.repeat(len(offsets)) + np.tile(offsets, len(dS_T))

    # Assign values using np.repeat
    matrix[indices] = np.repeat(dS_T, len(offsets))
    matrix = matrix.reshape(len(dS_T), 3, 3)

    return matrix


def eigenvec(orientation_rad, Faults_Id):
    # orientation_rad: minimum stress orientation
    matrix = np.zeros(Faults_Id * 9)
    matrix = matrix.reshape(Faults_Id, 3, 3)
    eigenvec = np.array(
        [
            [0, 0, 1],
            [np.sin(orientation_rad), np.cos(orientation_rad), 0],
            [
                np.sin(orientation_rad + np.pi / 2),
                np.cos(orientation_rad + np.pi / 2),
                0,
            ],
        ]
    )
    matrix = matrix + eigenvec

    return matrix


def stress_fault_df(faults, depth_reservoir, df_inc, stress_df=""):
    dff = faults  # reading faults
    # depth of position on fault
    dff["z"] = depth_reservoir[dff["ID"].values]
    # get principal stresses
    dff["SV"] = np.interp(
        dff["z"].values, stress_df["D"].values, stress_df["SV"].values
    )
    dff["Sh"] = np.interp(
        dff["z"].values, stress_df["D"].values, stress_df["S3"].values
    )
    # normal faulting or something else?
    dff["SH"] = dff["Sh"] + 0.33 * (dff["SV"] - dff["Sh"])
    # initial states
    dff["P0"] = (
        df_inc.loc[dff["ID"], "P"].values * 1e5
    )  # init pressure on faults - converts bar to Pa
    dff["T0"] = df_inc.loc[dff["ID"], "T"].values  # init temperature on faults - K

    # extra stresses holders
    dff["Sn"] = np.zeros(len(dff))  # Sn=normal on fault
    dff["Tau"] = np.zeros(len(dff))  # Tau=shear on fault
    dff["dS_T"] = np.zeros(len(dff))  # dS_T=thermal stress
    dff["Sp1"] = np.zeros(len(dff))  # Final MAX principal effective stress
    dff["Sp2"] = np.zeros(len(dff))  # Final INT principal effective stress
    dff["Sp3"] = np.zeros(len(dff))  # Final MIN principal effective stress
    dff["mu"] = np.zeros(len(dff))  # Final MIN principal effective stress

    return dff
