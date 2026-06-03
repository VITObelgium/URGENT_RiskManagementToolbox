"""
Helper functions for fault-reactivation objective calculations.
"""

import numpy as np


def fault_exceedance_vector(mu_vec, mu_crit):

    mu_vec = np.asarray(mu_vec, dtype=float)

    if mu_crit <= 0:
        raise ValueError("mu_crit must be positive.")

    # Compute per-fault-cell exceedance above critical friction ratio.
    r_vec = mu_vec / mu_crit
    exceedance_vec = np.maximum(0.0, r_vec - 1.0)

    return exceedance_vec

def update_fault_exceedance_cells(fault_exceedance_cells, mu_vec, mu_crit, dt_days):
    """
    Update cumulative time-integrated exceedance for every fault cell.
           fault_exceedance_cells_i += max(0, mu_i / mu_crit - 1) * dt_days
    """

    if dt_days < 0:
        raise ValueError("dt_days must be non-negative.")

    fault_exceedance_cells = np.asarray(fault_exceedance_cells, dtype=float)
    exceedance_vec = fault_exceedance_vector(mu_vec, mu_crit)

    if fault_exceedance_cells.shape != exceedance_vec.shape:
        raise ValueError(
            "fault_exceedance_cells and mu_vec must have the same shape. "
            f"Got {fault_exceedance_cells.shape} and {exceedance_vec.shape}."
        )

    # Update cumulative exceedance for each cell
    timestep_cells = exceedance_vec * dt_days
    fault_exceedance_cells += timestep_cells

    # sum contribution for this timestep
    timestep_contribution = float(np.sum(timestep_cells))

    return fault_exceedance_cells, timestep_contribution


def total_fault_exceedance(fault_exceedance_cells):

    # Compute sum over all fault cells to get total exceedance indicator
    fault_exceedance_cells = np.asarray(fault_exceedance_cells, dtype=float)
    FaultExceedance = float(np.sum(fault_exceedance_cells))
    print(f"FaultExceedance[cell_days] = {FaultExceedance:.6e}")
    
    return FaultExceedance