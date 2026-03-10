import numpy as np

from common import OptimizationStrategy
from services.solution_updater_service.core.engines.pso import PSOEngine


def test_pso_repairs_infeasible_population():
    """PSOEngine should repair an entire population that starts outside bounds and linear inequalities.

    The test places all particles outside the provided lb/ub and violating the linear
    constraint x0 + x1 <= 4. After a single call to `update_solution_to_next_iter`
    the returned positions must satisfy the box bounds and the linear inequality
    (within a small numerical tolerance).
    """

    engine = PSOEngine()

    # Three particles in 2D, all outside bounds and violating A x <= b
    parameters = np.array([[5.0, 5.0], [6.0, 7.0], [10.0, 10.0]], dtype=float)

    # Dummy results
    results = np.array([[1.0], [2.0], [3.0]], dtype=float)

    # Strict box bounds that the initial population violates
    lb = np.array([0.0, 0.0], dtype=float)
    ub = np.array([4.0, 4.0], dtype=float)

    # Simple linear inequality: x0 + x1 <= 4
    A = np.array([[1.0, 1.0]], dtype=float)
    b = np.array([4.0], dtype=float)

    new_positions = engine.update_solution_to_next_iter(
        parameters, results, lb, ub, {0: OptimizationStrategy.MINIMIZE}, A, b
    )

    # Basic sanity
    assert new_positions.shape == parameters.shape

    # All positions must be within box bounds
    assert np.all(new_positions >= lb - 1e-8), f"Positions below lb: {new_positions}"
    assert np.all(new_positions <= ub + 1e-8), f"Positions above ub: {new_positions}"

    # And they should satisfy the linear inequality A x <= b (with small tolerance)
    Ax = (A @ new_positions.T).T
    assert np.all(Ax <= b + 1e-6), (
        f"Linear constraints violated after repair: {Ax.flatten()}"
    )

    assert not np.allclose(new_positions, parameters), (
        "Positions were not altered from infeasible start"
    )


def test_pso_repairs_infeasible_population_no_linear_constraints():
    """PSOEngine should repair an entire population that starts outside bounds."""

    engine = PSOEngine()

    # Three particles in 2D, all outside bounds and violating lb and ub
    parameters = np.array([[5.0, 5.0], [6.0, 7.0], [10.0, 10.0]], dtype=float)

    # Dummy results
    results = np.array([[1.0], [2.0], [3.0]], dtype=float)

    # Strict box bounds that the initial population violates
    lb = np.array([10.0, 10.0], dtype=float)
    ub = np.array([12.0, 12.0], dtype=float)

    new_positions = engine.update_solution_to_next_iter(
        parameters, results, lb, ub, {0: OptimizationStrategy.MINIMIZE}
    )

    # All positions must be within box bounds
    assert np.all(new_positions >= lb - 1e-8), f"Positions below lb: {new_positions}"
    assert np.all(new_positions <= ub + 1e-8), f"Positions above ub: {new_positions}"

    assert not np.allclose(new_positions, parameters), (
        "Positions were not altered from infeasible start"
    )


def test_pso_repairs_particles_with_tiny_violations():
    """Particles slightly outside bounds should be corrected within tolerance."""
    engine = PSOEngine()

    epsilon = 1e-9
    parameters = np.array([[0.0 - epsilon, 4.0 + epsilon]], dtype=float)
    results = np.array([[1.0]], dtype=float)

    lb = np.array([0.0, 0.0], dtype=float)
    ub = np.array([4.0, 4.0], dtype=float)

    new_positions = engine.update_solution_to_next_iter(
        parameters, results, lb, ub, {0: OptimizationStrategy.MINIMIZE}
    )

    assert np.all(new_positions >= lb - 1e-8), f"Positions below lb: {new_positions}"
    assert np.all(new_positions <= ub + 1e-8), f"Positions above ub: {new_positions}"


def test_pso_repairs_multiple_linear_constraints():
    """Particles should satisfy multiple linear inequalities after repair."""
    engine = PSOEngine()

    parameters = np.array([[5.0, 5.0], [6.0, -1.0]], dtype=float)
    results = np.array([[1.0], [2.0]], dtype=float)

    lb = np.array([0.0, 0.0], dtype=float)
    ub = np.array([4.0, 4.0], dtype=float)

    # Two constraints: x0 + x1 <= 4, and x0 - x1 <= 1
    A = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=float)
    b = np.array([4.0, 1.0], dtype=float)

    new_positions = engine.update_solution_to_next_iter(
        parameters, results, lb, ub, {0: OptimizationStrategy.MINIMIZE}, A, b
    )

    Ax = (A @ new_positions.T).T
    assert np.all(Ax <= b + 1e-6), f"Multiple linear constraints violated: {Ax}"


def test_pso_repairs_large_population():
    """Repair should handle large populations and make all feasible."""
    engine = PSOEngine()

    parameters = np.random.uniform(5.0, 10.0, size=(100, 2))
    results = np.random.rand(100, 1)

    lb = np.array([0.0, 0.0], dtype=float)
    ub = np.array([4.0, 4.0], dtype=float)
    A = np.array([[1.0, 1.0]], dtype=float)
    b = np.array([4.0], dtype=float)

    new_positions = engine.update_solution_to_next_iter(
        parameters, results, lb, ub, {0: OptimizationStrategy.MINIMIZE}, A, b
    )

    assert new_positions.shape == parameters.shape
    assert np.all(new_positions >= lb - 1e-8), f"Positions below lb: {new_positions}"
    assert np.all(new_positions <= ub + 1e-8), f"Positions above ub: {new_positions}"
    Ax = (A @ new_positions.T).T
    assert np.all(Ax <= b + 1e-6), (
        f"Linear constraints violated in large population: {Ax.flatten()}"
    )


def test_pso_repair_deterministic_with_seed():
    """Repair should be deterministic under fixed random seed."""
    engine1 = PSOEngine(seed=42)
    engine2 = PSOEngine(seed=42)

    parameters = np.array([[5.0, 5.0], [6.0, 7.0]], dtype=float)
    results = np.array([[1.0], [2.0]], dtype=float)

    lb = np.array([0.0, 0.0], dtype=float)
    ub = np.array([4.0, 4.0], dtype=float)

    positions1 = engine1.update_solution_to_next_iter(
        parameters, results, lb, ub, {0: OptimizationStrategy.MINIMIZE}
    )
    positions2 = engine2.update_solution_to_next_iter(
        parameters, results, lb, ub, {0: OptimizationStrategy.MINIMIZE}
    )

    assert np.allclose(positions1, positions2), (
        f"Repair not deterministic with fixed seed:\n{positions1}\n{positions2}"
    )


def test_pso_keeps_md_within_bounds_when_optimal_out_of_reach():
    """If the unconstrained optimum lies beyond the md upper bound, PSO must keep values within [lb, ub]."""
    engine = PSOEngine()

    parameters = np.array([[95.0], [96.0], [97.0]], dtype=float)
    results = np.array([[10.0], [9.0], [8.0]], dtype=float)

    lb = np.array([0.0], dtype=float)
    ub = np.array([100.0], dtype=float)

    _ = engine.update_solution_to_next_iter(
        parameters, results, lb, ub, {0: OptimizationStrategy.MINIMIZE}
    )

    # Force large positive velocities to attempt to move beyond upper bound
    engine._state.velocities = np.array([[20.0], [30.0], [50.0]], dtype=float)

    # The unconstrained step would be parameters + velocities and should exceed ub
    attempted = parameters + engine._state.velocities
    assert np.any(attempted > ub), (
        "Test setup failed to push unconstrained positions beyond ub"
    )

    # The second call should clip/reflection the positions to remain within bounds
    new_positions = engine.update_solution_to_next_iter(
        parameters, results, lb, ub, {0: OptimizationStrategy.MINIMIZE}
    )

    assert np.all(new_positions <= ub + 1e-8), f"Positions above ub: {new_positions}"
    assert np.all(new_positions >= lb - 1e-8), f"Positions below lb: {new_positions}"
    assert not np.allclose(new_positions, parameters), (
        "Positions did not change as expected"
    )
