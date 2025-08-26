import numpy as np

from services.solution_updater_service.core.engines.pso import PSOEngine


def test_pso_repairs_infeasible_population():
    """PSOEngine should repair an entire population that starts outside bounds and linear inequalities.

    The test places all particles outside the provided lb/ub and violating the linear
    constraint x0 + x1 <= 4. After a single call to `update_solution_to_next_iter`
    the returned positions must satisfy the box bounds and the linear inequality
    (within a small numerical tolerance).
    """

    rng_seed = 42
    engine = PSOEngine(seed=rng_seed)

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
        parameters, results, lb, ub, A, b
    )

    # Basic sanity
    assert new_positions.shape == parameters.shape

    # All positions must be within box bounds
    assert np.all(new_positions >= lb - 1e-8), f"Positions below lb: {new_positions}"
    assert np.all(new_positions <= ub + 1e-8), f"Positions above ub: {new_positions}"

    # And they should satisfy the linear inequality A x <= b (with small tolerance)
    Ax = (A @ new_positions.T).T
    assert np.all(
        Ax <= b + 1e-6
    ), f"Linear constraints violated after repair: {Ax.flatten()}"

    assert not np.allclose(
        new_positions, parameters
    ), "Positions were not altered from infeasible start"


def test_pso_repairs_infeasible_population_no_linear_constraints():
    """PSOEngine should repair an entire population that starts outside bounds."""

    rng_seed = 42
    engine = PSOEngine(seed=rng_seed)

    # Three particles in 2D, all outside bounds and violating lb and ub
    parameters = np.array([[5.0, 5.0], [6.0, 7.0], [10.0, 10.0]], dtype=float)

    # Dummy results
    results = np.array([[1.0], [2.0], [3.0]], dtype=float)

    # Strict box bounds that the initial population violates
    lb = np.array([10.0, 10.0], dtype=float)
    ub = np.array([12.0, 12.0], dtype=float)

    new_positions = engine.update_solution_to_next_iter(parameters, results, lb, ub)

    # All positions must be within box bounds
    assert np.all(new_positions >= lb - 1e-8), f"Positions below lb: {new_positions}"
    assert np.all(new_positions <= ub + 1e-8), f"Positions above ub: {new_positions}"

    assert not np.allclose(
        new_positions, parameters
    ), "Positions were not altered from infeasible start"


def test_pso_repairs_particles_with_tiny_violations():
    """Particles slightly outside bounds should be corrected within tolerance."""
    rng_seed = 42
    engine = PSOEngine(seed=rng_seed)

    epsilon = 1e-9
    parameters = np.array([[0.0 - epsilon, 4.0 + epsilon]], dtype=float)
    results = np.array([[1.0]], dtype=float)

    lb = np.array([0.0, 0.0], dtype=float)
    ub = np.array([4.0, 4.0], dtype=float)

    new_positions = engine.update_solution_to_next_iter(parameters, results, lb, ub)

    assert np.all(new_positions >= lb - 1e-8), f"Positions below lb: {new_positions}"
    assert np.all(new_positions <= ub + 1e-8), f"Positions above ub: {new_positions}"


def test_pso_repairs_multiple_linear_constraints():
    """Particles should satisfy multiple linear inequalities after repair."""
    rng_seed = 42
    engine = PSOEngine(seed=rng_seed)

    parameters = np.array([[5.0, 5.0], [6.0, -1.0]], dtype=float)
    results = np.array([[1.0], [2.0]], dtype=float)

    lb = np.array([0.0, 0.0], dtype=float)
    ub = np.array([4.0, 4.0], dtype=float)

    # Two constraints: x0 + x1 <= 4, and x0 - x1 <= 1
    A = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=float)
    b = np.array([4.0, 1.0], dtype=float)

    new_positions = engine.update_solution_to_next_iter(
        parameters, results, lb, ub, A, b
    )

    Ax = (A @ new_positions.T).T
    assert np.all(Ax <= b + 1e-6), f"Multiple linear constraints violated: {Ax}"


def test_pso_repairs_large_population():
    """Repair should handle large populations and make all feasible."""
    rng_seed = 42
    engine = PSOEngine(seed=rng_seed)

    np.random.seed(rng_seed)
    parameters = np.random.uniform(5.0, 10.0, size=(100, 2))
    results = np.random.rand(100, 1)

    lb = np.array([0.0, 0.0], dtype=float)
    ub = np.array([4.0, 4.0], dtype=float)
    A = np.array([[1.0, 1.0]], dtype=float)
    b = np.array([4.0], dtype=float)

    new_positions = engine.update_solution_to_next_iter(
        parameters, results, lb, ub, A, b
    )

    assert new_positions.shape == parameters.shape
    assert np.all(new_positions >= lb - 1e-8), f"Positions below lb: {new_positions}"
    assert np.all(new_positions <= ub + 1e-8), f"Positions above ub: {new_positions}"
    Ax = (A @ new_positions.T).T
    assert np.all(
        Ax <= b + 1e-6
    ), f"Linear constraints violated in large population: {Ax.flatten()}"


def test_pso_repair_deterministic_with_seed():
    """Repair should be deterministic under fixed random seed."""
    rng_seed = 42
    engine1 = PSOEngine(seed=rng_seed)
    engine2 = PSOEngine(seed=rng_seed)

    parameters = np.array([[5.0, 5.0], [6.0, 7.0]], dtype=float)
    results = np.array([[1.0], [2.0]], dtype=float)

    lb = np.array([0.0, 0.0], dtype=float)
    ub = np.array([4.0, 4.0], dtype=float)

    positions1 = engine1.update_solution_to_next_iter(parameters, results, lb, ub)
    positions2 = engine2.update_solution_to_next_iter(parameters, results, lb, ub)

    assert np.allclose(
        positions1, positions2
    ), f"Repair not deterministic with fixed seed:\n{positions1}\n{positions2}"
