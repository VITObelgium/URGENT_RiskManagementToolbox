import numpy as np
import pytest

from common import OptimizationStrategy
from services.solution_updater_service.core.engines.pso import PSOEngine

# Shared helpers
_MINIMIZE = OptimizationStrategy.MINIMIZE
_MAXIMIZE = OptimizationStrategy.MAXIMIZE
_MULTI_OBJ = {0: _MINIMIZE, 1: _MAXIMIZE}


def _make_engine(seed: int = 0, archive_size: int = 100, **kwargs) -> PSOEngine:
    return PSOEngine(seed=seed, archive_size=archive_size, **kwargs)


def _run_one_iter(
    engine: PSOEngine,
    n_particles: int = 8,
    n_dims: int = 3,
    objectives=None,
    lb=None,
    ub=None,
    results=None,
    A=None,
    b=None,
    iteration_ratio: float | None = None,
):
    if objectives is None:
        objectives = _MULTI_OBJ
    rng = np.random.default_rng(42)
    params = rng.uniform(0, 4, (n_particles, n_dims))
    lb = lb if lb is not None else np.zeros(n_dims)
    ub = ub if ub is not None else np.full(n_dims, 4.0)
    if results is None:
        n_obj = len(objectives)
        results = rng.uniform(0, 10, (n_particles, n_obj))
    return (
        engine.update_solution_to_next_iter(
            params, results, lb, ub, objectives, A, b, iteration_ratio
        ),
        params,
        lb,
        ub,
    )


# ---------------------------------------------------------------------------
# Multi-objective integration tests
# ---------------------------------------------------------------------------


class TestMultiObjectiveUpdate:
    def test_returns_correct_shape(self):
        engine = _make_engine()
        new_pos, params, lb, ub = _run_one_iter(engine)
        assert new_pos.shape == params.shape

    def test_positions_within_bounds_after_update(self):
        engine = _make_engine()
        new_pos, _, lb, ub = _run_one_iter(engine)
        assert np.all(new_pos >= lb - 1e-8)
        assert np.all(new_pos <= ub + 1e-8)

    def test_multi_objective_multiple_iterations(self):
        engine = _make_engine(seed=7)
        n_particles, n_dims = 10, 4
        lb = np.zeros(n_dims)
        ub = np.ones(n_dims) * 5.0
        rng = np.random.default_rng(7)
        params = rng.uniform(0, 5, (n_particles, n_dims))

        for i in range(5):
            results = rng.uniform(0, 10, (n_particles, 2))
            new_params = engine.update_solution_to_next_iter(
                params, results, lb, ub, _MULTI_OBJ, iteration_ratio=i / 5.0
            )
            assert np.all(new_params >= lb - 1e-8)
            assert np.all(new_params <= ub + 1e-8)
            params = new_params

    def test_mutation_is_applied_multi_objective(self):
        """With high mutation probability, positions should change across iterations."""
        engine = _make_engine(seed=1, mutation_probability=1.0)
        new_pos, params, _, _ = _run_one_iter(engine)
        # Mutation + velocity update should change at least some positions
        assert not np.allclose(new_pos, params)

    def test_positions_stay_in_bounds_with_linear_constraints(self):
        engine = _make_engine(seed=99)
        n_dims = 2
        lb = np.zeros(n_dims)
        ub = np.full(n_dims, 4.0)
        A = np.array([[1.0, 1.0]])
        b = np.array([4.0])
        results = np.array(
            [[1.0, 2.0], [3.0, 1.0], [2.0, 3.0], [0.5, 1.5], [3.5, 0.5], [1.0, 1.0]]
        )
        params = np.array(
            [[2.0, 1.0], [1.0, 2.0], [0.5, 1.5], [3.0, 0.5], [1.5, 2.0], [2.5, 1.0]]
        )
        new_pos = engine.update_solution_to_next_iter(
            params, results, lb, ub, _MULTI_OBJ, A, b
        )
        assert np.all(new_pos >= lb - 1e-8)
        assert np.all(new_pos <= ub + 1e-8)

    def test_nan_results_handled_multi_objective(self):
        """NaN in results should be replaced with inf/neg-inf without crashing.

        At least one fully-finite row must exist so the archive has a valid Pareto
        entry and crowding distance can be computed properly.
        """
        engine = _make_engine(seed=5)
        # One row has NaN in first objective, two rows are fully finite
        params = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=float)
        results = np.array(
            [[np.nan, 2.0], [1.0, 5.0], [3.0, 3.0], [4.0, 1.0]], dtype=float
        )
        lb = np.zeros(2)
        ub = np.full(2, 5.0)
        new_pos = engine.update_solution_to_next_iter(
            params, results, lb, ub, _MULTI_OBJ
        )
        assert new_pos.shape == params.shape
        assert np.all(new_pos >= lb - 1e-8)
        assert np.all(new_pos <= ub + 1e-8)


# ---------------------------------------------------------------------------
# _dominates static method
# ---------------------------------------------------------------------------


class TestDominates:
    def test_a_dominates_b_all_minimize(self):
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 3.0])
        assert PSOEngine._dominates(a, b, {0: _MINIMIZE, 1: _MINIMIZE})

    def test_b_dominates_a_all_minimize(self):
        a = np.array([3.0, 4.0])
        b = np.array([2.0, 3.0])
        assert not PSOEngine._dominates(a, b, {0: _MINIMIZE, 1: _MINIMIZE})

    def test_equal_not_dominant(self):
        a = np.array([2.0, 3.0])
        b = np.array([2.0, 3.0])
        assert not PSOEngine._dominates(a, b, {0: _MINIMIZE, 1: _MINIMIZE})

    def test_mixed_strategy_dominance(self):
        # a[0]=1 < b[0]=2 (minimize, a better), a[1]=5 > b[1]=3 (maximize, a better)
        a = np.array([1.0, 5.0])
        b = np.array([2.0, 3.0])
        assert PSOEngine._dominates(a, b, {0: _MINIMIZE, 1: _MAXIMIZE})

    def test_mixed_strategy_no_dominance(self):
        # a[0]=1 < b[0]=2 (minimize, a better), a[1]=2 < b[1]=3 (maximize, b better)
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 3.0])
        assert not PSOEngine._dominates(a, b, {0: _MINIMIZE, 1: _MAXIMIZE})

    def test_maximize_a_dominates_b(self):
        a = np.array([5.0, 6.0])
        b = np.array([3.0, 4.0])
        assert PSOEngine._dominates(a, b, {0: _MAXIMIZE, 1: _MAXIMIZE})

    def test_within_epsilon_not_dominant(self):
        eps = 1e-9
        a = np.array([1.0, 2.0])
        b = np.array([1.0 + eps / 2, 2.0])
        # a[0] is slightly less than b[0] but within epsilon → not strictly better
        assert not PSOEngine._dominates(a, b, {0: _MINIMIZE, 1: _MINIMIZE})


# ---------------------------------------------------------------------------
# _epsilon_dominates
# ---------------------------------------------------------------------------


class TestEpsilonDominates:
    def test_without_epsilon_falls_back_to_dominates(self):
        engine = PSOEngine(epsilon_dominance=None)
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 3.0])
        assert engine._epsilon_dominates(a, b, {0: _MINIMIZE, 1: _MINIMIZE})

    def test_with_epsilon_requires_larger_gap(self):
        engine = PSOEngine(epsilon_dominance=1.0)
        # a[0]=0.5 vs b[0]=1.0, difference=0.5 < epsilon=1.0 → not strictly better
        a = np.array([0.5, 0.5])
        b = np.array([1.0, 1.0])
        assert not engine._epsilon_dominates(a, b, {0: _MINIMIZE, 1: _MINIMIZE})

    def test_with_epsilon_large_gap_dominates(self):
        engine = PSOEngine(epsilon_dominance=0.1)
        a = np.array([1.0, 2.0])
        b = np.array([5.0, 8.0])
        assert engine._epsilon_dominates(a, b, {0: _MINIMIZE, 1: _MINIMIZE})

    def test_epsilon_dominates_maximize(self):
        engine = PSOEngine(epsilon_dominance=0.5)
        # a[0]=10 > b[0]=5 by 5.0 > 0.5 → strictly better to maximize
        a = np.array([10.0, 20.0])
        b = np.array([5.0, 10.0])
        assert engine._epsilon_dominates(a, b, {0: _MAXIMIZE, 1: _MAXIMIZE})

    def test_epsilon_no_dominance_maximize(self):
        engine = PSOEngine(epsilon_dominance=0.5)
        # a[0]=5 < b[0]=10 → worse for maximize → returns False immediately
        a = np.array([5.0, 20.0])
        b = np.array([10.0, 10.0])
        assert not engine._epsilon_dominates(a, b, {0: _MAXIMIZE, 1: _MAXIMIZE})


# ---------------------------------------------------------------------------
# _replace_nan_with_inf (multi-objective path)
# ---------------------------------------------------------------------------


class TestReplaceNanWithInf:
    def test_multi_minimize_nan_becomes_inf(self):
        results = np.array([[np.nan, 1.0], [2.0, np.nan]])
        objectives = {0: _MINIMIZE, 1: _MINIMIZE}
        out = PSOEngine._replace_nan_with_inf(results, objectives)
        assert out[0, 0] == np.inf
        assert out[1, 1] == np.inf

    def test_multi_maximize_nan_becomes_neg_inf(self):
        results = np.array([[np.nan, 1.0], [2.0, np.nan]])
        objectives = {0: _MAXIMIZE, 1: _MAXIMIZE}
        out = PSOEngine._replace_nan_with_inf(results, objectives)
        assert out[0, 0] == -np.inf
        assert out[1, 1] == -np.inf

    def test_no_nan_returns_copy_unchanged(self):
        results = np.array([[1.0, 2.0], [3.0, 4.0]])
        objectives = {0: _MINIMIZE, 1: _MAXIMIZE}
        out = PSOEngine._replace_nan_with_inf(results, objectives)
        assert np.allclose(out, results)
        assert out is not results  # it's a copy

    def test_single_minimize_nan_becomes_inf(self):
        results = np.array([np.nan, 2.0, 3.0])
        objectives = {0: _MINIMIZE}
        out = PSOEngine._replace_nan_with_inf(results, objectives)
        assert out[0] == np.inf

    def test_single_maximize_nan_becomes_neg_inf(self):
        results = np.array([np.nan, 2.0, 3.0])
        objectives = {0: _MAXIMIZE}
        out = PSOEngine._replace_nan_with_inf(results, objectives)
        assert out[0] == -np.inf


# ---------------------------------------------------------------------------
# _non_dominated_mask
# ---------------------------------------------------------------------------


class TestNonDominatedMask:
    def test_all_nondominated_on_pareto_front(self):
        """Points on the Pareto front should all be non-dominated."""
        # Perfect trade-off: each point dominates the others in one objective only
        results = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        engine = _make_engine()
        mask = engine._non_dominated_mask(results, {0: _MINIMIZE, 1: _MINIMIZE})
        assert np.all(mask)

    def test_dominated_point_excluded(self):
        # [2, 2] is dominated by [1, 1]
        results = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 0.5]])
        engine = _make_engine()
        mask = engine._non_dominated_mask(results, {0: _MINIMIZE, 1: _MINIMIZE})
        assert mask[0]  # [1,1] is not dominated
        assert not mask[1]  # [2,2] is dominated by [1,1]

    def test_single_point_is_nondominated(self):
        results = np.array([[5.0, 5.0]])
        engine = _make_engine()
        mask = engine._non_dominated_mask(results, {0: _MINIMIZE, 1: _MINIMIZE})
        assert mask[0]


# ---------------------------------------------------------------------------
# _compute_crowding_distance
# ---------------------------------------------------------------------------


class TestComputeCrowdingDistance:
    def test_boundary_points_get_inf(self):
        results = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        crowding = PSOEngine._compute_crowding_distance(results)
        assert crowding[0] == np.inf or crowding[2] == np.inf

    def test_middle_point_finite(self):
        results = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        crowding = PSOEngine._compute_crowding_distance(results)
        # Middle point should have finite distance
        assert np.isfinite(crowding[1])
        assert crowding[1] > 0

    def test_identical_objectives_no_division_error(self):
        results = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        crowding = PSOEngine._compute_crowding_distance(results)
        assert crowding.shape == (3,)

    def test_returns_correct_length(self):
        results = np.random.default_rng(0).uniform(0, 10, (10, 3))
        crowding = PSOEngine._compute_crowding_distance(results)
        assert len(crowding) == 10


# ---------------------------------------------------------------------------
# _select_leader_from_archive
# ---------------------------------------------------------------------------


class TestSelectLeaderFromArchive:
    def test_single_entry_returns_zero(self):
        engine = _make_engine(seed=0)
        results = np.array([[1.0, 2.0]])
        idx = engine._select_leader_from_archive(results)
        assert idx == 0

    def test_returns_valid_index(self):
        engine = _make_engine(seed=0)
        results = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        idx = engine._select_leader_from_archive(results)
        assert 0 <= idx < len(results)

    def test_all_zero_crowding_still_returns_valid(self):
        engine = _make_engine(seed=0)
        # All identical → crowding sums to 0 → fallback to random choice
        results = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        idx = engine._select_leader_from_archive(results)
        assert 0 <= idx < len(results)


# ---------------------------------------------------------------------------
# _select_leaders_for_particles
# ---------------------------------------------------------------------------


class TestSelectLeadersForParticles:
    def test_empty_archive_returns_none(self):
        engine = _make_engine(seed=0)
        with pytest.raises(
            ValueError, match="Archive is empty; cannot select leaders."
        ):
            _ = engine._select_leaders_for_particles(5, np.empty((0, 2)))

    def test_single_archive_entry_returns_zeros(self):
        engine = _make_engine(seed=0)
        results = np.array([[1.0, 2.0]])
        leaders = engine._select_leaders_for_particles(4, results)
        assert leaders is not None
        assert np.all(leaders == 0)
        assert len(leaders) == 4

    def test_returns_valid_indices(self):
        engine = _make_engine(seed=0)
        archive_results = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        n_particles = 10
        leaders = engine._select_leaders_for_particles(n_particles, archive_results)
        assert leaders is not None
        assert len(leaders) == n_particles
        assert np.all(leaders >= 0)
        assert np.all(leaders < len(archive_results))

    def test_all_zero_crowding_still_returns_valid_indices(self):
        engine = _make_engine(seed=42)
        archive_results = np.array([[1.0, 1.0], [1.0, 1.0]])
        leaders = engine._select_leaders_for_particles(3, archive_results)
        assert leaders is not None
        assert np.all((leaders >= 0) & (leaders < 2))


# ---------------------------------------------------------------------------
# _apply_mutation
# ---------------------------------------------------------------------------


class TestApplyMutation:
    def test_no_mutation_when_probability_zero(self):
        engine = PSOEngine(seed=0, mutation_probability=0.0)
        positions = np.array([[1.0, 2.0], [3.0, 1.0]])
        lb = np.zeros(2)
        ub = np.full(2, 4.0)
        mutated = engine._apply_mutation(positions, lb, ub)
        assert np.allclose(mutated, positions)

    def test_mutated_positions_within_bounds(self):
        engine = PSOEngine(seed=0, mutation_probability=1.0)
        positions = np.random.default_rng(0).uniform(0.5, 3.5, (20, 4))
        lb = np.zeros(4)
        ub = np.full(4, 4.0)
        mutated = engine._apply_mutation(positions, lb, ub)
        assert np.all(mutated >= lb)
        assert np.all(mutated <= ub)

    def test_mutation_changes_positions_when_probability_one(self):
        engine = PSOEngine(seed=1, mutation_probability=1.0)
        positions = np.full((5, 3), 2.0)
        lb = np.zeros(3)
        ub = np.full(3, 4.0)
        mutated = engine._apply_mutation(positions, lb, ub)
        # With probability=1 some elements should be mutated
        assert not np.allclose(mutated, positions)

    def test_zero_range_dimension_skipped(self):
        """Dimensions with lb == ub should not be mutated."""
        engine = PSOEngine(seed=0, mutation_probability=1.0)
        positions = np.array([[2.0, 3.0]])
        lb = np.array([0.0, 3.0])  # second dim has zero range
        ub = np.array([4.0, 3.0])
        mutated = engine._apply_mutation(positions, lb, ub)
        # Second dimension must remain unchanged (zero range)
        assert mutated[0, 1] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _compute_penalized_results
# ---------------------------------------------------------------------------


class TestComputePenalizedResults:
    def test_feasible_solutions_unpenalized(self):
        """Solutions satisfying A x <= b should not be penalized."""
        parameters = np.array([[1.0, 1.0], [2.0, 1.0]])
        results = np.array([[5.0], [3.0]])
        A = np.array([[1.0, 1.0]])  # x0 + x1 <= 4
        b = np.array([4.0])
        objectives = {0: _MINIMIZE}
        penalized = PSOEngine._compute_penalized_results(
            parameters, results, A, b, objectives
        )
        assert np.allclose(penalized, results)

    def test_infeasible_solutions_penalized_minimize(self):
        """Violations increase the objective for MINIMIZE."""
        parameters = np.array([[3.0, 3.0]])  # violates x0+x1 <= 4
        results = np.array([[1.0]])
        A = np.array([[1.0, 1.0]])
        b = np.array([4.0])
        objectives = {0: _MINIMIZE}
        penalized = PSOEngine._compute_penalized_results(
            parameters, results, A, b, objectives
        )
        assert penalized[0, 0] > results[0, 0]

    def test_infeasible_solutions_penalized_maximize(self):
        """Violations decrease the objective for MAXIMIZE."""
        parameters = np.array([[3.0, 3.0]])  # violates x0+x1 <= 4
        results = np.array([[100.0]])
        A = np.array([[1.0, 1.0]])
        b = np.array([4.0])
        objectives = {0: _MAXIMIZE}
        penalized = PSOEngine._compute_penalized_results(
            parameters, results, A, b, objectives
        )
        assert penalized[0, 0] < results[0, 0]


# ---------------------------------------------------------------------------
# _prune_archive_with_grid
# ---------------------------------------------------------------------------


class TestPruneArchiveWithGrid:
    def test_no_pruning_when_archive_within_limit(self):
        engine = _make_engine(archive_size=100)
        positions = np.random.default_rng(0).uniform(0, 10, (10, 3))
        results = np.random.default_rng(0).uniform(0, 10, (10, 2))
        kept_pos, kept_res = engine._prune_archive_with_grid(positions, results)
        assert len(kept_pos) == len(positions)

    def test_prunes_to_archive_size(self):
        engine = _make_engine(archive_size=5, seed=0)
        positions = np.random.default_rng(0).uniform(0, 10, (20, 3))
        results = np.random.default_rng(0).uniform(0, 10, (20, 2))
        kept_pos, kept_res = engine._prune_archive_with_grid(positions, results)
        assert len(kept_pos) <= engine.archive_size

    def test_returns_consistent_positions_and_results(self):
        engine = _make_engine(archive_size=5, seed=1)
        positions = np.random.default_rng(1).uniform(0, 10, (30, 2))
        results = np.random.default_rng(1).uniform(0, 10, (30, 2))
        kept_pos, kept_res = engine._prune_archive_with_grid(positions, results)
        assert len(kept_pos) == len(kept_res)


# ---------------------------------------------------------------------------
# global_best_result / global_best_control_vector before initialization
# ---------------------------------------------------------------------------


class TestPSOStateAccessBeforeInit:
    def test_global_best_result_raises_before_init(self):
        engine = _make_engine()
        with pytest.raises(ValueError):
            _ = engine.global_best_result

    def test_global_best_control_vector_raises_before_init(self):
        engine = _make_engine()
        with pytest.raises(ValueError):
            _ = engine.global_best_control_vector


# ---------------------------------------------------------------------------
# Single-objective MAXIMIZE branch (update_global_best)
# ---------------------------------------------------------------------------


class TestSingleObjectiveMaximize:
    def test_maximize_global_best_updates_correctly(self):
        """Global best should track the maximum across iterations."""
        engine = _make_engine(seed=0)
        params = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        lb, ub = np.zeros(2), np.full(2, 5.0)
        objectives = {0: _MAXIMIZE}

        results_iter1 = np.array([[10.0], [20.0], [5.0]])
        engine.update_solution_to_next_iter(params, results_iter1, lb, ub, objectives)
        best_after_1 = engine.global_best_result
        assert best_after_1 == pytest.approx(20.0, rel=1e-3)

        results_iter2 = np.array([[50.0], [1.0], [2.0]])
        engine.update_solution_to_next_iter(params, results_iter2, lb, ub, objectives)
        best_after_2 = engine.global_best_result
        assert best_after_2 == pytest.approx(50.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Inertia weight computation
# ---------------------------------------------------------------------------


class TestComputeInertiaWeight:
    def test_none_ratio_returns_w_max(self):
        engine = PSOEngine(w_max=0.9, w_min=0.4)
        assert engine._compute_inertia_weight(None) == pytest.approx(0.9)

    def test_zero_ratio_returns_w_max(self):
        engine = PSOEngine(w_max=0.9, w_min=0.4)
        assert engine._compute_inertia_weight(0.0) == pytest.approx(0.9)

    def test_one_ratio_returns_w_min(self):
        engine = PSOEngine(w_max=0.9, w_min=0.4)
        assert engine._compute_inertia_weight(1.0) == pytest.approx(0.4)

    def test_half_ratio_returns_midpoint(self):
        engine = PSOEngine(w_max=0.9, w_min=0.4)
        assert engine._compute_inertia_weight(0.5) == pytest.approx(0.65)

    def test_clamped_above_one(self):
        engine = PSOEngine(w_max=0.9, w_min=0.4)
        assert engine._compute_inertia_weight(2.0) == pytest.approx(0.4)

    def test_clamped_below_zero(self):
        engine = PSOEngine(w_max=0.9, w_min=0.4)
        assert engine._compute_inertia_weight(-0.5) == pytest.approx(0.9)
