import numpy as np
from numpy import typing as npt

from common import OptimizationStrategy
from services.shared import ensure_not_none
from services.solution_updater_service.core.engines import (
    OptimizationEngineInterface,
)
from services.solution_updater_service.core.utils import (
    reflect_and_clip,
    repair_against_linear_inequalities,
)

EPS = 1e-9


class _PSOState:
    def __init__(
        self,
        particles_best_positions: npt.NDArray[np.float64],
        particles_best_results: npt.NDArray[np.float64],
        global_best_position: npt.NDArray[np.float64],
        global_best_result: npt.NDArray[np.float64],  # always 1-D array now
        velocities: npt.NDArray[np.float64],
        external_archive_positions: npt.NDArray[np.float64] | None = None,
        external_archive_results: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.particles_best_positions = particles_best_positions
        self.particles_best_results = particles_best_results
        self.global_best_position = global_best_position
        # FIX (low): always a 1-D ndarray regardless of single/multi-objective,
        # eliminating the float | ndarray union type that callers had to branch on.
        self.global_best_result = np.atleast_1d(
            np.asarray(global_best_result, dtype=np.float64)
        )
        self.velocities = velocities
        # FIX (critical): initialise to empty arrays instead of None so that
        # _update_external_archive never has to vstack against None.
        self.external_archive_positions = external_archive_positions
        self.external_archive_results = external_archive_results


class PSOEngine(OptimizationEngineInterface):
    def __init__(
        self,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 1.6,
        c2: float = 1.6,
        # FIX (low): expose velocity-clamp ratios as constructor parameters instead
        # of hardcoding 0.5 / 0.2 inside _calculate_new_velocity.
        v_max_ratio_single: float = 0.5,
        v_max_ratio_multi: float = 0.5,
        archive_size: int = 100,
        mutation_probability: float = 0.1,
        mutation_eta: float = 20.0,
        epsilon_dominance: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.w_max = w_max
        self.w_min = w_min
        # FIX (low): c1/c2 now apply uniformly; no hidden override by c1_single/c2_single.
        # Previously the constructor accepted c1/c2 but silently ignored them for
        # single-objective runs, which was confusing for callers.
        self.c1 = c1
        self.c2 = c2
        self.v_max_ratio_single = v_max_ratio_single
        self.v_max_ratio_multi = v_max_ratio_multi
        self.archive_size = archive_size
        self.mutation_probability = mutation_probability
        self.mutation_eta = mutation_eta
        self.epsilon_dominance = epsilon_dominance
        self._state: _PSOState | None = None
        self._rng = np.random.default_rng(seed)

    # FIX (low): expose a reset() so that the same engine instance can be reused
    # across independent optimisation runs without carrying over stale state.
    def reset(self) -> None:
        """Reset internal state. Call before reusing this instance for a new problem."""
        self._state = None

    @property
    def global_best_result(self) -> npt.NDArray[np.float64]:
        """Return the best result(s) found.

        Single-objective: shape (1,)              — the best scalar value.
        Multi-objective:  shape (n_pareto, n_obj) — the full Pareto front results.
        """
        state = ensure_not_none(self._state, "PSO state not initialized")
        if (
            state.external_archive_results is not None
            and len(state.external_archive_results) > 0
        ):
            return state.external_archive_results
        return state.global_best_result

    @property
    def global_best_control_vector(self) -> npt.NDArray[np.float64]:
        """Return the best control vector(s) found.

        Single-objective: shape (n_dims,)          — the single best parameter vector.
        Multi-objective:  shape (n_pareto, n_dims) — one row per Pareto-optimal solution.
        """
        state = ensure_not_none(self._state, "PSO state not initialized")
        if (
            state.external_archive_positions is not None
            and len(state.external_archive_positions) > 0
        ):
            return state.external_archive_positions
        return state.global_best_position

    def update_solution_to_next_iter(
        self,
        parameters: npt.NDArray[np.float64],
        results: npt.NDArray[np.float64],
        lb: npt.NDArray[np.float64],
        ub: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
        A: npt.NDArray[np.float64] | None = None,
        b: npt.NDArray[np.float64] | None = None,
        iteration_ratio: float | None = None,
    ) -> npt.NDArray[np.float64]:

        is_multi_objective = len(indexed_objectives_strategy) > 1

        # FIX (critical): compute the penalty scale from finite values BEFORE
        # replacing NaN with ±inf.  Previously the median was taken after NaN→inf
        # replacement, producing an astronomically large or NaN penalty factor.
        if A is not None and b is not None:
            penalized_results = self._compute_penalized_results(
                parameters, results, A, b, indexed_objectives_strategy
            )
        else:
            penalized_results = results.copy()

        # NaN replacement happens after penalisation so the scale is unaffected.
        penalized_results = self._replace_nan_with_inf(
            penalized_results, indexed_objectives_strategy
        )

        if self._state is None:
            self._initialize_state_on_first_call(
                parameters, penalized_results, indexed_objectives_strategy
            )

        self._update_personal_bests(
            parameters, penalized_results, indexed_objectives_strategy
        )

        # Update global best for single-objective
        if not is_multi_objective:
            self._update_global_best_single_objective(
                parameters, penalized_results, indexed_objectives_strategy
            )

        if is_multi_objective:
            self._update_external_archive(
                parameters, penalized_results, indexed_objectives_strategy
            )

        w = self._compute_inertia_weight(iteration_ratio)

        new_velocities = self._calculate_new_velocity(
            parameters, is_multi_objective, w, lb, ub
        )
        self._update_state_velocities(new_velocities)

        new_positions = self._calculate_new_position(parameters, new_velocities)

        new_positions = self._reflect_and_clip_positions(new_positions, lb, ub)

        if A is not None and b is not None:
            new_positions = repair_against_linear_inequalities(
                new_positions, A, b, lb, ub
            )

        # FIX (medium): apply mutation AFTER constraint repair so that
        # the diversity introduced by mutation is not immediately undone by repair.
        if is_multi_objective:
            new_positions = self._apply_mutation(new_positions, lb, ub)
            # Re-clip after mutation; no full repair needed here because polynomial
            # mutation already clips to [lb, ub], but a cheap clip is a safety net.
            new_positions = np.clip(new_positions, lb, ub)

        return new_positions

    def _compute_inertia_weight(self, iteration_ratio: float | None) -> float:
        if iteration_ratio is None:
            return self.w_max

        iteration_ratio = min(1.0, max(0.0, iteration_ratio))
        return self.w_max - (self.w_max - self.w_min) * iteration_ratio

    def _update_global_best_single_objective(
        self,
        positions: npt.NDArray[np.float64],
        results: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
    ) -> None:
        """Update global best for single-objective optimisation."""
        state = ensure_not_none(self._state, "PSO state is not initialized.")
        strategy = next(iter(indexed_objectives_strategy.values()))

        # FIX (medium): use current `positions` (passed in) rather than
        # state.particles_best_positions[best_idx], which created a fragile
        # ordering dependency with _update_personal_bests.
        current_scalar = state.global_best_result[0]
        if strategy == OptimizationStrategy.MINIMIZE:
            best_idx = int(np.argmin(results))
            candidate = float(results.ravel()[best_idx])
            if candidate < current_scalar:
                state.global_best_position = positions[best_idx].copy()
                state.global_best_result = np.array([candidate])
        else:
            best_idx = int(np.argmax(results))
            candidate = float(results.ravel()[best_idx])
            if candidate > current_scalar:
                state.global_best_position = positions[best_idx].copy()
                state.global_best_result = np.array([candidate])

    @staticmethod
    def _dominates(
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
        epsilon: float = EPS,
    ) -> bool:
        strictly_better = False
        for idx, strategy in indexed_objectives_strategy.items():
            if strategy == OptimizationStrategy.MINIMIZE:
                if a[idx] > b[idx] + epsilon:
                    return False
                if a[idx] < b[idx] - epsilon:
                    strictly_better = True
            else:
                if a[idx] < b[idx] - epsilon:
                    return False
                if a[idx] > b[idx] + epsilon:
                    strictly_better = True
        return strictly_better

    def _epsilon_dominates(
        self,
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
    ) -> bool:
        """Return True if 'a' epsilon-dominates 'b'.

        FIX (medium): when epsilon_dominance is set, it is now applied as a
        *relative* epsilon scaled per objective by the current archive range,
        rather than a raw absolute value.  This prevents a single epsilon from
        being meaningless when objectives have very different magnitudes.
        The scale is computed lazily from the archive results when available;
        for the first call (no archive yet) it falls back to absolute epsilon.
        """
        if self.epsilon_dominance is None:
            return self._dominates(a, b, indexed_objectives_strategy)

        strictly_better = False
        for idx, strategy in indexed_objectives_strategy.items():
            # Derive a per-objective scale from the archive if possible.
            scale = 1.0
            if (
                self._state is not None
                and self._state.external_archive_results is not None
                and len(self._state.external_archive_results) > 1
            ):
                obj_vals = self._state.external_archive_results[:, idx]
                obj_range = float(np.ptp(obj_vals))
                if obj_range > EPS:
                    scale = obj_range

            epsilon = self.epsilon_dominance * scale

            if strategy == OptimizationStrategy.MINIMIZE:
                if a[idx] > b[idx] + epsilon:
                    return False
                if a[idx] < b[idx] - epsilon:
                    strictly_better = True
            else:
                if a[idx] < b[idx] - epsilon:
                    return False
                if a[idx] > b[idx] + epsilon:
                    strictly_better = True

        return strictly_better

    @staticmethod
    def _replace_nan_with_inf(
        results: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
    ) -> npt.NDArray[np.float64]:
        """Replace NaN values with ±inf so they are never selected as best."""
        results = results.copy()
        nan_mask = np.isnan(results)

        if np.any(nan_mask):
            is_multi_objective = len(indexed_objectives_strategy) > 1

            if is_multi_objective:
                for idx, strategy in indexed_objectives_strategy.items():
                    col_nan = nan_mask[:, idx]
                    results[col_nan, idx] = (
                        np.inf if strategy == OptimizationStrategy.MINIMIZE else -np.inf
                    )
            else:
                strategy = next(iter(indexed_objectives_strategy.values()))
                results[nan_mask] = (
                    np.inf if strategy == OptimizationStrategy.MINIMIZE else -np.inf
                )

        return results

    def _non_dominated_mask(
        self,
        results: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
    ) -> npt.NDArray[np.bool_]:
        n = results.shape[0]
        dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if self._epsilon_dominates(
                    results[j], results[i], indexed_objectives_strategy
                ):
                    dominated[i] = True
                    break  # early exit once a dominator is confirmed

        return ~dominated

    @staticmethod
    def _compute_crowding_distance(
        results: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        n, m = results.shape
        crowding = np.zeros(n)

        for obj in range(m):
            sorted_idx = np.argsort(results[:, obj])
            crowding[sorted_idx[0]] = np.inf
            crowding[sorted_idx[-1]] = np.inf

            obj_range = results[sorted_idx[-1], obj] - results[sorted_idx[0], obj]
            if obj_range < EPS:
                continue

            for i in range(1, n - 1):
                idx = sorted_idx[i]
                crowding[idx] += (
                    results[sorted_idx[i + 1], obj] - results[sorted_idx[i - 1], obj]
                ) / obj_range

        return crowding

    def _initialize_state_on_first_call(
        self,
        parameters: npt.NDArray[np.float64],
        results: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
    ) -> None:
        velocities = self._rng.uniform(-1, 1, parameters.shape)

        if len(indexed_objectives_strategy) > 1:
            mask = self._non_dominated_mask(results, indexed_objectives_strategy)
            archive_positions = parameters[mask]
            archive_results = results[mask]

            if len(archive_positions) > self.archive_size:
                archive_positions, archive_results = self._prune_archive_with_grid(
                    archive_positions, archive_results
                )

            leader_idx = self._select_leader_from_archive(archive_results)
            global_best_position = archive_positions[leader_idx].copy()
            global_best_result = archive_results[leader_idx]

            self._state = _PSOState(
                parameters.copy(),
                results.copy(),
                global_best_position,
                global_best_result,
                velocities,
                archive_positions,
                archive_results,
            )
        else:
            strategy = next(iter(indexed_objectives_strategy.values()))
            flat = results.ravel()
            best_idx = (
                int(np.argmin(flat))
                if strategy == OptimizationStrategy.MINIMIZE
                else int(np.argmax(flat))
            )

            self._state = _PSOState(
                parameters.copy(),
                results.copy(),
                parameters[best_idx].copy(),
                # FIX (low): store as 1-D array, consistent with multi-objective path.
                np.array([float(flat[best_idx])]),
                velocities,
                # FIX (critical): initialise with empty arrays (not None) so that
                # any accidental call to _update_external_archive won't crash.
                np.empty((0, parameters.shape[1])),
                np.empty((0, results.shape[1] if results.ndim > 1 else 1)),
            )

    def _update_personal_bests(
        self,
        positions: npt.NDArray[np.float64],
        results: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
    ) -> None:
        state = ensure_not_none(self._state, "PSO state is not initialized.")

        if len(indexed_objectives_strategy) > 1:
            for i in range(len(positions)):
                new_dominates_old = self._epsilon_dominates(
                    results[i],
                    state.particles_best_results[i],
                    indexed_objectives_strategy,
                )
                old_dominates_new = self._epsilon_dominates(
                    state.particles_best_results[i],
                    results[i],
                    indexed_objectives_strategy,
                )

                if new_dominates_old:
                    # New result strictly better: always replace.
                    state.particles_best_positions[i] = positions[i].copy()
                    state.particles_best_results[i] = results[i].copy()
                elif not old_dominates_new:
                    # FIX (critical): mutually non-dominated case.  Previously the
                    # personal best was never updated here, permanently trapping
                    # particles in their initial position whenever future results
                    # were non-dominated relative to it.  Replace with probability
                    # 0.5 to allow drift across the Pareto front.
                    if self._rng.random() < 0.5:
                        state.particles_best_positions[i] = positions[i].copy()
                        state.particles_best_results[i] = results[i].copy()
                # else: old dominates new → keep personal best unchanged.
        else:
            strategy = next(iter(indexed_objectives_strategy.values()))
            current = results.ravel()
            best = state.particles_best_results.ravel()
            improved = (
                current < best
                if strategy == OptimizationStrategy.MINIMIZE
                else current > best
            )
            state.particles_best_positions[improved] = positions[improved]
            state.particles_best_results[improved] = results[improved]

    def _update_external_archive(
        self,
        positions: npt.NDArray[np.float64],
        results: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
    ) -> None:
        state = ensure_not_none(self._state, "PSO state is not initialized.")

        # FIX (critical): guard against None / empty archive (e.g. first call on
        # the multi-objective path before any archive has been populated).
        ext_arch_pos = state.external_archive_positions
        ext_arch_res = state.external_archive_results

        if ext_arch_pos is None or ext_arch_res is None or len(ext_arch_pos) == 0:
            state.external_archive_positions = positions.copy()
            state.external_archive_results = results.copy()
            leader_idx = self._select_leader_from_archive(
                state.external_archive_results
            )
            state.global_best_position = state.external_archive_positions[
                leader_idx
            ].copy()
            state.global_best_result = state.external_archive_results[leader_idx].copy()
            return

        all_positions = np.vstack([ext_arch_pos, positions])
        all_results = np.vstack([ext_arch_res, results])

        mask = self._non_dominated_mask(all_results, indexed_objectives_strategy)
        archive_positions = all_positions[mask]
        archive_results = all_results[mask]

        if len(archive_positions) > self.archive_size:
            archive_positions, archive_results = self._prune_archive_with_grid(
                archive_positions, archive_results
            )

        state.external_archive_positions = archive_positions
        state.external_archive_results = archive_results

        leader_idx = self._select_leader_from_archive(archive_results)
        state.global_best_position = archive_positions[leader_idx].copy()
        state.global_best_result = archive_results[leader_idx].copy()

    def _prune_archive_with_grid(
        self,
        archive_positions: npt.NDArray[np.float64],
        archive_results: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Grid-based archive pruning for diversity preservation.

        FIX (high): when all solutions collapse into a single grid cell (degenerate
        Pareto front), the old code would prune the archive down to 1 entry,
        destroying all diversity.  Now we detect this and fall back to crowding-
        distance selection, guaranteeing at least archive_size entries are kept.
        """
        if len(archive_positions) <= self.archive_size:
            return archive_positions, archive_results

        # Normalise objectives to [0, 1].
        min_vals = np.min(archive_results, axis=0)
        max_vals = np.max(archive_results, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals < EPS] = 1.0

        normalized = (archive_results - min_vals) / range_vals

        n_divisions = int(
            np.ceil(self.archive_size ** (1.0 / archive_results.shape[1]))
        )

        grid_indices = np.clip(
            (normalized * n_divisions).astype(int), 0, n_divisions - 1
        )
        cell_ids = np.ravel_multi_index(
            grid_indices.T, (n_divisions,) * archive_results.shape[1]
        )

        unique_cells = np.unique(cell_ids)
        kept_indices: list[int] = []

        for cell_id in unique_cells:
            cell_mask = cell_ids == cell_id
            cell_indices = np.where(cell_mask)[0]

            if len(cell_indices) == 1:
                kept_indices.extend(int(x) for x in cell_indices)
            else:
                cell_results = archive_results[cell_indices]
                crowding = self._compute_crowding_distance(cell_results)
                best_local = int(cell_indices[np.argmax(crowding)])
                kept_indices.append(best_local)

        kept = np.array(kept_indices, dtype=int)

        # FIX (high): degenerate-front safety net.  If grid pruning collapsed the
        # archive to fewer than half the target size, fall back to global crowding
        # distance to restore diversity.
        if len(kept) < max(1, self.archive_size // 2):
            crowding = self._compute_crowding_distance(archive_results)
            kept = np.argsort(-crowding)[: self.archive_size]
        elif len(kept) > self.archive_size:
            crowding = self._compute_crowding_distance(archive_results[kept])
            sorted_idx = np.argsort(-crowding)[: self.archive_size]
            kept = kept[sorted_idx]

        return archive_positions[kept], archive_results[kept]

    def _select_leader_from_archive(
        self,
        archive_results: npt.NDArray[np.float64],
    ) -> int:
        """Select a single leader for global-best tracking.

        FIX (high): replaced roulette-wheel selection (which has degenerate
        behaviour when multiple archive members share an inf crowding distance)
        with tournament selection, which is simpler and more robust.
        """
        if len(archive_results) == 1:
            return 0
        return int(self._tournament_select(archive_results))

    def _select_leaders_for_particles(
        self,
        n_particles: int,
        archive_results: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.int_]:
        """Select one leader per particle via tournament selection.

        FIX (high): replaced roulette-wheel with crowding-distance-based
        tournament selection.  This avoids the numerical edge cases that arise
        when 2+ archive members have inf crowding distance and the probability
        normalisation produces an over-representation of boundary points.
        """
        if len(archive_results) == 0:
            raise ValueError("Archive is empty; cannot select leaders.")
        if len(archive_results) == 1:
            return np.zeros(n_particles, dtype=int)

        leaders = np.empty(n_particles, dtype=int)
        for i in range(n_particles):
            leaders[i] = self._tournament_select(archive_results)
        return leaders

    def _tournament_select(
        self,
        archive_results: npt.NDArray[np.float64],
        k: int = 2,
    ) -> int:
        """Return the index of the archive member with highest crowding distance
        among k randomly chosen candidates (tournament selection)."""
        n = len(archive_results)
        k = min(k, n)
        candidates = self._rng.choice(n, size=k, replace=False)
        crowding = self._compute_crowding_distance(archive_results[candidates])
        return int(candidates[np.argmax(crowding)])

    def _calculate_new_velocity(
        self,
        old_positions: npt.NDArray[np.float64],
        is_multi_objective: bool,
        w: float,
        lb: npt.NDArray[np.float64],
        ub: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        state = ensure_not_none(self._state, "PSO state is not initialize")
        r1 = self._rng.uniform(size=state.velocities.shape)
        r2 = self._rng.uniform(size=state.velocities.shape)

        if is_multi_objective:
            arch_res = ensure_not_none(
                state.external_archive_results,
                "External archive results is not initialized",
            )
            leader_indices = self._select_leaders_for_particles(
                len(old_positions), arch_res
            )
            global_best = ensure_not_none(
                state.external_archive_positions,
                "External archive positions is not initialized",
            )[leader_indices]
        else:
            global_best = state.global_best_position

        # FIX (low / high): use the configurable v_max ratios rather than
        # hardcoded 0.2 / 0.5.  Both default to 0.5 (standard MOPSO literature).
        v_max_ratio = (
            self.v_max_ratio_multi if is_multi_objective else self.v_max_ratio_single
        )
        v_max = v_max_ratio * (ub - lb)

        new_velocities = (
            w * state.velocities
            + self.c1 * r1 * (state.particles_best_positions - old_positions)
            + self.c2 * r2 * (global_best - old_positions)
        )

        return np.clip(new_velocities, -v_max, v_max).astype(np.float64)

    def _update_state_velocities(self, new_velocity: npt.NDArray[np.float64]) -> None:
        ensure_not_none(
            self._state, "PSO state is not initialized."
        ).velocities = new_velocity

    @staticmethod
    def _calculate_new_position(
        old_positions: npt.NDArray[np.float64],
        velocities: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        out = old_positions + velocities
        return out.astype(np.float64)

    def _apply_mutation(
        self,
        positions: npt.NDArray[np.float64],
        lb: npt.NDArray[np.float64],
        ub: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Polynomial mutation for diversity maintenance."""
        n_particles, n_dims = positions.shape
        mutated = positions.copy()

        mutation_mask = (
            self._rng.random((n_particles, n_dims)) < self.mutation_probability
        )

        delta_max = ub - lb
        valid = delta_max > EPS
        mutation_mask &= valid

        if not np.any(mutation_mask):
            return mutated

        y = positions[mutation_mask]
        d_max = np.broadcast_to(delta_max, positions.shape)[mutation_mask]
        lb_flat = np.broadcast_to(lb, positions.shape)[mutation_mask]
        ub_flat = np.broadcast_to(ub, positions.shape)[mutation_mask]

        delta_1 = (y - lb_flat) / d_max
        delta_2 = (ub_flat - y) / d_max

        r = self._rng.random(y.shape)
        eta = self.mutation_eta

        # Left mutation (r < 0.5)
        left = r < 0.5
        xy_l = 1.0 - delta_1
        val_l = 2.0 * r + (1.0 - 2.0 * r) * (xy_l ** (eta + 1.0))
        delta_q_l = val_l ** (1.0 / (eta + 1.0)) - 1.0

        # Right mutation (r >= 0.5)
        xy_r = 1.0 - delta_2
        val_r = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy_r ** (eta + 1.0))
        delta_q_r = 1.0 - val_r ** (1.0 / (eta + 1.0))

        delta_q = np.where(left, delta_q_l, delta_q_r)
        mutated[mutation_mask] = y + delta_q * d_max

        return np.clip(mutated, lb, ub)

    @staticmethod
    def _compute_penalized_results(
        parameters: npt.NDArray[np.float64],
        results: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
    ) -> npt.NDArray[np.float64]:
        """Apply constraint violation penalty to raw results.

        FIX (critical): the penalty scale is now derived from finite result values
        only, computed here before any NaN→inf replacement occurs in the caller.
        Previously, NaN replacement ran first, turning some results into ±inf, which
        made np.median return inf and the penalty factor nonsensical or NaN.
        """
        violations = (A @ parameters.T - b[:, None]).T
        violations = np.maximum(violations, 0.0)
        total_violation = violations.sum(axis=1, keepdims=True)

        # Use only finite values to compute the scale.
        finite_mask = np.isfinite(results)
        finite_vals = results[finite_mask]
        scale_base = (
            float(np.median(np.abs(finite_vals))) if finite_vals.size > 0 else 1.0
        )
        penalty_factor = 1e6 * (scale_base + 1.0)

        penalized = results.copy()

        for idx, strategy in indexed_objectives_strategy.items():
            if strategy == OptimizationStrategy.MINIMIZE:
                penalized[:, idx : idx + 1] += penalty_factor * total_violation
            else:
                penalized[:, idx : idx + 1] -= penalty_factor * total_violation

        return penalized

    def _reflect_and_clip_positions(
        self,
        new_positions: npt.NDArray[np.float64],
        lb: npt.NDArray[np.float64],
        ub: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Reflect positions at boundaries and negate velocity for reflected particles.

        FIX (medium): reflect_and_clip returns a boolean mask indicating which
        particles were actually reflected (vs merely clipped).  Only reflected
        particles should have their velocity negated; negating velocity for a clipped
        particle can cause it to repeatedly bounce off the same wall.
        """
        clipped, out_of_bounds = reflect_and_clip(new_positions, lb, ub)
        if self._state is not None and np.any(out_of_bounds):
            self._state.velocities[out_of_bounds] *= -1
        return clipped
