import numpy as np
from numpy import typing as npt

from common import OptimizationStrategy
from services.solution_updater_service.core.engines import (
    OptimizationEngineInterface,
)
from services.solution_updater_service.core.utils import (
    ensure_not_none,
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
        global_best_result: float | npt.NDArray[np.float64],
        velocities: npt.NDArray[np.float64],
        external_archive_positions: npt.NDArray[np.float64] | None = None,
        external_archive_results: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.particles_best_positions = particles_best_positions
        self.particles_best_results = particles_best_results
        self.global_best_position = global_best_position
        self.global_best_result = global_best_result
        self.velocities = velocities
        self.external_archive_positions = external_archive_positions
        self.external_archive_results = external_archive_results


class PSOEngine(OptimizationEngineInterface):
    def __init__(
        self,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 1.6,
        c2: float = 1.6,
        archive_size: int = 100,
        mutation_probability: float = 0.1,
        mutation_eta: float = 20.0,
        epsilon_dominance: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        # For single-objective: stronger exploitation
        self.c1_single = 1.49445  # Cognitive component
        self.c2_single = 1.49445  # Social component (standard PSO parameters)
        self.archive_size = archive_size
        self.mutation_probability = mutation_probability
        self.mutation_eta = mutation_eta
        self.epsilon_dominance = epsilon_dominance
        self._state: _PSOState | None = None
        self._rng = np.random.default_rng(seed)

    @property
    def global_best_control_vector(self) -> npt.NDArray[np.float64]:
        return ensure_not_none(self._state).global_best_position

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

        results = self._replace_nan_with_inf(results, indexed_objectives_strategy)

        if A is not None and b is not None:
            penalized_results = self._compute_penalized_results(
                parameters, results, A, b, indexed_objectives_strategy
            )
        else:
            penalized_results = results.copy()

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
                penalized_results, indexed_objectives_strategy
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

        # Apply mutation for multi-objective problems
        if is_multi_objective:
            new_positions = self._apply_mutation(new_positions, lb, ub)

        self._update_generation_summary(penalized_results, indexed_objectives_strategy)

        if A is not None and b is not None:
            new_positions = repair_against_linear_inequalities(
                new_positions, A, b, lb, ub
            )

        return new_positions

    def _compute_inertia_weight(self, iteration_ratio: float | None) -> float:
        if iteration_ratio is None:
            return self.w_max

        iteration_ratio = min(1.0, max(0.0, iteration_ratio))
        return self.w_max - (self.w_max - self.w_min) * iteration_ratio

    def _update_global_best_single_objective(
        self, results, indexed_objectives_strategy
    ):
        """Update global best for single-objective optimization."""
        state = ensure_not_none(self._state)
        strategy = next(iter(indexed_objectives_strategy.values()))

        if strategy == OptimizationStrategy.MINIMIZE:
            best_idx = np.argmin(results)
            if results[best_idx] < state.global_best_result:
                state.global_best_position = state.particles_best_positions[
                    best_idx
                ].copy()
                state.global_best_result = float(results[best_idx].item())
        else:
            best_idx = np.argmax(results)
            if results[best_idx] > state.global_best_result:
                state.global_best_position = state.particles_best_positions[
                    best_idx
                ].copy()
                state.global_best_result = float(results[best_idx].item())

    @staticmethod
    def _dominates(a, b, indexed_objectives_strategy, epsilon: float = EPS) -> bool:
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

    def _epsilon_dominates(self, a, b, indexed_objectives_strategy) -> bool:
        """Check if 'a' epsilon-dominates 'b'."""
        if self.epsilon_dominance is None:
            return self._dominates(a, b, indexed_objectives_strategy)

        epsilon = self.epsilon_dominance
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

    @staticmethod
    def _replace_nan_with_inf(
        results: npt.NDArray[np.float64],
        indexed_objectives_strategy: dict[int, OptimizationStrategy],
    ) -> npt.NDArray[np.float64]:
        """
        Replace NaN values with +inf (minimize) or -inf (maximize)
        so they are never selected as best values.
        """
        results = results.copy()
        nan_mask = np.isnan(results)

        if np.any(nan_mask):
            is_multi_objective = len(indexed_objectives_strategy) > 1

            if is_multi_objective:
                for idx, strategy in indexed_objectives_strategy.items():
                    if strategy == OptimizationStrategy.MINIMIZE:
                        results[nan_mask[:, idx], idx] = np.inf
                    else:
                        results[nan_mask[:, idx], idx] = -np.inf
            else:
                strategy = next(iter(indexed_objectives_strategy.values()))
                if strategy == OptimizationStrategy.MINIMIZE:
                    results[nan_mask] = np.inf
                else:
                    results[nan_mask] = -np.inf

        return results

    @staticmethod
    def _non_dominated_mask(results, indexed_objectives_strategy):
        n = results.shape[0]
        dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if PSOEngine._dominates(
                    results[j], results[i], indexed_objectives_strategy
                ):
                    dominated[i] = True
                    break

        return ~dominated

    @staticmethod
    def _compute_crowding_distance(results):
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
        self, parameters, results, indexed_objectives_strategy
    ):
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
            global_best_position = archive_positions[leader_idx]
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
            best_idx = (
                np.argmin(results)
                if strategy == OptimizationStrategy.MINIMIZE
                else np.argmax(results)
            )

            self._state = _PSOState(
                parameters.copy(),
                results.copy(),
                parameters[best_idx].copy(),
                float(results[best_idx].item()),
                velocities,
            )

    def _update_personal_bests(self, positions, results, indexed_objectives_strategy):
        state = ensure_not_none(self._state)

        if len(indexed_objectives_strategy) > 1:
            for i in range(len(positions)):
                if self._dominates(
                    results[i],
                    state.particles_best_results[i],
                    indexed_objectives_strategy,
                ):
                    state.particles_best_positions[i] = positions[i]
                    state.particles_best_results[i] = results[i]
        else:
            strategy = next(iter(indexed_objectives_strategy.values()))
            current = results.ravel()
            best = state.particles_best_results.ravel()
            if strategy == OptimizationStrategy.MINIMIZE:
                improved = current < best
            else:
                improved = current > best
            state.particles_best_positions[improved] = positions[improved]
            state.particles_best_results[improved] = results[improved]

    def _update_external_archive(self, positions, results, indexed_objectives_strategy):
        state = ensure_not_none(self._state)

        all_positions = np.vstack([state.external_archive_positions, positions])
        all_results = np.vstack([state.external_archive_results, results])

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
        state.global_best_position = archive_positions[leader_idx]
        state.global_best_result = archive_results[leader_idx]

    def _prune_archive_with_grid(self, archive_positions, archive_results):
        """Grid-based archive pruning for better diversity."""
        if len(archive_positions) <= self.archive_size:
            return archive_positions, archive_results

        # Normalize objectives to [0, 1]
        min_vals = np.min(archive_results, axis=0)
        max_vals = np.max(archive_results, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals < EPS] = 1.0

        normalized = (archive_results - min_vals) / range_vals

        # Create adaptive grid
        n_divisions = int(
            np.ceil(self.archive_size ** (1.0 / archive_results.shape[1]))
        )

        # Assign solutions to grid cells
        grid_indices = (normalized * n_divisions).astype(int)
        grid_indices = np.clip(grid_indices, 0, n_divisions - 1)

        # Convert to unique cell identifiers
        cell_ids = np.ravel_multi_index(
            grid_indices.T, (n_divisions,) * archive_results.shape[1]
        )

        # Count solutions per cell
        unique_cells = np.unique(cell_ids)

        # Keep one solution from each cell
        kept_indices = []
        for cell_id in unique_cells:
            cell_mask = cell_ids == cell_id
            cell_indices = np.where(cell_mask)[0]

            if len(cell_indices) == 1:
                kept_indices.extend(cell_indices)
            else:
                # Keep solution with best crowding distance in crowded cells
                cell_results = archive_results[cell_indices]
                crowding = self._compute_crowding_distance(cell_results)
                best_local = cell_indices[np.argmax(crowding)]
                kept_indices.append(best_local)

        kept_indices = np.array(kept_indices)

        # If still too many, use crowding distance
        if len(kept_indices) > self.archive_size:
            crowding = self._compute_crowding_distance(archive_results[kept_indices])
            sorted_idx = np.argsort(-crowding)[: self.archive_size]
            kept_indices = kept_indices[sorted_idx]

        return archive_positions[kept_indices], archive_results[kept_indices]

    def _select_leader_from_archive(self, archive_results):
        """Select a single leader from archive (used for global best tracking)."""
        if len(archive_results) == 1:
            return 0

        crowding = self._compute_crowding_distance(archive_results)
        crowding[np.isinf(crowding)] = (
            np.max(crowding[np.isfinite(crowding)]) * 2
            if np.any(np.isfinite(crowding))
            else 1.0
        )

        total = np.sum(crowding)
        if total < EPS:
            return self._rng.integers(0, len(archive_results))

        probabilities = crowding / total
        return self._rng.choice(len(archive_results), p=probabilities)

    def _select_leaders_for_particles(self, n_particles, archive_results):
        """Select a leader from archive for each particle independently."""
        if len(archive_results) == 0:
            return None

        if len(archive_results) == 1:
            return np.zeros(n_particles, dtype=int)

        crowding = self._compute_crowding_distance(archive_results)
        crowding[np.isinf(crowding)] = (
            np.max(crowding[np.isfinite(crowding)]) * 2
            if np.any(np.isfinite(crowding))
            else 1.0
        )

        total = np.sum(crowding)
        if total < EPS:
            return self._rng.integers(0, len(archive_results), size=n_particles)

        probabilities = crowding / total
        return self._rng.choice(len(archive_results), size=n_particles, p=probabilities)

    def _calculate_new_velocity(self, old_positions, is_multi_objective, w, lb, ub):
        state = ensure_not_none(self._state)
        r1 = self._rng.uniform(size=state.velocities.shape)
        r2 = self._rng.uniform(size=state.velocities.shape)

        if is_multi_objective:
            # Each particle gets its own leader for multi-objective
            leader_indices = self._select_leaders_for_particles(
                len(old_positions), state.external_archive_results
            )
            global_best = state.external_archive_positions[leader_indices]
            c1, c2 = self.c1, self.c2
        else:
            # Single-objective: all particles follow the same global best
            global_best = state.global_best_position
            # Use stronger parameters for single-objective
            c1, c2 = self.c1_single, self.c2_single

        new_velocities = (
            w * state.velocities
            + c1 * r1 * (state.particles_best_positions - old_positions)
            + c2 * r2 * (global_best - old_positions)
        )

        # Velocity clamping
        v_max = 0.5 * (ub - lb) if not is_multi_objective else 0.2 * (ub - lb)
        return np.clip(new_velocities, -v_max, v_max)

    def _update_state_velocities(self, new_velocity):
        ensure_not_none(self._state).velocities = new_velocity

    @staticmethod
    def _calculate_new_position(old_positions, velocities):
        return old_positions + velocities

    def _apply_mutation(self, positions, lb, ub):
        """Apply polynomial mutation to maintain diversity."""
        n_particles, n_dims = positions.shape
        mutated = positions.copy()

        # Determine which elements mutate
        mutation_mask = (
            self._rng.random((n_particles, n_dims)) < self.mutation_probability
        )

        delta_max = ub - lb
        valid = delta_max > EPS
        mutation_mask &= valid  # Skip dimensions with zero range

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
        parameters, results, A, b, indexed_objectives_strategy
    ):
        violations = (A @ parameters.T - b[:, None]).T
        violations = np.maximum(violations, 0.0)
        total_violation = violations.sum(axis=1, keepdims=True)
        penalty_factor = 1e6 * (np.median(np.abs(results)) + 1.0)

        penalized = results.copy()

        for idx, strategy in indexed_objectives_strategy.items():
            if strategy == OptimizationStrategy.MINIMIZE:
                penalized[:, idx : idx + 1] += penalty_factor * total_violation
            else:
                penalized[:, idx : idx + 1] -= penalty_factor * total_violation

        return penalized

    def _reflect_and_clip_positions(self, new_positions, lb, ub):
        clipped, out_of_bounds = reflect_and_clip(new_positions, lb, ub)
        if self._state:
            self._state.velocities[out_of_bounds] *= -1
        return clipped
