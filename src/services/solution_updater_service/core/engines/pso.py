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


class _PSOState:
    def __init__(
        self,
        particles_best_positions: npt.NDArray[np.float64],
        particles_best_results: npt.NDArray[np.float64],
        global_best_position: npt.NDArray[np.float64],
        global_best_result: float,
        velocities: npt.NDArray[np.float64],
    ) -> None:
        self.particles_best_positions = particles_best_positions
        self.particles_best_results = particles_best_results
        self.global_best_position = global_best_position
        self.global_best_result = global_best_result
        self.velocities = velocities


class PSOEngine(OptimizationEngineInterface):
    def __init__(
        self,
        strategy: OptimizationStrategy,
        w: float = 0.8,
        c1: float = 1.6,
        c2: float = 1.6,
        seed: int | None = None,
    ) -> None:
        """Initialize PSO parameters with optional random seed for reproducibility.

        Parameters:
            w (float): The inertia weight, which controls the influence of the particle's previous velocity on its current velocity.
                A larger value encourages exploration, while a smaller value promotes exploitation by reducing the influence
                of previous velocities.

            c1 (float): The cognitive coefficient, which determines the degree of attraction the particle has toward its personal
                best-known position. A higher value intensifies the particle's self-awareness, encouraging movement toward its own
                best solution.

            c2 (float): The social coefficient, which defines the influence of the global best-known position on the particle's
                movement. A higher value makes the particle more socially conscious, urging it to move closer to the global best
                solution.
        """
        super().__init__()
        self.w, self.c1, self.c2 = w, c1, c2
        self._state: _PSOState | None = None
        self._rng = np.random.default_rng(seed)
        self._strategy = strategy

    @property
    def global_best_control_vector(self) -> npt.NDArray[np.float64]:
        return ensure_not_none(self._state).global_best_position

    def update_solution_to_next_iter(
        self,
        parameters: npt.NDArray[np.float64],
        results: npt.NDArray[np.float64],
        lb: npt.NDArray[np.float64],
        ub: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64] | None = None,
        b: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Updates particle positions with optional linear inequality constraints A x <= b.

        Notes:
            Original user-provided directions (<=, >=, <, >) are normalized upstream into
            the unified form A x <= b before invoking the engine. Strict inequalities are
            currently treated as non-strict (<=).

        Constraint Handling:
            If A,b provided, feasibility handled by adding a static penalty to objective for ranking
            personal/global bests (Deb's rule simplified: penalized value = result + penalty*sum(violations)).

        NaN Handling:
            NaN results are replaced with +inf (for minimization) or -inf (for maximization)
            so they are never selected as best values.
        """

        # Replace NaN values based on optimization strategy
        results = self._replace_nan_with_inf(results)

        if A is not None and b is not None:
            penalized_results = self._compute_penalized_results(
                parameters, results, A, b
            )
        else:
            penalized_results = results.copy()

        if self._state is None:
            self._initialize_state_on_first_call(parameters, penalized_results)

        self._update_state_positions(parameters, penalized_results)
        new_velocities = self._calculate_new_velocity(parameters)
        self._update_state_velocities(new_velocities)
        new_positions = self._calculate_new_position(parameters, new_velocities)
        self._update_generation_summary(penalized_results)
        new_positions = self._reflect_and_clip_positions(new_positions, lb, ub)

        if A is not None and b is not None:
            new_positions = repair_against_linear_inequalities(
                new_positions, A, b, lb, ub
            )

        return new_positions

    def _replace_nan_with_inf(
        self, results: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Replace NaN values with +inf (minimize) or -inf (maximize) to exclude them from selection."""
        results = results.copy()
        nan_mask = np.isnan(results)

        if np.any(nan_mask):
            if self._strategy == OptimizationStrategy.MINIMIZE:
                # For minimization, NaN becomes +inf (worst possible value)
                results[nan_mask] = np.inf
            else:
                # For maximization, NaN becomes -inf (worst possible value)
                results[nan_mask] = -np.inf

        return results

    def _initialize_state_on_first_call(
        self, parameters: npt.NDArray[np.float64], results: npt.NDArray[np.float64]
    ) -> None:
        """Initializes the state on the first call, setting up best positions, velocities, and global best."""
        best_index = (
            np.argmin(results)
            if self._strategy == OptimizationStrategy.MINIMIZE
            else np.argmax(results)
        )
        self._state = _PSOState(
            particles_best_positions=np.copy(parameters),
            particles_best_results=np.copy(results),
            global_best_position=parameters[best_index],
            global_best_result=float(results[best_index].item()),
            velocities=np.atleast_1d(
                self._rng.uniform(-1, 1, parameters.shape)
            ),  # Random initial velocities
        )

    def _calculate_new_velocity(
        self, old_positions: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Computes new velocities using inertia, cognitive, and social components."""
        state: _PSOState = ensure_not_none(self._state)
        r1, r2 = (
            self._rng.uniform(size=state.velocities.shape),
            self._rng.uniform(size=state.velocities.shape),
        )

        return (
            self.w * state.velocities
            + self.c1 * r1 * (state.particles_best_positions - old_positions)
            + self.c2 * r2 * (state.global_best_position - old_positions)
        ).astype(np.float64)

    def _update_state_positions(
        self, positions: npt.NDArray[np.float64], results: npt.NDArray[np.float64]
    ) -> None:
        """Updates both individual best positions and the global best position."""
        self._update_particles_best_positions(positions, results)
        self._update_global_best_position(positions, results)

    def _update_particles_best_positions(
        self, positions: npt.NDArray[np.float64], results: npt.NDArray[np.float64]
    ) -> None:
        """Updates the best-known positions of each particle."""
        state: _PSOState = ensure_not_none(self._state)
        if self._strategy == OptimizationStrategy.MINIMIZE:
            mask = results < state.particles_best_results
        else:
            mask = results > state.particles_best_results

        state.particles_best_positions[mask[:, 0]] = positions[mask[:, 0]]
        state.particles_best_results[mask] = results[mask]

    def _update_global_best_position(
        self, positions: npt.NDArray[np.float64], results: npt.NDArray[np.float64]
    ) -> None:
        """Updates the global best position if a new best is found."""
        state: _PSOState = ensure_not_none(self._state)
        best_index = (
            np.argmin(results)
            if self._strategy == OptimizationStrategy.MINIMIZE
            else np.argmax(results)
        )
        best_result = float(results[best_index].item())

        if self._strategy == OptimizationStrategy.MINIMIZE:
            is_better = best_result < state.global_best_result
        else:
            is_better = best_result > state.global_best_result

        if is_better:
            state.global_best_position = positions[best_index]
            state.global_best_result = best_result

    def _update_state_velocities(self, new_velocity: npt.NDArray[np.float64]) -> None:
        """Updates the velocity of particles."""
        state: _PSOState = ensure_not_none(self._state)
        state.velocities = new_velocity

    @staticmethod
    def _calculate_new_position(
        old_positions: npt.NDArray[np.float64], velocities: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Computes new positions by adding velocities to old positions."""
        return np.array(old_positions + velocities, dtype=np.float64)

    def _compute_penalized_results(
        self,
        parameters: npt.NDArray[np.float64],
        results: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Computes penalized results for constraint violations."""
        violations = (A @ parameters.T - b[:, None]).T
        violations = np.maximum(violations, 0.0)
        total_violation = violations.sum(axis=1, keepdims=True)
        penalty_factor = 1e6 * (np.median(np.abs(results)) + 1.0)

        # For minimization: add penalty (worse = higher value)
        # For maximization: subtract penalty (worse = lower value)
        if self._strategy == OptimizationStrategy.MINIMIZE:
            return results + penalty_factor * total_violation
        else:
            return results - penalty_factor * total_violation

    def _repair_infeasible_positions(self, *args, **kwargs):
        # Deprecated in favor of shared helper. Keep signature for backward compatibility if referenced elsewhere.
        positions, A, b, lb, ub = args[:5]
        repaired = repair_against_linear_inequalities(positions, A, b, lb, ub)
        return repaired

    def _reflect_and_clip_positions(
        self,
        new_positions: npt.NDArray[np.float64],
        lb: npt.NDArray[np.float64],
        ub: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Applies reflection at boundaries and ensures particles remain within bounds."""
        clipped, out_of_bounds = reflect_and_clip(new_positions, lb, ub)
        new_positions = clipped

        # Reverse velocities where reflection occurred when state is initialized
        if self._state:
            self._state.velocities[out_of_bounds] *= -1

        return new_positions
