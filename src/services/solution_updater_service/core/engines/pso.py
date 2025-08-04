import numpy as np
from numpy import typing as npt

from services.solution_updater_service.core.engines.common import (
    OptimizationEngineInterface,
)
from services.solution_updater_service.core.utils.type_checks import ensure_not_none


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
        self, w: float = 0.8, c1: float = 1.6, c2: float = 1.6, seed: int | None = None
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

    @property
    def global_best_controll_vector(self) -> npt.NDArray[np.float64]:
        return ensure_not_none(self._state).global_best_position

    def update_solution_to_next_iter(
        self,
        parameters: npt.NDArray[np.float64],
        results: npt.NDArray[np.float64],
        lb: npt.NDArray[np.float64],
        ub: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Updates particles positions based on their velocities and applies reflection boundary."""
        if self._state is None:
            self._initialize_state_on_first_call(parameters, results)
        self._update_state_positions(parameters, results)
        new_velocities = self._calculate_new_velocity(parameters)
        self._update_state_velocities(new_velocities)
        new_positions = self._calculate_new_position(parameters, new_velocities)
        self._update_metrics(results)
        return self._reflect_and_clip_positions(new_positions, lb, ub)

    def _initialize_state_on_first_call(
        self, parameters: npt.NDArray[np.float64], results: npt.NDArray[np.float64]
    ) -> None:
        """Initializes the state on the first call, setting up best positions, velocities, and global best."""
        best_index = np.argmin(results)
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
        mask = results < state.particles_best_results

        state.particles_best_positions[mask[:, 0]] = positions[mask[:, 0]]
        state.particles_best_results[mask] = results[mask]

    def _update_global_best_position(
        self, positions: npt.NDArray[np.float64], results: npt.NDArray[np.float64]
    ) -> None:
        """Updates the global best position if a new best is found."""
        state: _PSOState = ensure_not_none(self._state)
        best_index = np.argmin(results)
        best_result = float(results[best_index].item())

        if best_result < state.global_best_result:
            state.global_best_position = positions[best_index]
            state.global_best_result = best_result

    def _update_state_velocities(self, new_velocity: npt.NDArray[np.float64]) -> None:
        """Updates the velocity of particles."""
        state: _PSOState = ensure_not_none(self._state)
        state.velocities = new_velocity

    @staticmethod
    def _calculate_new_position(
        old_position: npt.NDArray[np.float64], new_velocity: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Computes the new position of particles by adding velocity."""
        return (old_position + new_velocity).astype(np.float64)

    def _reflect_and_clip_positions(
        self,
        new_positions: npt.NDArray[np.float64],
        lb: npt.NDArray[np.float64],
        ub: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Applies reflection at boundaries and ensures particles remain within bounds."""
        state: _PSOState = ensure_not_none(self._state)

        # Find out-of-bounds positions
        out_of_bounds = (new_positions < lb) | (new_positions > ub)

        # Reflect positions
        reflected_positions = np.where(
            new_positions < lb, 2 * lb - new_positions, 2 * ub - new_positions
        )
        new_positions[out_of_bounds] = reflected_positions[out_of_bounds]

        # Reverse velocities where reflection occurred
        state.velocities[out_of_bounds] *= -1

        # Ensure positions are strictly within bounds (edge-case handling)
        return np.clip(new_positions, lb, ub)
