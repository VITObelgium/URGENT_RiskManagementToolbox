from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from common import OptimizationStrategy


class OptimizationEngineInterface(ABC):
    @abstractmethod
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
    ) -> npt.NDArray[np.float64]: ...

    @property
    @abstractmethod
    def global_best_result(self) -> float | npt.NDArray[np.float64]: ...

    @property
    @abstractmethod
    def global_best_control_vector(self) -> npt.NDArray[np.float64]: ...

    @abstractmethod
    def get_checkpoint_state(self) -> dict[str, Any]: ...

    @abstractmethod
    def restore_checkpoint_state(self, state: dict[str, Any]) -> None: ...
