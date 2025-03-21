from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class OptimizationEngineInterface(ABC):
    @abstractmethod
    def update_solution_to_next_iter(
        self,
        parameters: npt.NDArray[np.float64],
        results: npt.NDArray[np.float64],
        lb: npt.NDArray[np.float64],
        ub: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]: ...
