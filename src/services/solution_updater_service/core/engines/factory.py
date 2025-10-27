from common import OptimizationStrategy
from services.solution_updater_service.core.engines.common import (
    OptimizationEngineInterface,
)
from services.solution_updater_service.core.engines.pso import PSOEngine
from services.solution_updater_service.core.models.user import OptimizationEngine


class OptimizationEngineFactory:
    @staticmethod
    def get_engine(
        engine: OptimizationEngine,
        strategy: OptimizationStrategy = OptimizationStrategy.MINIMIZE,
    ) -> OptimizationEngineInterface:
        match engine:
            case OptimizationEngine.PSO:
                return PSOEngine(strategy)
            case _:
                raise NotImplementedError()
