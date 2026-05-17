import pytest

from services.shared.type_checks import ensure_not_none
from services.solution_updater_service.core.engines.factory import (
    OptimizationEngineFactory,
)
from plugins.optimizers.pso import PSOEngine


class TestOptimizationEngineFactory:
    def test_get_pso_engine_returns_pso_instance(self):
        engine = OptimizationEngineFactory.get_engine(PSOEngine.EngineName)
        assert isinstance(engine, PSOEngine)

    def test_get_pso_engine_with_seed(self):
        engine = OptimizationEngineFactory.get_engine(PSOEngine.EngineName, seed=42)
        assert isinstance(engine, PSOEngine)

    def test_factory_raises_when_engine_name_empty(self):
        with pytest.raises(ValueError):
            OptimizationEngineFactory.get_engine("")

    def test_unsupported_engine_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            OptimizationEngineFactory.get_engine("not_a_registered_engine")


class TestEnsureNotNone:
    def test_returns_non_none_value(self):
        assert ensure_not_none("hello") == "hello"
        assert ensure_not_none(42) == 42
        assert ensure_not_none([1, 2, 3]) == [1, 2, 3]

    def test_raises_value_error_for_none(self):
        with pytest.raises(ValueError):
            ensure_not_none(None)

    def test_raises_with_custom_message(self):
        with pytest.raises(ValueError, match="custom error"):
            ensure_not_none(None, "custom error")

    def test_raises_with_default_message(self):
        with pytest.raises(ValueError, match="Expected a non-None object"):
            ensure_not_none(None)
