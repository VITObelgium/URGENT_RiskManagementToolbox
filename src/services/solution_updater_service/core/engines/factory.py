from __future__ import annotations

import inspect

from logger import get_logger
from services.solution_updater_service.core.engines.common import (
    OptimizationEngineInterface,
)

from urgent_plugins import (
    OptimizerPlugin,
    PluginKind,
    get_registry,
    normalize_plugin_name,
)
from urgent_plugins.loader import load_local_plugin
from urgent_plugins.paths import default_plugins_root

logger = get_logger(__name__)


class OptimizationEngineFactory:
    @staticmethod
    def get_engine(
        engine_name: str,
        seed: int | None = None,
    ) -> OptimizationEngineInterface:
        """Resolve an optimization engine by plugin name.

        Consults the :mod:`urgent_plugins` registry and, when the explicitly
        named plugin has not been loaded in this process yet, loads it from the
        configured plugin root. When the configured engine class does not accept
        a ``seed`` keyword argument, the engine is instantiated without it (a
        warning is logged if a seed was requested).
        """

        if not engine_name or not engine_name.strip():
            raise ValueError(
                "engine_name must be specified explicitly in the plugins config."
            )
        name = normalize_plugin_name(engine_name)

        descriptor = get_registry().get(PluginKind.OPTIMIZER, name)
        if descriptor is None:
            descriptor = load_local_plugin(
                PluginKind.OPTIMIZER, name, default_plugins_root()
            )
        if isinstance(descriptor, OptimizerPlugin):
            return OptimizationEngineFactory._instantiate(
                descriptor.implementation, seed
            )
        raise NotImplementedError(
            f"Unknown optimizer {engine_name!r}; not registered in urgent_plugins."
        )

    @staticmethod
    def _instantiate(
        engine_cls: type[OptimizationEngineInterface], seed: int | None
    ) -> OptimizationEngineInterface:
        # Decide from the signature instead of catching TypeError, which would
        # also swallow genuine errors raised inside the constructor.
        params = inspect.signature(engine_cls).parameters.values()
        accepts_seed = any(
            p.name == "seed" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params
        )
        if accepts_seed:
            return engine_cls(seed=seed)  # type: ignore[call-arg]
        if seed is not None:
            logger.warning(
                "Optimizer %s does not accept a 'seed' argument; seed=%s is ignored "
                "and the run will not be reproducible.",
                engine_cls.__name__,
                seed,
            )
        return engine_cls()
