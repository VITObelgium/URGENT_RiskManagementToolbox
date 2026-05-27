from __future__ import annotations

from services.solution_updater_service.core.engines.common import (
    OptimizationEngineInterface,
)

from urgent_plugins import (
    OptimizerPlugin,
    PluginKind,
    get_registry,
)
from urgent_plugins.loader import load_local_plugin
from urgent_plugins.paths import default_plugins_root


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
        a ``seed`` keyword argument, the engine is instantiated without it.
        """

        name = engine_name.strip().lower()
        if not name:
            raise ValueError(
                "engine_name must be specified explicitly in the plugins config."
            )

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
        try:
            return engine_cls(seed=seed)  # type: ignore[call-arg]
        except TypeError:
            return engine_cls()
