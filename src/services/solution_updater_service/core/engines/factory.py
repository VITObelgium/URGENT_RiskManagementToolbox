from __future__ import annotations

from services.solution_updater_service.core.engines.common import (
    OptimizationEngineInterface,
)

from urgent_plugins import (
    OptimizerPlugin,
    PluginKind,
    get_registry,
    register_builtins,
)


class OptimizationEngineFactory:
    @staticmethod
    def get_engine(
        engine_name: str,
        seed: int | None = None,
    ) -> OptimizationEngineInterface:
        """Resolve an optimization engine by plugin name.

        Consults the :mod:`urgent_plugins` registry (which seeds built-ins
        through the same loader as third-party plugins). When the configured
        engine class does not accept a ``seed`` keyword argument, the engine is
        instantiated without it so third-party engines need not pretend to
        understand a parameter they do not use.
        ``engine_name`` must be supplied explicitly.
        """

        name = engine_name.strip().lower()
        if not name:
            raise ValueError(
                "engine_name must be specified explicitly in the plugins config."
            )

        register_builtins()
        descriptor = get_registry().get(PluginKind.OPTIMIZER, name)
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
