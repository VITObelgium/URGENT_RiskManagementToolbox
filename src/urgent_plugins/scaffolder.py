from __future__ import annotations

import ast
import copy
import inspect
import re
import types
from dataclasses import dataclass
from pathlib import Path

from urgent_plugins.api import PluginKind, normalize_plugin_name
from urgent_plugins.paths import default_plugins_root

_ABSTRACTMETHOD_DECORATOR_NAMES = {"abstractmethod", "abc.abstractmethod"}


@dataclass(frozen=True)
class PluginKindSpec:
    """Describes one plugin kind for the scaffolder.

    Attributes:
        kind: The :class:`PluginKind` member identifying the plugin category.
        cli_alias: User-facing identifier accepted on the CLI (e.g. ``connector``).
        interface_module: Dotted module path containing the interface ABC.
        interface_class_name: Name of the interface ABC inside that module.
        descriptor_class_name: Name of the corresponding plugin descriptor.
        name_attr: Class attribute holding the plugin name on the user class.
    """

    kind: PluginKind
    cli_alias: str
    interface_module: str
    interface_class_name: str
    descriptor_class_name: str
    name_attr: str


_PLUGIN_KIND_SPECS: tuple[PluginKindSpec, ...] = (
    PluginKindSpec(
        kind=PluginKind.CONNECTOR,
        cli_alias="connector",
        interface_module="services.simulation_service.core.connectors.runner",
        interface_class_name="SubprocessConnectorInterface",
        descriptor_class_name="ConnectorPlugin",
        name_attr="ConnectorName",
    ),
    PluginKindSpec(
        kind=PluginKind.OPTIMIZER,
        cli_alias="optimizer",
        interface_module="services.solution_updater_service.core.engines.common",
        interface_class_name="OptimizationEngineInterface",
        descriptor_class_name="OptimizerPlugin",
        name_attr="EngineName",
    ),
    PluginKindSpec(
        kind=PluginKind.DOMAIN_SERVICE,
        cli_alias="domain",
        interface_module="services.problem_dispatcher_service.core.service.interface",
        interface_class_name="DomainServiceInterface",
        descriptor_class_name="DomainServicePlugin",
        name_attr="ServiceName",
    ),
)


def get_kind_specs() -> tuple[PluginKindSpec, ...]:
    return _PLUGIN_KIND_SPECS


def find_kind_spec_by_alias(alias: str) -> PluginKindSpec:
    normalized = alias.strip().lower()
    for spec in _PLUGIN_KIND_SPECS:
        if spec.cli_alias == normalized:
            return spec
    valid = ", ".join(s.cli_alias for s in _PLUGIN_KIND_SPECS)
    raise ValueError(f"Unknown plugin kind alias {alias!r}; valid: {valid}")


_CAMEL_TO_SNAKE_RE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_TO_SNAKE_RE_2 = re.compile(r"([a-z0-9])([A-Z])")


def to_snake_case(class_name: str) -> str:
    """Convert a CamelCase class name to a snake_case plugin identifier."""
    if not class_name:
        raise ValueError("class_name must be non-empty")
    intermediate = _CAMEL_TO_SNAKE_RE_1.sub(r"\1_\2", class_name)
    return _CAMEL_TO_SNAKE_RE_2.sub(r"\1_\2", intermediate).lower()


def _import_interface(spec: PluginKindSpec) -> type:
    module = __import__(spec.interface_module, fromlist=[spec.interface_class_name])
    return getattr(module, spec.interface_class_name)


def _is_abstractmethod_decorator(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _ABSTRACTMETHOD_DECORATOR_NAMES
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and f"{node.value.id}.{node.attr}" in _ABSTRACTMETHOD_DECORATOR_NAMES
        )
    return False


def _strip_abstract_decorators(decorators: list[ast.expr]) -> list[ast.expr]:
    return [d for d in decorators if not _is_abstractmethod_decorator(d)]


def _extract_docstring_node(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.stmt | None:
    if (
        method.body
        and isinstance(method.body[0], ast.Expr)
        and isinstance(method.body[0].value, ast.Constant)
        and isinstance(method.body[0].value.value, str)
    ):
        return method.body[0]
    return None


def _stub_body(method: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    docstring = _extract_docstring_node(method)
    raise_stmt = ast.Raise(
        exc=ast.Call(
            func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
            args=[],
            keywords=[],
        ),
        cause=None,
    )
    body: list[ast.stmt] = (
        [raise_stmt] if docstring is None else [docstring, raise_stmt]
    )
    for stmt in body:
        ast.fix_missing_locations(stmt)
    return body


def _build_method_stubs(interface_cls: type) -> list[ast.stmt]:
    """Return stubs for every abstract method declared on ``interface_cls``."""
    source = inspect.getsource(inspect.getmodule(interface_cls) or interface_cls)
    module_ast = ast.parse(source)

    class_def = next(
        (
            node
            for node in module_ast.body
            if isinstance(node, ast.ClassDef) and node.name == interface_cls.__name__
        ),
        None,
    )
    if class_def is None:
        raise RuntimeError(
            f"Could not locate class {interface_cls.__name__!r} in its own module AST"
        )

    target_names = frozenset(getattr(interface_cls, "__abstractmethods__", frozenset()))

    stubs: list[ast.stmt] = []
    for node in class_def.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in target_names:
            continue
        stub = copy.deepcopy(node)
        stub.body = _stub_body(node)
        stub.decorator_list = _strip_abstract_decorators(stub.decorator_list)
        ast.fix_missing_locations(stub)
        stubs.append(stub)

    return stubs


def _owned_public_names(module: types.ModuleType) -> list[str]:
    """Return public names *defined in* ``module`` (not re-imports).

    Honours ``__all__`` when present; otherwise selects names whose value
    reports ``__module__`` equal to the interface module's name. This excludes
    incidental re-exports like ``ABC``, ``abstractmethod``, or other modules
    imported at the top of the interface file.
    """
    explicit = getattr(module, "__all__", None)
    if explicit is not None:
        return sorted({str(name) for name in explicit})

    module_name = getattr(module, "__name__", None)
    owned: list[str] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        value = getattr(module, name, None)
        if getattr(value, "__module__", None) == module_name:
            owned.append(name)
    return sorted(owned)


def _collect_interface_top_level_imports(module: types.ModuleType) -> list[ast.stmt]:
    """Return the interface module's top-level ``Import``/``ImportFrom`` nodes.

    These cover external dependencies referenced in abstract method
    annotations (e.g. ``threading``, ``numpy``) that need to be available
    inside the generated plugin file.
    """
    source = inspect.getsource(module)
    module_ast = ast.parse(source)
    imports: list[ast.stmt] = []
    for node in module_ast.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
    return imports


def _build_class_def(
    class_name: str,
    interface_cls: type,
    name_attr: str,
    plugin_name: str,
    method_stubs: list[ast.stmt],
) -> ast.ClassDef:
    name_assign = ast.AnnAssign(
        target=ast.Name(id=name_attr, ctx=ast.Store()),
        annotation=ast.Name(id="str", ctx=ast.Load()),
        value=ast.Constant(value=plugin_name),
        simple=1,
    )
    body: list[ast.stmt] = [name_assign, *method_stubs]
    class_def = ast.ClassDef(
        name=class_name,
        bases=[ast.Name(id=interface_cls.__name__, ctx=ast.Load())],
        keywords=[],
        body=body,
        decorator_list=[],
        type_params=[],
    )
    ast.fix_missing_locations(class_def)
    return class_def


def _build_descriptor_assignment(
    class_name: str,
    descriptor_class_name: str,
    name_attr: str,
) -> ast.Assign:
    keywords: list[ast.keyword] = [
        ast.keyword(
            arg="name",
            value=ast.Attribute(
                value=ast.Name(id=class_name, ctx=ast.Load()),
                attr=name_attr,
                ctx=ast.Load(),
            ),
        ),
        ast.keyword(
            arg="implementation",
            value=ast.Name(id=class_name, ctx=ast.Load()),
        ),
    ]
    assign = ast.Assign(
        targets=[ast.Name(id="plugin", ctx=ast.Store())],
        value=ast.Call(
            func=ast.Name(id=descriptor_class_name, ctx=ast.Load()),
            args=[],
            keywords=keywords,
        ),
    )
    ast.fix_missing_locations(assign)
    return assign


def generate_plugin_source(
    spec: PluginKindSpec,
    class_name: str,
    plugin_name: str | None = None,
) -> str:
    """Generate the full plugin file source for ``class_name``.

    By default the plugin's registered name is ``to_snake_case(class_name)``.
    ``plugin_name`` may be supplied explicitly to preserve a legacy identifier
    (e.g. when migrating an existing implementation into the plugin layout).

    The output is validated with :func:`ast.parse` before being returned so a
    caller never writes a syntactically invalid stub.
    """
    if not class_name or not class_name[0].isalpha() or not class_name.isidentifier():
        raise ValueError(
            f"class_name must be a valid Python identifier starting with a letter; got {class_name!r}"
        )

    interface_cls = _import_interface(spec)
    plugin_name = (
        normalize_plugin_name(plugin_name) if plugin_name else to_snake_case(class_name)
    )

    method_stubs = _build_method_stubs(interface_cls)
    if not method_stubs:
        raise RuntimeError(
            f"Interface {spec.interface_class_name!r} has no abstract methods to scaffold."
        )

    future_import = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations", asname=None)], level=0
    )
    interface_module = inspect.getmodule(interface_cls)
    if interface_module is None:
        raise RuntimeError(
            f"Could not locate the source module of {spec.interface_class_name!r}"
        )
    interface_dep_imports = _collect_interface_top_level_imports(interface_module)

    interface_owned_names = _owned_public_names(interface_module)
    if spec.interface_class_name not in interface_owned_names:
        interface_owned_names.append(spec.interface_class_name)
    interface_import = ast.ImportFrom(
        module=spec.interface_module,
        names=[ast.alias(name=n, asname=None) for n in interface_owned_names],
        level=0,
    )
    descriptor_import = ast.ImportFrom(
        module="urgent_plugins",
        names=[ast.alias(name=spec.descriptor_class_name, asname=None)],
        level=0,
    )

    class_def = _build_class_def(
        class_name=class_name,
        interface_cls=interface_cls,
        name_attr=spec.name_attr,
        plugin_name=plugin_name,
        method_stubs=method_stubs,
    )
    descriptor_assign = _build_descriptor_assignment(
        class_name=class_name,
        descriptor_class_name=spec.descriptor_class_name,
        name_attr=spec.name_attr,
    )

    docstring_node = ast.Expr(
        value=ast.Constant(
            value=(
                f"Auto-generated {spec.kind.value} plugin stub for {class_name}. "
                "Implement the abstract methods below."
            )
        )
    )

    body: list[ast.stmt] = [
        docstring_node,
        future_import,
        *interface_dep_imports,
        interface_import,
        descriptor_import,
        class_def,
        descriptor_assign,
    ]

    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)

    source = ast.unparse(module) + "\n"

    ast.parse(source)
    return source


def write_plugin(
    spec: PluginKindSpec,
    class_name: str,
    plugins_root: Path | None = None,
    *,
    plugin_name: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Generate the plugin source and write it under ``plugins_root``.

    ``plugin_name`` overrides the registered name (defaults to snake_case of
    ``class_name``). Returns the path of the generated file.
    Raises :class:`FileExistsError` when the target already exists and
    ``overwrite`` is False.
    """
    root = plugins_root or default_plugins_root()
    resolved_plugin_name = (
        normalize_plugin_name(plugin_name) if plugin_name else to_snake_case(class_name)
    )
    target_dir = root / spec.kind.directory
    target = target_dir / f"{resolved_plugin_name}.py"
    if target.exists() and not overwrite:
        raise FileExistsError(f"Plugin file already exists: {target}")

    target_dir.mkdir(parents=True, exist_ok=True)
    source = generate_plugin_source(spec, class_name, plugin_name=resolved_plugin_name)
    target.write_text(source)
    return target
