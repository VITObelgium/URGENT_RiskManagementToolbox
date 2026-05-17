from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from services.simulation_service.core.connectors.common import ConnectorInterface
from services.solution_updater_service.core.engines.common import (
    OptimizationEngineInterface,
)
from urgent_plugins import ConnectorPlugin, OptimizerPlugin, get_registry
from urgent_plugins.scaffolder import (
    find_kind_spec_by_alias,
    generate_plugin_source,
    to_snake_case,
    write_plugin,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    get_registry().clear()
    yield
    get_registry().clear()


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "input_name, expected",
    [
        ("Eclipse", "eclipse"),
        ("MyCoolConnector", "my_cool_connector"),
        ("HTTPRunner", "http_runner"),
        ("Genetic", "genetic"),
        ("XYZEngine", "xyz_engine"),
    ],
)
def test_to_snake_case(input_name: str, expected: str) -> None:
    assert to_snake_case(input_name) == expected


def test_find_kind_spec_by_alias_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        find_kind_spec_by_alias("nope")


def test_generate_connector_plugin_source_is_valid_python() -> None:
    spec = find_kind_spec_by_alias("simulation")
    source = generate_plugin_source(spec, "Eclipse")
    ast.parse(source)
    assert "class Eclipse(ConnectorInterface):" in source
    assert "ConnectorName: str = 'eclipse'" in source
    assert "plugin = ConnectorPlugin(name=Eclipse.ConnectorName" in source


def test_generate_optimizer_plugin_source_contains_all_abstract_methods() -> None:
    spec = find_kind_spec_by_alias("algorithm")
    source = generate_plugin_source(spec, "Genetic")
    ast.parse(source)

    for method in OptimizationEngineInterface.__abstractmethods__:
        assert f"def {method}" in source, f"missing stub for {method}"

    assert "class Genetic(OptimizationEngineInterface):" in source
    assert "EngineName: str = 'genetic'" in source
    assert "plugin = OptimizerPlugin(name=Genetic.EngineName" in source


def test_generate_connector_scaffold_omits_helpers_not_on_interface() -> None:
    spec = find_kind_spec_by_alias("simulation")
    source = generate_plugin_source(spec, "Eclipse")

    interface_methods = ConnectorInterface.__abstractmethods__
    assert interface_methods == {"run"}, (
        "Test premise out of date: ConnectorInterface abstract surface changed"
    )
    assert "def get_well_connection_cells" not in source
    assert "def broadcast_result" not in source


def test_write_plugin_creates_file_and_loads_back(tmp_path: Path) -> None:
    spec = find_kind_spec_by_alias("simulation")
    target = write_plugin(spec, "Eclipse", plugins_root=tmp_path)

    assert target == tmp_path / "connectors" / "eclipse.py"
    assert target.is_file()

    module = _load_module(target, "_scaffolded_eclipse_test")
    descriptor = getattr(module, "plugin")
    assert isinstance(descriptor, ConnectorPlugin)
    assert descriptor.name == "eclipse"
    assert descriptor.implementation.__name__ == "Eclipse"

    instance = descriptor.implementation()
    with pytest.raises(NotImplementedError):
        instance.run("/tmp/x.json", {})


def test_write_optimizer_plugin_creates_loadable_stub(tmp_path: Path) -> None:
    spec = find_kind_spec_by_alias("algorithm")
    target = write_plugin(spec, "Genetic", plugins_root=tmp_path)

    assert target == tmp_path / "optimizers" / "genetic.py"
    module = _load_module(target, "_scaffolded_genetic_test")
    descriptor = getattr(module, "plugin")
    assert isinstance(descriptor, OptimizerPlugin)
    assert descriptor.name == "genetic"
    assert descriptor.implementation.__name__ == "Genetic"


def test_write_plugin_refuses_to_overwrite_existing(tmp_path: Path) -> None:
    spec = find_kind_spec_by_alias("simulation")
    write_plugin(spec, "Eclipse", plugins_root=tmp_path)
    with pytest.raises(FileExistsError):
        write_plugin(spec, "Eclipse", plugins_root=tmp_path)


def test_write_plugin_overwrites_when_requested(tmp_path: Path) -> None:
    spec = find_kind_spec_by_alias("simulation")
    target = write_plugin(spec, "Eclipse", plugins_root=tmp_path)
    target.write_text("# tampered\n")
    write_plugin(spec, "Eclipse", plugins_root=tmp_path, overwrite=True)
    assert "tampered" not in target.read_text()
    assert "class Eclipse" in target.read_text()


def test_generate_rejects_invalid_class_name() -> None:
    spec = find_kind_spec_by_alias("simulation")
    with pytest.raises(ValueError):
        generate_plugin_source(spec, "1Bad")
    with pytest.raises(ValueError):
        generate_plugin_source(spec, "has space")
