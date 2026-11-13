"""Tests for the two-phase config validation and run-plugin bootstrap."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestration.risk_management_service.core.service import plugin_bootstrap
from orchestration.risk_management_service.core.service.plugin_bootstrap import (
    ResolvedRunPlugins,
    bootstrap_run_plugins,
    validate_problem_definition,
)
from services.problem_dispatcher_service.core.models import (
    DomainServiceItem,
    ProblemDispatcherDefinition,
)
from services.problem_dispatcher_service.core.service.interface import (
    DomainServiceInterface,
)
from urgent_plugins import PluginTraceError, resolve_domain_service_plugins

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture
def multi_domain_raw_config() -> dict:
    return json.loads((_EXAMPLES_DIR / "multi_domain_model.json").read_text())


def test_validate_problem_definition_multi_domain_passes(
    multi_domain_raw_config,
) -> None:
    definition = validate_problem_definition(multi_domain_raw_config)

    assert isinstance(definition, ProblemDispatcherDefinition)
    assert set(definition.domain_services) == {"well_design", "well_counter"}
    assert definition.plugins.domain_services == ["well_design", "well_counter"]

    # Item states were validated and normalised through the plugin adapters:
    # the item name is injected and well_counter's count is coerced to float.
    counter_item = definition.domain_services["well_counter"][0]
    assert counter_item.initial_state == {"name": "field", "count": 2.0}
    for item in definition.domain_services["well_design"]:
        assert item.initial_state["name"] == item.name


def test_validate_problem_definition_missing_plugins_section_raises(
    multi_domain_raw_config,
) -> None:
    raw = copy.deepcopy(multi_domain_raw_config)
    del raw["plugins"]

    with pytest.raises(ValueError, match="plugins"):
        validate_problem_definition(raw)


def test_validate_problem_definition_section_plugin_mismatch_raises(
    multi_domain_raw_config,
) -> None:
    raw = copy.deepcopy(multi_domain_raw_config)
    del raw["domain_services"]["well_counter"]

    with pytest.raises(ValueError, match="must match plugins.domain_services"):
        validate_problem_definition(raw)


def test_validate_problem_definition_invalid_initial_state_raises(
    multi_domain_raw_config,
) -> None:
    raw = copy.deepcopy(multi_domain_raw_config)
    raw["domain_services"]["well_counter"][0]["initial_state"]["count"] = -1

    with pytest.raises(ValueError, match=r"Item 'field'.*'well_counter'"):
        validate_problem_definition(raw)


def test_validate_problem_definition_trace_mismatch_raises(
    multi_domain_raw_config,
) -> None:
    # opendarts requires the 'well_design' trace; well_counter alone can't satisfy it.
    raw = copy.deepcopy(multi_domain_raw_config)
    raw["plugins"]["domain_services"] = ["well_counter"]
    raw["domain_services"] = {"well_counter": raw["domain_services"]["well_counter"]}
    del raw["optimization_parameters"]["linear_inequalities"]

    with pytest.raises(PluginTraceError, match="well_design"):
        validate_problem_definition(raw)


def test_bootstrap_run_plugins_resolves_all_domain_services(
    multi_domain_raw_config,
) -> None:
    definition = validate_problem_definition(multi_domain_raw_config)

    run_plugins = bootstrap_run_plugins(definition)

    assert isinstance(run_plugins, ResolvedRunPlugins)
    assert run_plugins.connector_name == "opendarts"
    assert run_plugins.optimizer_name == "pso"
    assert set(run_plugins.domain_services) == {"well_design", "well_counter"}
    for service in run_plugins.domain_services.values():
        assert isinstance(service, DomainServiceInterface)


def test_validate_item_states_happy_path_normalizes_state() -> None:
    resolve_domain_service_plugins(["well_counter"])
    item = DomainServiceItem(name="F1", initial_state={"count": 3})
    problem_definition = SimpleNamespace(domain_services={"well_counter": [item]})

    plugin_bootstrap._validate_item_states(problem_definition, "well_counter")

    assert item.initial_state == {"name": "F1", "count": 3.0}


def test_validate_item_states_invalid_state_mentions_item_and_plugin() -> None:
    resolve_domain_service_plugins(["well_counter"])
    item = DomainServiceItem(name="F1", initial_state={"count": "not_a_number"})
    problem_definition = SimpleNamespace(domain_services={"well_counter": [item]})

    with pytest.raises(ValueError, match=r"Item 'F1'.*'well_counter'"):
        plugin_bootstrap._validate_item_states(problem_definition, "well_counter")
