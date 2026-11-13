from __future__ import annotations

import pytest

from services.problem_dispatcher_service.core.models import (
    PluginConfig,
    ProblemDispatcherDefinition,
)

from ._builtin_classes import (
    get_builtin_connector_class,
    get_builtin_optimizer_class,
)


def _minimal_problem_definition_payload() -> dict:
    return {
        "run_mode": "evaluation",
        "domain_services": {
            "well_design": [
                {
                    "name": "W1",
                    "initial_state": {
                        "well_type": "IWell",
                        "wellhead": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "md": 1.0,
                    },
                }
            ]
        },
        "plugins": {
            "connector": get_builtin_connector_class().ConnectorName,  # type: ignore[attr-defined]
            "optimizer": get_builtin_optimizer_class().EngineName,  # type: ignore[attr-defined]
            "domain_services": ["well_design"],
        },
    }


def test_plugins_section_required() -> None:
    payload = _minimal_problem_definition_payload()
    del payload["plugins"]
    with pytest.raises(ValueError):
        ProblemDispatcherDefinition(**payload)


def test_plugins_section_stores_provided_names() -> None:
    pd = ProblemDispatcherDefinition(**_minimal_problem_definition_payload())
    assert isinstance(pd.plugins, PluginConfig)
    assert pd.plugins.connector == get_builtin_connector_class().ConnectorName  # type: ignore[attr-defined]
    assert pd.plugins.optimizer == get_builtin_optimizer_class().EngineName  # type: ignore[attr-defined]
    assert pd.plugins.domain_services == ["well_design"]


def test_plugins_section_accepts_custom_connector_and_optimizer() -> None:
    payload = _minimal_problem_definition_payload()
    payload["plugins"] = {
        "connector": "eclipse",
        "optimizer": "genetic",
        "domain_services": ["custom"],
    }
    payload["domain_services"] = {
        "custom": payload["domain_services"]["well_design"],
    }
    pd = ProblemDispatcherDefinition(**payload)
    assert pd.plugins.connector == "eclipse"
    assert pd.plugins.optimizer == "genetic"
    assert pd.plugins.domain_services == ["custom"]


def test_plugins_section_rejects_unknown_keys() -> None:
    payload = _minimal_problem_definition_payload()
    payload["plugins"] = {"unknown_field": "foo"}
    with pytest.raises(ValueError):
        ProblemDispatcherDefinition(**payload)


def test_plugins_section_rejects_empty_domain_services() -> None:
    payload = _minimal_problem_definition_payload()
    payload["plugins"]["domain_services"] = []
    with pytest.raises(ValueError):
        ProblemDispatcherDefinition(**payload)


def test_plugins_section_rejects_duplicate_domain_services() -> None:
    payload = _minimal_problem_definition_payload()
    payload["plugins"]["domain_services"] = ["well_design", "well_design"]
    with pytest.raises(ValueError, match="unique"):
        ProblemDispatcherDefinition(**payload)


def test_domain_services_sections_must_match_plugins() -> None:
    payload = _minimal_problem_definition_payload()
    payload["plugins"]["domain_services"] = ["well_design", "well_counter"]
    with pytest.raises(ValueError, match="must match plugins.domain_services"):
        ProblemDispatcherDefinition(**payload)


def test_plugin_names_are_normalised() -> None:
    payload = _minimal_problem_definition_payload()
    payload["plugins"] = {
        "connector": "OpenDarts",
        "optimizer": "PSO",
        "domain_services": ["Well-Design"],
    }
    payload["domain_services"] = {
        "Well_Design": payload["domain_services"]["well_design"]
    }

    pd = ProblemDispatcherDefinition(**payload)

    assert pd.plugins.connector == "opendarts"
    assert pd.plugins.optimizer == "pso"
    assert pd.plugins.domain_services == ["well_design"]
    assert list(pd.domain_services) == ["well_design"]


def test_domain_service_names_duplicate_after_normalisation_rejected() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        PluginConfig(
            connector="opendarts",
            optimizer="pso",
            domain_services=["well_design", "Well-Design"],
        )


def test_domain_service_sections_duplicate_after_normalisation_rejected() -> None:
    payload = _minimal_problem_definition_payload()
    items = payload["domain_services"]["well_design"]
    payload["domain_services"] = {"well_design": items, "Well-Design": items}

    with pytest.raises(ValueError, match="after name normalisation"):
        ProblemDispatcherDefinition(**payload)


def test_invalid_plugin_name_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid plugin name"):
        PluginConfig(
            connector="open darts", optimizer="pso", domain_services=["well_design"]
        )
