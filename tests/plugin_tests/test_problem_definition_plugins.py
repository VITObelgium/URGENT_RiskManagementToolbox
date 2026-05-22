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
        "well_design": [
            {
                "well_name": "W1",
                "initial_state": {
                    "well_type": "IWell",
                    "wellhead": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "md": 1.0,
                },
            }
        ],
        "plugins": {
            "connector": get_builtin_connector_class().ConnectorName,  # type: ignore[attr-defined]
            "optimizer": get_builtin_optimizer_class().EngineName,  # type: ignore[attr-defined]
            "domain_service": "builtin",
        },
    }


def test_plugins_section_required() -> None:
    payload = {
        "run_mode": "evaluation",
        "well_design": [
            {
                "well_name": "W1",
                "initial_state": {
                    "well_type": "IWell",
                    "wellhead": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "md": 1.0,
                },
            }
        ],
    }
    with pytest.raises(ValueError):
        ProblemDispatcherDefinition(**payload)


def test_plugins_section_stores_provided_names() -> None:
    pd = ProblemDispatcherDefinition(**_minimal_problem_definition_payload())
    assert isinstance(pd.plugins, PluginConfig)
    assert pd.plugins.connector == get_builtin_connector_class().ConnectorName  # type: ignore[attr-defined]
    assert pd.plugins.optimizer == get_builtin_optimizer_class().EngineName  # type: ignore[attr-defined]


def test_plugins_section_accepts_custom_connector_and_optimizer() -> None:
    payload = _minimal_problem_definition_payload()
    payload["plugins"] = {
        "connector": "eclipse",
        "optimizer": "genetic",
        "domain_service": "custom",
    }
    pd = ProblemDispatcherDefinition(**payload)
    assert pd.plugins.connector == "eclipse"
    assert pd.plugins.optimizer == "genetic"


def test_plugins_section_rejects_unknown_keys() -> None:
    payload = _minimal_problem_definition_payload()
    payload["plugins"] = {"unknown_field": "foo"}
    with pytest.raises(ValueError):
        ProblemDispatcherDefinition(**payload)
