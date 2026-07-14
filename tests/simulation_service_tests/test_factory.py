import pytest

from services.simulation_service.core.connectors.factory import ConnectorFactory
from plugins.connectors.opendarts import OpenDartsConnector


@pytest.mark.parametrize("connector_name", ["opendarts", "OPENDARTS"])
def test_get_connector_returns_opendarts(connector_name) -> None:
    assert isinstance(
        ConnectorFactory.get_connector(connector_name), OpenDartsConnector
    )


@pytest.mark.parametrize("connector_name", ["", "  "])
def test_get_connector_empty_name_raises(connector_name) -> None:
    with pytest.raises(ValueError):
        ConnectorFactory.get_connector(connector_name)


def test_get_connector_unknown_name_raises() -> None:
    with pytest.raises(FileNotFoundError):
        ConnectorFactory.get_connector("not_registered_plugin")
