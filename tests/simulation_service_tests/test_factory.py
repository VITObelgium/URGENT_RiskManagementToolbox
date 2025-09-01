import pytest

from services.simulation_service.core.connectors.factory import (
    ConnectorFactory,
    Simulator,
)
from services.simulation_service.core.connectors.open_darts import OpenDartsConnector


@pytest.mark.parametrize(
    "simulator, expected_result",
    [
        (Simulator.OPENDARTS, OpenDartsConnector),  # Valid case: OPENDARTS connector
        (Simulator.UNDEFINED, NotImplementedError),  # Invalid case
        (999, NotImplementedError),  # Invalid simulator (not in enum)
    ],
)
def test_get_connector(simulator, expected_result) -> None:  # type: ignore
    if issubclass(expected_result, Exception):
        # Handle cases where exception is expected
        with pytest.raises(expected_result):
            ConnectorFactory.get_connector(simulator)
    else:
        # Handle valid cases where a connector is expected
        connector = ConnectorFactory.get_connector(simulator)
        assert isinstance(connector, expected_result), (
            f"Expected {expected_result}, but got {type(connector)}"
        )
