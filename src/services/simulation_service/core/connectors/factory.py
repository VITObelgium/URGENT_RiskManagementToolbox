"""
NOTE:
This module must be aligned with python 3.10 syntax, as open-darts whl requires it.
"""

from enum import Enum

from .common import ConnectorInterface
from .open_darts import OpenDartsConnector


class Simulator(Enum):
    # order must be the same as in .proto
    UNDEFINED = 0
    OPENDARTS = 1


class ConnectorFactory:
    @staticmethod
    def get_connector(simulator: Simulator | int) -> ConnectorInterface:
        match simulator:
            case Simulator.OPENDARTS | Simulator.OPENDARTS.value:
                return OpenDartsConnector()
            case _:
                raise NotImplementedError()
