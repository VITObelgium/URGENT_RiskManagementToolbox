from abc import ABC, abstractmethod
from typing import MutableMapping

from services.well_management_service.core.models import PerforationRange, Trajectory


class SectionInterface(ABC):
    """
    md: float - measured length of the segment
    user_inclination: float - override the inclination calculated from trajectory
    """

    def __init__(self, md: float, user_inclination: float | None = None):
        self._md: float = md
        self._user_inclination: float | None = user_inclination

    @abstractmethod
    def append_to_trajectory(
        self,
        trajectory: Trajectory,
        md_step: float,
        perforations: MutableMapping[str, PerforationRange] | None = None,
    ) -> Trajectory:
        """
        Append section partial trajectory to well trajectory
        Parameters
        ----------
        trajectory
        md_step
        perforations

        Returns
        -------
        Trajectory

        """
