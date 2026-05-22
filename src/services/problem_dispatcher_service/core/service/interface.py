from __future__ import annotations

from abc import ABC, abstractmethod

from services.well_management_service.core.models import WellDesignServiceResponse
from services.well_management_service.core.models.user import WellModel


class DomainServiceInterface(ABC):
    """Plugin interface for converting parsed well models into simulation inputs.

    Receives a list of typed well models and returns a ``WellDesignServiceResponse``.
    The three PSO-formulation concerns (initial state, boundaries, task packaging)
    are handled by the framework — the plugin is responsible only for geometry
    construction, which is the only part that genuinely varies per well model type.
    """

    ServiceName: str

    @abstractmethod
    def build(self, wells: list[WellModel]) -> WellDesignServiceResponse:
        """Convert typed well models to a simulation-ready response.

        Called once per optimization candidate.  Receives fully-parsed,
        validated ``WellModel`` objects so the implementation never needs to
        re-validate geometry; its only responsibility is to produce the
        ``WellDesignServiceResponse`` that the simulation service consumes.
        """
        ...
