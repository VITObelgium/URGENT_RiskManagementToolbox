from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from services.well_management_service.core.models import WellDesignServiceResponse


class DomainServiceInterface(ABC):
    """Plugin interface for converting well-design requests into simulation inputs.

    The canonical entry point mirrors ``WellDesignService.process_request``:
    plugins receive a request dictionary with a ``models`` list and return a
    ``WellDesignServiceResponse``. The optimization framework still handles the
    initial state, boundaries, and task packaging; the plugin owns the domain
    translation from well-design models to simulation-ready geometry.
    """

    ServiceName: str

    @abstractmethod
    def process_request(
        self, request_dict: dict[str, Any]
    ) -> WellDesignServiceResponse:
        """Convert a well-design request to a simulation-ready response.

        Called once per optimization candidate with a request dictionary shaped
        like ``{"models": [...]}``.
        """
        ...
