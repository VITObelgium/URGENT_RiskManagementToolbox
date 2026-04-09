from __future__ import annotations

from enum import StrEnum
from typing import Any


class ServiceType(StrEnum):
    WellDesignService = "well_design"


type ServiceRequest = list[dict[str, Any]]
