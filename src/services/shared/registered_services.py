from __future__ import annotations

from enum import StrEnum
from typing import Any

from typing_extensions import TypeAlias


class ServiceType(StrEnum):
    WellDesignService = "well_design"


ServiceRequest: TypeAlias = list[dict[str, Any]]
