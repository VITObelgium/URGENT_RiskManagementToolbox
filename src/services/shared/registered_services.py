from enum import StrEnum
from typing import Any


class ServiceType(StrEnum):
    WellDesignService = "well_design"


type ServiceRequest = list[
    dict[str, Any]
]  # generic type which should be compiled with service(s) request
