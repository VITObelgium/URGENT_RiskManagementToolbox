from __future__ import annotations

from typing import Any

import services.well_management_service.core.well_templates as well_templates
from logger import get_logger
from services.well_management_service.core.models import (
    HWellModel,
    IWellModel,
    JWellModel,
    SimulationWellModel,
    SWellModel,
    Well,
    WellDesignServiceRequest,
    WellDesignServiceResponse,
)


class WellDesignService:
    _logger = get_logger(__name__)

    @staticmethod
    def process_request(request_dict: dict[str, Any]) -> WellDesignServiceResponse:
        # Parse the incoming request
        request = WellDesignServiceRequest(**request_dict)
        WellDesignService._logger.debug(
            "Parsed request into WellManagementServiceRequest: %s", request
        )

        # Build wells
        response = WellDesignService._build_wells(request)
        WellDesignService._logger.debug(
            "Built WellManagementServiceResponse: %s", response
        )

        return response

    @staticmethod
    def _build_wells(
            config: WellDesignServiceRequest,
    ) -> WellDesignServiceResponse:
        WellDesignService._logger.debug("Building wells from configuration: %s", config)

        wells: list[Well] = []

        for model in config.models:
            if isinstance(model, IWellModel):
                WellDesignService._logger.debug("Processing IWellModel: %s", model)
                wells.append(well_templates.IWellTemplate.from_model(model).build())
            elif isinstance(model, JWellModel):
                WellDesignService._logger.debug("Processing JWellModel: %s", model)
                wells.append(well_templates.JWellTemplate.from_model(model).build())
            elif isinstance(model, SWellModel):
                WellDesignService._logger.debug("Processing SWellModel: %s", model)
                wells.append(well_templates.SWellTemplate.from_model(model).build())
            elif isinstance(model, HWellModel):
                WellDesignService._logger.debug("Processing HWellModel: %s", model)
                wells.append(well_templates.HWellTemplate.from_model(model).build())
            else:
                WellDesignService._logger.warning(
                    "Unrecognized model type: %s", type(model)
                )

        response = WellDesignServiceResponse(
            wells=[SimulationWellModel.from_well(w) for w in wells]
        )
        WellDesignService._logger.debug(
            "Completed well building with response: %s", response
        )
        return response
