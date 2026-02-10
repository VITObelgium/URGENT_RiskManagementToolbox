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

logger = get_logger(__name__)


class WellDesignService:
    @staticmethod
    def process_request(request_dict: dict[str, Any]) -> WellDesignServiceResponse:
        # Parse the incoming request
        request = WellDesignServiceRequest(**request_dict)
        logger.debug("Parsed request into WellManagementServiceRequest: %s", request)

        # Build wells
        response = WellDesignService._build_wells(request)
        logger.debug("Built WellManagementServiceResponse: %s", response)

        return response

    @staticmethod
    def _build_wells(
        config: WellDesignServiceRequest,
    ) -> WellDesignServiceResponse:
        logger.debug("Building wells from configuration: %s", config)

        wells: list[Well] = []

        for model in config.models:
            match model:
                case IWellModel():
                    logger.debug("Processing IWellModel: %s", model)
                    wells.append(well_templates.IWellTemplate.from_model(model).build())
                case JWellModel():
                    logger.debug("Processing JWellModel: %s", model)
                    wells.append(well_templates.JWellTemplate.from_model(model).build())
                case SWellModel():
                    logger.debug("Processing SWellModel: %s", model)
                    wells.append(well_templates.SWellTemplate.from_model(model).build())
                case HWellModel():
                    logger.debug("Processing HWellModel: %s", model)
                    wells.append(well_templates.HWellTemplate.from_model(model).build())
                case _:
                    logger.warning("Unrecognized model type: %s", type(model))

        response = WellDesignServiceResponse(
            wells=[SimulationWellModel.from_well(w) for w in wells]
        )
        logger.debug("Completed well building with response: %s", response)
        return response
