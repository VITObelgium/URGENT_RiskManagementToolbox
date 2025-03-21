from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import services.well_management_service.core.well_templates as well_templates
from logger.u_logger import get_logger
from services.well_management_service.core.models import (
    HWellModel,
    IWellModel,
    JWellModel,
    SimulationWellModel,
    SWellModel,
    Well,
    WellManagementServiceRequest,
    WellManagementServiceResponse,
)


class WellManagementService:
    _logger = get_logger(__name__)

    @staticmethod
    def process_request(request_dict: dict[str, Any]) -> WellManagementServiceResponse:
        # Parse the incoming request
        request = WellManagementServiceRequest(**request_dict)
        WellManagementService._logger.debug(
            "Parsed request into WellManagementServiceRequest: %s", request
        )

        # Build wells
        response = WellManagementService._build_wells(request)
        WellManagementService._logger.debug(
            "Built WellManagementServiceResponse: %s", response
        )

        return response

    @staticmethod
    def _build_wells(
        config: WellManagementServiceRequest,
    ) -> WellManagementServiceResponse:
        WellManagementService._logger.debug(
            "Building wells from configuration: %s", config
        )

        wells: list[Well] = []

        for model in config.models:
            if isinstance(model, IWellModel):
                WellManagementService._logger.debug("Processing IWellModel: %s", model)
                wells.append(well_templates.IWellTemplate.from_model(model).build())
            elif isinstance(model, JWellModel):
                WellManagementService._logger.debug("Processing JWellModel: %s", model)
                wells.append(well_templates.JWellTemplate.from_model(model).build())
            elif isinstance(model, SWellModel):
                WellManagementService._logger.debug("Processing SWellModel: %s", model)
                wells.append(well_templates.SWellTemplate.from_model(model).build())
            elif isinstance(model, HWellModel):
                WellManagementService._logger.debug("Processing HWellModel: %s", model)
                wells.append(well_templates.HWellTemplate.from_model(model).build())
            else:
                WellManagementService._logger.warning(
                    "Unrecognized model type: %s", type(model)
                )

        response = WellManagementServiceResponse(
            wells=[SimulationWellModel.from_well(w) for w in wells]
        )
        WellManagementService._logger.debug(
            "Completed well building with response: %s", response
        )
        return response

    @staticmethod
    def dump_results_schema(path: Path | str) -> None:
        WellManagementService._logger.info("Dumping result schema to path: %s", path)

        with open(path, mode="w+") as fp:
            schema = WellManagementServiceResponse.model_json_schema()
            WellManagementService._logger.debug("Result schema: %s", schema)
            fp.write(json.dumps(obj=schema))

        WellManagementService._logger.info(
            "Result schema successfully written to: %s", path
        )
