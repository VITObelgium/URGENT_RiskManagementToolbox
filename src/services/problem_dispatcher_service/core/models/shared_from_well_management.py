# from __future__ import annotations
#
# import itertools
# from typing import Annotated, Literal, Sequence, Union
#
# import numpy as np
# from pydantic import BaseModel, Field, FiniteFloat, model_validator
#
#
# class PerforationRangeModel(BaseModel, extra="forbid"):
#     start_md: float = Field(ge=0.0)
#     end_md: float = Field(ge=0.0)
#
#     @model_validator(mode="after")
#     def validate_perforation_start_and_end(self) -> PerforationRangeModel:
#         if self.start_md >= self.end_md:
#             raise ValueError(
#                 f"Invalid range: start ({self.start_md}) must be less than end ({self.end_md})"
#             )
#         return self
#
#
# class PositionModel(BaseModel, extra="forbid"):
#     x: FiniteFloat
#     y: FiniteFloat
#     z: FiniteFloat
#
#
# class IWellModel(BaseModel, extra="forbid", str_strip_whitespace=True):
#     well_type: Literal["IWell"] = Field(default="IWell")
#     name: str
#     md: float = Field(gt=0.0)
#     wellhead: PositionModel
#     md_step: float = Field(ge=0.1, default=0.5)
#     perforations: list[PerforationRangeModel] | None = Field(
#         default=None
#     )  # simplification -> only one perforation
#
#     @model_validator(mode="after")
#     def sort_perforations(self) -> IWellModel:
#         if self.perforations:
#             self.perforations = sorted(self.perforations, key=lambda p: p.start_md)
#         return self
#
#     @model_validator(mode="after")
#     def check_perforations_not_overlap(self) -> IWellModel:
#         if not _are_perforations_non_overlapping(self.perforations):
#             raise ValueError("Perforations can't overlap")
#         return self
#
#     @model_validator(mode="after")
#     def validate_perforation_within_well_md(self) -> IWellModel:
#         if not _is_perforation_within_total_well_md(
#             (self.md,),
#             self.perforations,
#         ):
#             raise ValueError(
#                 "One or more perforation intervals extend beyond the total measured depth of the well."
#             )
#         return self
#
#     @model_validator(mode="after")
#     def perforate_whole_md_if_no_perforation(self):
#         if not self.perforations:
#             self.perforations = [PerforationRangeModel(start_md=0, end_md=self.md)]
#         return self
#
#     @model_validator(mode="after")
#     def accept_only_one_perforation(self):
#         if self.perforations and len(self.perforations) > 1:
#             raise ValueError("Only single perforation is supported")
#         return self
#
#
# class JwellModel(BaseModel, extra="forbid", str_strip_whitespace=True):
#     well_type: Literal["JWell"] = Field(default="JWell")
#     name: str
#     md_linear1: float = Field(gt=0.0)
#     md_curved: float = Field(gt=0.0)
#     dls: float = Field(gt=-45.00, le=45.00)
#     md_linear2: float = Field(gt=0.0)
#     wellhead: PositionModel
#     azimuth: float
#     md_step: float = Field(ge=0.1, default=0.5)
#     perforations: list[PerforationRangeModel] | None = Field(default=None)
#
#     @model_validator(mode="after")
#     def sort_perforations(self) -> JwellModel:
#         if self.perforations:
#             self.perforations = sorted(self.perforations, key=lambda p: p.start_md)
#         return self
#
#     @model_validator(mode="after")
#     def check_perforations_not_overlap(self) -> JwellModel:
#         if not _are_perforations_non_overlapping(self.perforations):
#             raise ValueError("Perforations can't overlap")
#         return self
#
#     @model_validator(mode="after")
#     def validate_perforation_within_well_md(self) -> JwellModel:
#         if not _is_perforation_within_total_well_md(
#             (self.md_linear1, self.md_curved, self.md_linear2),
#             self.perforations,
#         ):
#             raise ValueError(
#                 "One or more perforation intervals extend beyond the total measured depth of the well."
#             )
#         return self
#
#     @model_validator(mode="after")
#     def perforate_whole_md_if_no_perforation(self) -> JwellModel:
#         if not self.perforations:
#             total_md = self.md_linear1 + self.md_curved + self.md_linear2
#             self.perforations = [PerforationRangeModel(start_md=0, end_md=total_md)]
#         return self
#
#     @model_validator(mode="after")
#     def accept_only_one_perforation(self) -> JwellModel:
#         if self.perforations and len(self.perforations) > 1:
#             raise ValueError("Only single perforation is supported")
#         return self
#
#
# class SWellModel(BaseModel, extra="forbid", str_strip_whitespace=True):
#     well_type: Literal["SWell"] = Field(default="SWell")
#     name: str
#     md_linear1: float = Field(gt=0.0)
#     md_curved1: float = Field(gt=0.0)
#     dls1: float = Field(gt=-45.00, le=45.00)
#     md_linear2: float = Field(gt=0.0)
#     md_curved2: float = Field(gt=0.0)
#     dls2: float = Field(gt=-45.00, le=45.00)
#     md_linear3: float = Field(gt=0.0)
#     wellhead: PositionModel
#     azimuth: float
#     md_step: float = Field(ge=0.1, default=0.5)
#     perforations: list[PerforationRangeModel] | None = Field(default=None)
#
#     @model_validator(mode="after")
#     def sort_perforations(self) -> SWellModel:
#         if self.perforations:
#             self.perforations = sorted(self.perforations, key=lambda p: p.start_md)
#         return self
#
#     @model_validator(mode="after")
#     def check_perforations_not_overlap(self) -> SWellModel:
#         if not _are_perforations_non_overlapping(self.perforations):
#             raise ValueError("Perforations can't overlap")
#         return self
#
#     @model_validator(mode="after")
#     def validate_perforation_within_well_md(self) -> SWellModel:
#         if not _is_perforation_within_total_well_md(
#             (
#                 self.md_linear1,
#                 self.md_curved1,
#                 self.md_linear2,
#                 self.md_curved2,
#                 self.md_linear3,
#             ),
#             self.perforations,
#         ):
#             raise ValueError(
#                 "One or more perforation intervals extend beyond the total measured depth of the well."
#             )
#         return self
#
#     @model_validator(mode="after")
#     def perforate_whole_md_if_no_perforation(self) -> SWellModel:
#         if not self.perforations:
#             total_md = (
#                 self.md_linear1
#                 + self.md_curved1
#                 + self.md_linear2
#                 + self.md_curved2
#                 + self.md_linear3
#             )
#             self.perforations = [PerforationRangeModel(start_md=0, end_md=total_md)]
#         return self
#
#     @model_validator(mode="after")
#     def accept_only_one_perforation(self) -> SWellModel:
#         if self.perforations and len(self.perforations) > 1:
#             raise ValueError("Only single perforation is supported")
#         return self
#
#
# class HWellModel(BaseModel, extra="forbid", str_strip_whitespace=True):
#     well_type: Literal["HWell"] = Field(default="HWell")
#     name: str
#     TVD: float = Field(gt=0.0)
#     md_lateral: float = Field(gt=0.0)
#     wellhead: PositionModel
#     azimuth: float
#     md_step: float = Field(ge=0.1, default=0.5)
#     perforations: list[PerforationRangeModel] | None = Field(default=None)
#
#     @model_validator(mode="after")
#     def sort_perforations(self) -> HWellModel:
#         if self.perforations:
#             self.perforations = sorted(self.perforations, key=lambda p: p.start_md)
#         return self
#
#     @model_validator(mode="after")
#     def check_perforations_not_overlap(self) -> HWellModel:
#         if not _are_perforations_non_overlapping(self.perforations):
#             raise ValueError("Perforations can't overlap")
#         return self
#
#     def _get_sections_mds(self) -> tuple[float, float, float]:
#         """Function borrowed from HWellTemplate.get_sections_mds"""
#         FIXED_ANGLE = 90.0
#         FIXED_DLS = 4.0
#         _CURVATURE_MD = (FIXED_ANGLE * 30) / FIXED_DLS
#         _CURVATURE_RADIUS = 2 * _CURVATURE_MD / np.pi
#
#         if (vertical_LWS_md := self.TVD - _CURVATURE_RADIUS) < 0:
#             raise ValueError(
#                 "Horizontal well true total depth is less than curved well section radius. Increase depth."
#             )
#
#         elif (horizontal_LWS_md := self.md_lateral - _CURVATURE_RADIUS) < 0:
#             raise ValueError(
#                 "Horizontal well width is less than curved well section radius. Increase width."
#             )
#
#         return vertical_LWS_md, _CURVATURE_MD, horizontal_LWS_md
#
#     @model_validator(mode="after")
#     def validate_perforation_within_well_md(self) -> HWellModel:
#         if not _is_perforation_within_total_well_md(
#             self._get_sections_mds(),
#             self.perforations,
#         ):
#             raise ValueError(
#                 "One or more perforation intervals extend beyond the total measured depth of the well."
#             )
#         return self
#
#     @model_validator(mode="after")
#     def perforate_whole_md_if_no_perforation(self) -> HWellModel:
#         if not self.perforations:
#             total_md = sum(self._get_sections_mds())
#             self.perforations = [PerforationRangeModel(start_md=0, end_md=total_md)]
#         return self
#
#     @model_validator(mode="after")
#     def accept_only_one_perforation(self) -> HWellModel:
#         if self.perforations and len(self.perforations) > 1:
#             raise ValueError("Only single perforation is supported")
#         return self
#
#
# WellModel = Annotated[
#     Union[IWellModel, JwellModel, SWellModel, HWellModel],
#     Field(discriminator="well_type"),
# ]
#
#
# def _is_perforation_within_total_well_md(
#     well_mds: Sequence[float], perforations: Sequence[PerforationRangeModel] | None
# ) -> bool:
#     if not perforations:
#         return True
#
#     total_measured_depth = np.sum(well_mds)
#     return all(
#         perf.start_md <= total_measured_depth and perf.end_md <= total_measured_depth
#         for perf in perforations
#     )
#
#
# def _are_perforations_non_overlapping(
#     perforations: Sequence[PerforationRangeModel] | None,
# ) -> bool:
#     if not perforations or len(perforations) < 2:
#         return True
#
#     return all(
#         current_p.end_md <= next_p.start_md
#         for current_p, next_p in itertools.pairwise(perforations)
#     )
