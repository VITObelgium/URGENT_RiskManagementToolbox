from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, FiniteFloat, field_validator, model_validator


class Boundaries(BaseModel, extra="forbid"):
    lb: FiniteFloat
    ub: FiniteFloat

    @model_validator(mode="after")
    def validate_boundaries(self) -> Self:
        if self.lb > self.ub:
            raise ValueError(
                f"Lower bound must be strictly less than upper bound. Got lb={self.lb}, ub={self.ub}"
            )
        return self


class LinearInequalities(BaseModel, extra="forbid"):
    A: list[dict[str, FiniteFloat]] = Field(min_length=1)
    b: list[FiniteFloat] = Field(min_length=1)
    sense: list[Literal["<=", ">=", "<", ">"]] = Field(min_length=1)

    @model_validator(mode="after")
    def check_dimensions(self) -> Self:
        if not len(self.A) == len(self.b) == len(self.sense):
            raise ValueError("A, b and sense must have the same length")
        return self

    @field_validator("A")
    @classmethod
    def check_rows_not_empty(cls, value):
        for i, row in enumerate(value):
            if not row:
                raise ValueError(f"Row {i} in A must not be empty")
        return value
