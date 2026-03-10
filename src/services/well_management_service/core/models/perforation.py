from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from services.well_management_service.core import models


@dataclass(frozen=True, slots=True)
class Perforation:
    name: str
    range: PerforationRange
    points: tuple[models.Point, ...]


@dataclass(frozen=True, slots=True)
class PerforationRange:
    start_md: float
    end_md: float

    def __iter__(self) -> Iterator[float]:
        return iter((self.start_md, self.end_md))
