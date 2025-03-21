from dataclasses import dataclass, field

from services.well_management_service.core import models


@dataclass(frozen=True, slots=True)
class Completion:
    perforations: tuple[models.Perforation, ...]


@dataclass(frozen=True, slots=True)
class Well:
    """
    This class is a data container for a well's specifications and should be used
    with a well template or well builder for proper instantiation. Direct use is discouraged.
    """

    name: str
    trajectory: models.Trajectory
    completion: Completion | None = field(default=None)
