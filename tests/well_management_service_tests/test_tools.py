from typing import MutableMapping

import numpy as np
import pytest

from services.well_management_service.core.models import PerforationRange
from services.well_management_service.core.sections.perforation_md_provider import (
    PerforationMdProvider,
)


@pytest.mark.parametrize(
    "perforation_range, section_range , expected_result",
    [
        (
            {"p1": PerforationRange(1.0, 3.0), "p2": PerforationRange(6.0, 10.0)},
            (0, 11),
            (1.0, 3.0, 6.0, 10.0, np.inf),
        ),
        (
            {"p1": PerforationRange(1.0, 3.0), "p2": PerforationRange(6.0, 10.0)},
            (2.0, 7.0),
            (3.0, 6.0, np.inf),
        ),
        ({}, (0.0, 10.0), (np.inf, np.inf)),
        ({"p1": PerforationRange(0, 10)}, (20.0, 100.0), (np.inf, np.inf)),
    ],
)
def test_perforation_md_provider(
    perforation_range: MutableMapping[str, PerforationRange],
    section_range: tuple[float, ...],
    expected_result: tuple[float, ...],
) -> None:
    section_start_md, section_end_md = section_range
    pmdp = PerforationMdProvider(perforation_range, section_start_md, section_end_md)
    assert all((pmdp.get_next_md() == next_md for next_md in expected_result))
