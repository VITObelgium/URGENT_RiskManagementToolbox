"""Unit tests for the well_counter demo domain service plugin."""

import pytest
from pydantic import ValidationError

from plugins.domain_services.well_counter import WellCounterDomainService, plugin


def test_descriptor_matches_service_name():
    assert plugin.name == "well_counter"


def test_build_payload_rounds_counts():
    payload = WellCounterDomainService().build_payload(
        [{"name": "a", "count": 2.4}, {"name": "b", "count": 2.6}]
    )
    assert payload == {"counts": {"a": 2, "b": 3}}


def test_build_payload_rejects_negative_count():
    with pytest.raises(ValidationError):
        WellCounterDomainService().build_payload([{"name": "a", "count": -1.0}])
