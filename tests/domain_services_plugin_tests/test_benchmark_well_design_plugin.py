"""Unit tests for the benchmark_well_design domain service plugin."""

import pytest
from pydantic import ValidationError

from plugins.domain_services.benchmark_well_design import (
    BenchmarkWellDesignDomainService,
    plugin,
)


def _item(name: str, x: float, y: float, z: float | None = None) -> dict:
    wellhead = {"x": x, "y": y} if z is None else {"x": x, "y": y, "z": z}
    return {"name": name, "wellhead": wellhead}


def test_descriptor_matches_service_name():
    assert plugin.name == "benchmark_well_design"
    assert plugin.implementation is BenchmarkWellDesignDomainService


def test_trace_provides_well_design_tag():
    assert BenchmarkWellDesignDomainService.trace() == frozenset(
        {"benchmark_well_design", "well_design"}
    )


def test_build_payload_flattens_wellheads_in_item_order():
    payload = BenchmarkWellDesignDomainService().build_payload(
        [_item("W1", 1.0, 2.0), _item("W2", 3.5, -4.0, z=7.0)]
    )

    assert payload["x"] == [1.0, 2.0, 3.5, -4.0]
    assert payload["wellheads"] == {
        "W1": {"x": 1.0, "y": 2.0, "z": 0.0},
        "W2": {"x": 3.5, "y": -4.0, "z": 7.0},
    }


def test_build_payload_rejects_duplicate_names():
    with pytest.raises(ValidationError, match="must be unique"):
        BenchmarkWellDesignDomainService().build_payload(
            [_item("W1", 0.0, 0.0), _item("W1", 1.0, 1.0)]
        )


def test_build_payload_requires_at_least_one_item():
    with pytest.raises(ValidationError):
        BenchmarkWellDesignDomainService().build_payload([])


def test_item_state_adapter_validates_and_rejects_extra_fields():
    adapter = BenchmarkWellDesignDomainService.get_item_state_adapter()

    state = adapter.dump_python(adapter.validate_python(_item("W1", 1.0, 2.0)))
    assert state == {"name": "W1", "wellhead": {"x": 1.0, "y": 2.0, "z": 0.0}}

    with pytest.raises(ValidationError):
        adapter.validate_python({**_item("W1", 1.0, 2.0), "md": 100.0})
    with pytest.raises(ValidationError):
        adapter.validate_python(_item("W1", float("inf"), 2.0))
