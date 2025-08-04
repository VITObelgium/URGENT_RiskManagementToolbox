import pytest

from services.simulation_service.core.models.shared_from_well_management import (
    SimulationWellModel,
    SimulationWellPerforationModel,
    WellManagementServiceResult,
)


def test_simulation_well_perforation_model_invalid_range():
    with pytest.raises(
        ValueError,
        match=r"Invalid range: start \(300.0\) must be less than end \(200.0\)",
    ):
        SimulationWellPerforationModel(range=(300, 200), points=((0.0, 0.0, 0.0),))


def test_simulation_well_perforation_model_valid_range():
    model = SimulationWellPerforationModel(range=(100, 200), points=((0.0, 0.0, 0.0),))
    assert model.range == (100, 200)


def test_well_management_service_result_unique_names():
    wells = [
        SimulationWellModel(
            name="Well1", trajectory=((0.0, 0.0, 0.0),), completion=None
        ),
        SimulationWellModel(
            name="Well2", trajectory=((0.0, 0.0, 0.0),), completion=None
        ),
    ]
    result = WellManagementServiceResult(wells=wells)
    assert result.wells == wells


def test_well_management_service_result_duplicate_names():
    wells = [
        SimulationWellModel(
            name="Well1", trajectory=((0.0, 0.0, 0.0),), completion=None
        ),
        SimulationWellModel(
            name="Well1", trajectory=((0.0, 0.0, 0.0),), completion=None
        ),
    ]
    with pytest.raises(ValueError, match="Wells names must be unique."):
        WellManagementServiceResult(wells=wells)
