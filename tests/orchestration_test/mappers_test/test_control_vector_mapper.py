from orchestration.risk_management_service.core.mappers.control_vector_mapper import (
    ControlVectorMapper,
)
from services.problem_dispatcher_service.core.models.shared_from_solution_updater_service import (
    ControlVector as PDControlVector,
)
from services.solution_updater_service.core.models.user import (
    ControlVector as SUControlVector,
)


def make_su_control_vector(items):
    return SUControlVector(items=items)


def make_pd_control_vector(items):
    return PDControlVector(items=items)


def test_convert_su_to_pd_basic():
    su_vecs = [
        make_su_control_vector({"a": 1.0, "b": 2.0}),
        make_su_control_vector({"a": 3.0, "b": 4.0}),
    ]
    result = ControlVectorMapper.convert_su_to_pd(su_vecs)
    assert isinstance(result, list)
    assert all(isinstance(v, PDControlVector) for v in result)
    assert result[0].items == {"a": 1.0, "b": 2.0}
    assert result[1].items == {"a": 3.0, "b": 4.0}


def test_convert_su_to_pd_none():
    assert ControlVectorMapper.convert_su_to_pd(None) is None


def test_convert_su_to_pd_empty_list():
    assert ControlVectorMapper.convert_su_to_pd([]) == []
