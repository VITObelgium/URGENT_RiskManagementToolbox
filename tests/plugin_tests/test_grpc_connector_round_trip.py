from __future__ import annotations

from services.simulation_service.core.infrastructure.generated import (
    simulation_messaging_pb2 as sm,
)


def test_simulation_options_round_trip() -> None:
    options = sm.SimulationOptions(connector="eclipse")
    sims = sm.Simulations(options=options)
    serialized = sims.SerializeToString()

    parsed = sm.Simulations()
    parsed.ParseFromString(serialized)

    assert parsed.HasField("options")
    assert parsed.options.connector == "eclipse"


def test_simulation_job_connector_field_round_trip() -> None:
    job = sm.SimulationJob(connector="eclipse", job_id=7)
    parsed = sm.SimulationJob()
    parsed.ParseFromString(job.SerializeToString())
    assert parsed.connector == "eclipse"
    assert parsed.job_id == 7


def test_simulation_job_connector_defaults_to_empty_string() -> None:
    job = sm.SimulationJob(job_id=1)
    parsed = sm.SimulationJob()
    parsed.ParseFromString(job.SerializeToString())
    assert parsed.connector == ""
