import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import services.simulation_service.core.infrastructure.generated.simulation_messaging_pb2 as sm
import services.simulation_service.core.infrastructure.worker.src._simulation_worker_grpc_thread as worker
from services.simulation_service.core.connectors.common import SimulationStatus


def _make_job(connector: str = "missing_connector") -> sm.SimulationJob:
    job = sm.SimulationJob(job_id=7, connector=connector)
    job.simulation.input.payload = json.dumps({"well_design": {"wells": []}})
    job.simulation.result.result = json.dumps({"HEAT": None, "COST": 0.0})
    return job


def test_run_simulator_connector_load_error_returns_exception_status():
    job = _make_job()
    with patch.object(
        worker.ConnectorFactory,
        "get_connector",
        side_effect=ImportError("plugin is broken"),
    ):
        status, results = worker._run_simulator(job)

    assert status == SimulationStatus.EXCEPTION
    assert results == {"HEAT": None, "COST": 0.0}


def test_connector_load_error_is_submitted_immediately(tmp_path):
    job = _make_job()
    stub = MagicMock()
    stub.SubmitSimulationJob = AsyncMock(return_value=MagicMock(message="ok"))

    with (
        patch.object(
            worker.ConnectorFactory,
            "get_connector",
            side_effect=ValueError("unknown connector"),
        ),
        patch.object(worker, "compute_worker_temp_dir", return_value=tmp_path),
    ):
        asyncio.run(worker.handle_simulation_job(stub, job, "w1"))

    stub.SubmitSimulationJob.assert_awaited_once()
    submitted = stub.SubmitSimulationJob.await_args.args[0]
    assert submitted.status == sm.JobStatus.EXCEPTION
    assert submitted.job_id == 7
