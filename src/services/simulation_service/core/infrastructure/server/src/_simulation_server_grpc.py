import asyncio
import os
from queue import SimpleQueue

import generated.simulation_messaging_pb2 as sm
import generated.simulation_messaging_pb2_grpc as sm_grpc
import grpc
from grpc import aio


class SimulationMessagingHandler(sm_grpc.SimulationMessagingServicer):
    def __init__(self) -> None:
        self._jobs = SimpleQueue()  # Queue of simulations
        self._running_jobs = {}  # Dictionary to track running jobs
        self._completed_jobs = {}  # Dictionary to store results
        self._job_event = asyncio.Event()  # Event to signal job completion
        self._simulation_model_archive: bytes | None = None

    async def TransferSimulationModel(self, request, context):
        self._simulation_model_archive = None
        print("Received simulation model archive")
        simulation_model_archive = request.package_archive
        self._simulation_model_archive = simulation_model_archive
        return sm.Ack(message="Simulation model archive stored on server")

    async def PerformSimulations(self, request, context):
        print("Initializing simulations...")

        self._jobs = SimpleQueue()
        self._running_jobs.clear()
        self._completed_jobs.clear()

        for i, sim in enumerate(request.simulations, start=1):
            self._jobs.put((i, sim))
            self._running_jobs[i] = None  # Mark job as waiting

        print(f"Queued {len(request.simulations)} simulation(s).")

        # Wait for all jobs to complete
        while self._running_jobs:
            await self._job_event.wait()  # Wait for job completion signal
            self._job_event.clear()

        print("All simulations completed.")
        simulations_jobs = self._completed_jobs.values()
        simulations = [j.simulation for j in simulations_jobs]
        print(f"Returning {simulations} completed simulation(s).")
        return sm.Simulations(simulations=simulations)

    async def RequestSimulationJob(self, request, context):
        if self._jobs.empty():
            print(f"Worker {request.worker_id}: No jobs available.")
            return sm.SimulationJob(
                simulation=sm.Simulation(),
                status=sm.JobStatus.NO_JOB_AVAILABLE,
                worker_id=request.worker_id,
                simulator=sm.Simulator.SIMULATOR_UNSPECIFIED,
            )

        print(f"Worker {request.worker_id}: Requested a job")

        job_id, simulation = self._jobs.get()
        self._running_jobs[job_id] = simulation  # Mark job as running

        print(f"Worker {request.worker_id}: Assigned job {job_id}")

        return sm.SimulationJob(
            simulation=simulation,
            status=sm.JobStatus.NEW,
            worker_id=request.worker_id,
            simulator=sm.Simulator.OPENDARTS,
            job_id=job_id,
        )

    async def SubmitSimulationJob(self, request, context):
        simulation_job = request

        job_id = simulation_job.job_id

        if job_id not in self._running_jobs:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return sm.Ack(message="Invalid job ID.")

        if simulation_job.status not in [
            sm.JobStatus.SUCCESS,
            sm.JobStatus.FAILED,
        ]:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Invalid job status '{simulation_job.status}'")
            return sm.Ack(
                message=f"Invalid job status from Worker {simulation_job.worker_id}"
            )

        print(
            f"Worker {simulation_job.worker_id}: Returned job with status {simulation_job.status}"
        )

        self._completed_jobs[job_id] = request
        del self._running_jobs[job_id]

        if not self._running_jobs:
            self._job_event.set()

        return sm.Ack(message=f"Job {job_id} completed.")

    async def RequestSimulationModel(self, request, context):
        print(f"Worker {request.worker_id}: Requested a simulation model archive.")
        if not self._simulation_model_archive:
            return sm.SimulationModel(
                package_archive=bytes(), status=sm.ModelStatus.NO_MODEL_AVAILABLE
            )

        return sm.SimulationModel(
            package_archive=self._simulation_model_archive,
            status=sm.ModelStatus.ON_SERVER,
        )


async def serve() -> None:
    server = aio.server()
    sm_grpc.add_SimulationMessagingServicer_to_server(
        SimulationMessagingHandler(), server
    )
    manager_port = os.environ.get("SERVER_PORT", "50051")
    server.add_insecure_port(f"[::]:{manager_port}")
    print(f"Async gRPC Server started on port {manager_port}.")
    await server.start()
    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        print("Server shutdown gracefully")


if __name__ == "__main__":
    asyncio.run(serve())
