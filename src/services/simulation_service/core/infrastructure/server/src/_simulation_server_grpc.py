import asyncio
import os
import time
from queue import SimpleQueue

import generated.simulation_messaging_pb2 as sm
import generated.simulation_messaging_pb2_grpc as sm_grpc
import grpc
from grpc import aio

from logger.u_logger import configure_logger, get_logger

configure_logger()

logger = get_logger(__name__)

server_options = [
    ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
    ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
]


class SimulationMessagingHandler(sm_grpc.SimulationMessagingServicer):
    def __init__(self) -> None:
        self._jobs = SimpleQueue()  # Queue of simulations
        self._running_jobs = {}  # Dictionary to track running jobs
        self._completed_jobs = {}  # Dictionary to store results
        self._job_event = asyncio.Event()  # Event to signal job completion
        self._simulation_model_archive: bytes | None = None

    async def TransferSimulationModel(self, request, context):
        archive_size_bytes = (
            len(request.package_archive) if hasattr(request, "package_archive") else 0
        )
        archive_size_mb = archive_size_bytes / (1024 * 1024)
        logger.info(
            f"Received simulation model archive of size {archive_size_mb:.2f} MB."
        )

        self._simulation_model_archive = None
        simulation_model_archive = request.package_archive
        self._simulation_model_archive = simulation_model_archive

        logger.info("Simulation model archive stored on server successfully.")

        return sm.Ack(
            message=f"Simulation model archive ({archive_size_mb:.2f} MB) stored on server."
        )

    async def PerformSimulations(self, request, context):
        logger.info(f"Initializing simulations with {len(request.simulations)} job(s)")

        # Store the total number of simulations for progress tracking
        total_simulations = len(request.simulations)

        self._jobs = SimpleQueue()
        self._running_jobs.clear()
        self._completed_jobs.clear()

        for i, sim in enumerate(request.simulations, start=1):
            self._jobs.put((i, sim))
            self._running_jobs[i] = None  # Mark job as waiting

        logger.info(
            f"All {len(request.simulations)} simulation(s) queued, waiting for completion"
        )

        # Set up progress tracking variables
        start_time = time.time()
        last_log_time = start_time
        log_interval = 30  # Log progress every 30 seconds

        # Wait for all jobs to complete with regular progress updates
        while self._running_jobs:
            current_time = time.time()
            time_since_last_log = current_time - last_log_time

            # Calculate timeout for wait_for
            # If we're due for a log, timeout immediately (0)
            # Otherwise wait until the next log interval is reached
            timeout = max(0, log_interval - time_since_last_log)

            try:
                # Wait for the event with timeout
                await asyncio.wait_for(self._job_event.wait(), timeout)
                self._job_event.clear()
            except asyncio.TimeoutError:
                # Timeout occurred - this means it's time to log progress
                pass

            # Check if it's time to log progress
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                completed = len(self._completed_jobs)
                remaining = len(self._running_jobs)
                percent_complete = (completed / total_simulations) * 100
                elapsed = current_time - start_time

                # Log basic progress at INFO level
                logger.info(
                    f"Progress: {percent_complete:.1f}% complete ({completed}/{total_simulations}) - {elapsed:.1f} seconds elapsed"
                )

                # Log detailed metrics if we have enough data
                if completed > 0:
                    # Calculate rate in jobs per minute
                    elapsed_minutes = elapsed / 60
                    rate_per_minute = (
                        completed / elapsed_minutes if elapsed_minutes > 0 else 0
                    )

                    # Estimate remaining time in minutes
                    est_remaining_minutes = (
                        remaining / rate_per_minute if rate_per_minute > 0 else 0
                    )

                    # Format for better readability
                    logger.info(
                        f"Processing rate: {rate_per_minute:.1f} jobs/min, estimated time remaining: {est_remaining_minutes:.1f} minutes"
                    )

                last_log_time = current_time

        # Calculate final performance metrics
        total_time = time.time() - start_time
        total_minutes = total_time / 60

        logger.info(
            f"All simulations completed in {total_time:.2f} seconds ({total_minutes:.2f} minutes)."
        )

        if total_minutes > 0:
            jobs_per_minute = total_simulations / total_minutes
            logger.info(f"Average processing rate: {jobs_per_minute:.1f} jobs/minute")

        # Count successes and failures
        simulations_jobs = self._completed_jobs.values()
        success_count = sum(
            1
            for job in simulations_jobs
            if getattr(job, "status", None) == sm.JobStatus.SUCCESS
        )
        failed_count = sum(
            1
            for job in simulations_jobs
            if getattr(job, "status", None) == sm.JobStatus.FAILED
        )

        # Log summary with appropriate level based on failures
        if failed_count > 0:
            logger.warning(
                f"Completed with {success_count} successful and {failed_count} failed simulation(s)"
            )
        else:
            logger.info(f"All {success_count} simulation(s) completed successfully")

        simulations = [j.simulation for j in simulations_jobs]
        logger.info(f"Returning {len(simulations)} completed simulation(s).")
        return sm.Simulations(simulations=simulations)

    async def RequestSimulationJob(self, request, context):
        worker_id = request.worker_id

        if self._jobs.empty():
            logger.info(f"Worker {worker_id}: No jobs available in queue")
            if self._running_jobs:
                logger.info(
                    f"Following jobs are still running: {[job for job in self._running_jobs]}"
                )
            return sm.SimulationJob(
                simulation=sm.Simulation(),
                status=sm.JobStatus.NO_JOB_AVAILABLE,
                worker_id=request.worker_id,
                simulator=sm.Simulator.SIMULATOR_UNSPECIFIED,
            )
        job_id, simulation = self._jobs.get()
        logger.info(f"Worker {worker_id}: Assigned job {job_id}")

        self._running_jobs[job_id] = simulation  # Mark job as running

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
        worker_id = request.worker_id

        if job_id not in self._running_jobs:
            logger.warning(
                f"Worker {worker_id}: Attempted to submit invalid job ID {job_id}"
            )
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return sm.Ack(message=f"Invalid job {job_id} ID.")

        if simulation_job.status not in [
            sm.JobStatus.SUCCESS,
            sm.JobStatus.FAILED,
        ]:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"Invalid job {job_id} status '{simulation_job.status}'"
            )
            logger.warning(
                f"Worker {worker_id}: Invalid job {job_id} status '{simulation_job.status}'"
            )
            return sm.Ack(
                message=f"Invalid job status from Worker {worker_id}, job {job_id} status {simulation_job.status}"
            )

        status_name = (
            sm.JobStatus.Name(simulation_job.status)
            if hasattr(sm.JobStatus, "Name")
            else simulation_job.status
        )
        logger.info(
            f"Worker {worker_id}: Returned job {job_id} with status {status_name}"
        )

        self._completed_jobs[job_id] = request
        del self._running_jobs[job_id]

        if not self._running_jobs:
            self._job_event.set()

        return sm.Ack(
            message=f"Worker {worker_id}: Returned job {job_id} with status {status_name}"
        )

    async def RequestSimulationModel(self, request, context):
        worker_id = request.worker_id

        logger.info(f"Worker {worker_id}: Requested a simulation model archive.")
        if not self._simulation_model_archive:
            return sm.SimulationModel(
                package_archive=bytes(), status=sm.ModelStatus.NO_MODEL_AVAILABLE
            )

        return sm.SimulationModel(
            package_archive=self._simulation_model_archive,
            status=sm.ModelStatus.ON_SERVER,
        )


async def serve() -> None:
    logger.info("Initializing Async gRPC Server setup...")

    try:
        # Log server options for troubleshooting and transparency
        logger.debug(f"gRPC server options: {server_options}")

        server = aio.server(options=server_options)
        logger.debug("gRPC aio server instance created.")

        sm_grpc.add_SimulationMessagingServicer_to_server(
            SimulationMessagingHandler(), server
        )
        logger.debug("SimulationMessagingServicer added to server.")

        manager_port = os.environ.get("SERVER_PORT", "50051")
        logger.info(
            f"Binding server to port {manager_port} (from SERVER_PORT env var)."
        )
        server.add_insecure_port(f"[::]:{manager_port}")

        logger.info(f"Async gRPC Server starting on port {manager_port}...")
        await server.start()
        logger.info("Async gRPC Server successfully started.")

        try:
            logger.info("Server is running and waiting for termination...")
            await server.wait_for_termination()
        except asyncio.CancelledError:
            logger.info("Received cancellation: shutting down server gracefully.")

    except Exception as e:
        logger.critical(f"Critical error during server startup or operation: {e}")
        raise
    finally:
        logger.info("Server serve() routine completed or interrupted.")


if __name__ == "__main__":
    asyncio.run(serve())
