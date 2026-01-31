import asyncio
import os
import time
from queue import SimpleQueue

import grpc
from grpc import aio

import services.simulation_service.core.infrastructure.generated.simulation_messaging_pb2 as sm
import services.simulation_service.core.infrastructure.generated.simulation_messaging_pb2_grpc as sm_grpc
from logger import get_logger
from services.simulation_service.core.config import get_simulation_config

logger = get_logger("threading-server", filename=__name__)

_SERVER_LOOP: asyncio.AbstractEventLoop | None = None
_SERVER: aio.Server | None = None


class SimulationMessagingHandler(sm_grpc.SimulationMessagingServicer):
    def __init__(self) -> None:
        self._jobs: SimpleQueue = SimpleQueue()  # Queue of simulations
        self._running_jobs: dict[
            int, asyncio.Task
        ] = {}  # Dictionary to track running jobs
        self._job_start_times: dict[int, float] = {}  # Track when jobs started
        self._completed_jobs: dict[
            int, sm.SimulationResult
        ] = {}  # Dictionary to store results
        self._job_event = asyncio.Event()  # Event to signal job completion
        self._simulation_model_archive: bytes | None = None
        self._job_timeout_seconds = get_simulation_config().job_timeout_seconds

    async def TransferSimulationModel(self, request, context):
        archive_size_bytes = len(getattr(request, "package_archive", b""))
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
        total_simulations = len(request.simulations)
        logger.info(f"Initializing simulations with {total_simulations} job(s)")

        # Store the total number of simulations for progress tracking
        self._jobs = SimpleQueue()
        self._running_jobs.clear()
        self._job_start_times.clear()
        self._completed_jobs.clear()

        for i, sim in enumerate(request.simulations, start=1):
            self._jobs.put((i, sim))
            self._running_jobs[i] = None  # Mark job as waiting

        logger.info(
            f"All {total_simulations} simulation(s) queued, waiting for completion"
        )

        # Set up progress tracking variables
        start_time = time.time()
        last_log_time = start_time
        LOG_INTERVAL_SEC = 30  # Log progress every 30 seconds

        # Wait for all jobs to complete with regular progress updates
        while self._running_jobs:
            current_time = time.time()
            time_since_last_log = current_time - last_log_time

            # Check for timed-out jobs
            timed_out_jobs = []
            for job_id, start_t in list(self._job_start_times.items()):
                if (
                    job_id in self._running_jobs
                    and (current_time - start_t) > self._job_timeout_seconds
                ):
                    timed_out_jobs.append(job_id)

            if timed_out_jobs:
                logger.warning(
                    f"Jobs timed out after {self._job_timeout_seconds}s: {timed_out_jobs}"
                )
                for job_id in timed_out_jobs:
                    # Create a timeout result with the original simulation and empty result
                    original_simulation = self._running_jobs[job_id]
                    timeout_simulation = sm.Simulation(
                        input=original_simulation.input,
                        control_vector=original_simulation.control_vector,
                        result=sm.SimulationResult(result="{}"),
                    )
                    timeout_result = sm.SimulationJob(
                        simulation=timeout_simulation,
                        status=sm.JobStatus.TIMEOUT,
                        worker_id="server-timeout",
                        simulator=sm.Simulator.OPENDARTS,
                        job_id=job_id,
                    )
                    self._completed_jobs[job_id] = timeout_result
                    del self._running_jobs[job_id]
                    del self._job_start_times[job_id]
                self._job_event.set()

            # Calculate timeout for wait_for
            # If we're due for a log, timeout immediately (0)
            # Otherwise wait until the next log interval is reached
            timeout = max(0, LOG_INTERVAL_SEC - time_since_last_log)

            try:
                # Wait for the event with timeout
                await asyncio.wait_for(self._job_event.wait(), timeout)
                self._job_event.clear()
            except asyncio.TimeoutError:
                # Timeout occurred - this means it's time to log progress
                pass

            # Check if it's time to log progress
            current_time = time.time()
            if current_time - last_log_time >= LOG_INTERVAL_SEC:
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
                    logger.debug(
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
            logger.debug(f"Average processing rate: {jobs_per_minute:.1f} jobs/minute")

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
        timeout_count = sum(
            1
            for job in simulations_jobs
            if getattr(job, "status", None) == sm.JobStatus.TIMEOUT
        )

        # Log summary with appropriate level based on failures
        if failed_count or timeout_count:
            logger.warning(
                "Completed with %d successful, %d failed and %d timeout simulation(s)",
                success_count,
                failed_count,
                timeout_count,
            )
        else:
            logger.info("All %d simulation(s) completed successfully", success_count)

        # Sort completed jobs by job_id to maintain order
        sorted_jobs = sorted(self._completed_jobs.items(), key=lambda x: x[0])
        simulations = [job.simulation for _, job in sorted_jobs]

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
        self._job_start_times[job_id] = time.time()  # Track start time

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
            sm.JobStatus.TIMEOUT,
            sm.JobStatus.ERROR,
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
        if job_id in self._job_start_times:
            del self._job_start_times[job_id]

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
    global _SERVER_LOOP, _SERVER
    logger.info("Initializing Async gRPC Server setup...")
    mode = os.getenv("OPEN_DARTS_RUNNER", "thread").lower()
    logger.info(f"Server starting with OPEN_DARTS_RUNNER={mode}")

    try:
        # Install an exception handler to silence noisy gRPC poller EAGAIN callbacks
        loop = asyncio.get_running_loop()
        _warned = {"done": False}

        def _ignore_poller_eagain(loop, context):
            exc = context.get("exception")
            message = context.get("message", "")
            if isinstance(exc, BlockingIOError) and getattr(exc, "errno", None) == 11:
                if not _warned["done"]:
                    logger.warning(
                        "Suppressing gRPC AIO poller EAGAIN noise (BlockingIOError 11)"
                    )
                    _warned["done"] = True
                return
            if "PollerCompletionQueue._handle_events" in message:
                # Some environments report only via message without exception
                if not _warned["done"]:
                    logger.warning(
                        "Suppressing gRPC AIO poller _handle_events noise via message"
                    )
                    _warned["done"] = True
                return
            # Fallback to default handler for other errors
            loop.default_exception_handler(context)

        loop.set_exception_handler(_ignore_poller_eagain)

        config = get_simulation_config()
        server_options = config.channel_options
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

        _SERVER_LOOP = asyncio.get_running_loop()
        _SERVER = server

        try:
            logger.info("Server is running and waiting for termination...")
            await server.wait_for_termination()
        except asyncio.CancelledError:
            logger.info("Received cancellation: shutting down server gracefully.")

    except Exception as e:
        logger.critical(f"Critical error during server startup or operation: {e}")
        raise
    finally:
        globals()["_SERVER_LOOP"] = None
        globals()["_SERVER"] = None
        logger.info("Server serve() routine completed or interrupted.")


if __name__ == "__main__":
    logger.info("Starting Simulation gRPC Server...")
    asyncio.run(serve())


def driver():
    asyncio.run(serve())


def request_server_shutdown(timeout: float | None = 2.0) -> None:
    """Signal the in-process gRPC server to stop gracefully.

    Safe to call from other threads. If the server hasn't finished initializing,
    will wait up to `timeout` seconds for the loop/server references to appear.
    """
    end = time.time() + (timeout or 0.0)
    while (_SERVER_LOOP is None or _SERVER is None) and time.time() < end:
        time.sleep(0.05)

    if _SERVER_LOOP is None or _SERVER is None:
        logger.warning(
            "request_server_shutdown: server not initialized; nothing to stop."
        )
        return

    def _stop():
        try:
            if _SERVER is not None:
                logger.info(
                    "request_server_shutdown: initiating graceful server.stop()"
                )
                if _SERVER_LOOP is not None:
                    _SERVER_LOOP.create_task(_SERVER.stop(grace=None))
                else:
                    asyncio.create_task(_SERVER.stop(grace=None))
        except Exception:
            logger.exception("Error while requesting server shutdown")

    try:
        _SERVER_LOOP.call_soon_threadsafe(_stop)
    except Exception:
        logger.exception("Failed to schedule server shutdown on event loop")
