import asyncio
import collections
import os
import time
from typing import Any

import grpc
from grpc import aio

import services.simulation_service.core.infrastructure.generated.simulation_messaging_pb2 as sm
import services.simulation_service.core.infrastructure.generated.simulation_messaging_pb2_grpc as sm_grpc
from logger import get_logger
from services.simulation_service.core.config import get_simulation_config

logger = get_logger("threading-server", filename=__name__)

_SERVER_LOOP: asyncio.AbstractEventLoop | None = None
_SERVER: aio.Server | None = None
_SERVER_READY: asyncio.Event | None = None


class SimulationMessagingHandler(sm_grpc.SimulationMessagingServicer):
    def __init__(self) -> None:
        self._pending_jobs: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()
        self._running_jobs: dict[int, Any] = {}
        self._timeout_handles: dict[int, asyncio.TimerHandle] = {}
        self._completed_jobs: dict[int, sm.SimulationJob] = {}
        self._job_event: asyncio.Event = asyncio.Event()
        self._simulation_model_archive: bytes | None = None
        self._job_timeout_seconds: float = get_simulation_config().job_timeout_seconds
        self._log_interval_seconds: int = get_simulation_config().log_interval_seconds
        self._critical_error: RuntimeError | None = None
        self._total_simulations: int = 0

    def _cancel_timeout(self, job_id: int) -> None:
        handle = self._timeout_handles.pop(job_id, None)
        if handle is not None:
            handle.cancel()

    def _schedule_timeout(self, job_id: int) -> None:
        def _on_timeout() -> None:
            if job_id not in self._running_jobs:
                return
            logger.warning(
                "Job %d timed out after %.1fs", job_id, self._job_timeout_seconds
            )
            self._completed_jobs[job_id] = sm.SimulationJob(
                simulation=sm.Simulation(),
                status=sm.JobStatus.TIMEOUT,
                worker_id="server-timeout",
                job_id=job_id,
            )
            self._running_jobs.pop(job_id, None)
            self._timeout_handles.pop(job_id, None)
            self._job_event.set()

        loop = asyncio.get_running_loop()
        self._timeout_handles[job_id] = loop.call_later(
            self._job_timeout_seconds, _on_timeout
        )

    def _all_done(self) -> bool:
        return self._pending_jobs.empty() and not self._running_jobs

    async def TransferSimulationModel(self, request, context):
        size_mb = len(getattr(request, "package_archive", b"")) / (1024 * 1024)
        logger.info("Received simulation model archive of size %.2f MB.", size_mb)
        self._simulation_model_archive = request.package_archive
        logger.info("Simulation model archive stored on server successfully.")
        return sm.Ack(
            message="Simulation model archive (%.2f MB) stored on server." % size_mb
        )

    async def PerformSimulations(self, request, context):
        total = len(request.simulations)
        self._total_simulations = total
        logger.info("Initializing simulations with %d job(s)", total)

        self._pending_jobs = asyncio.Queue()
        self._running_jobs.clear()
        for h in self._timeout_handles.values():
            h.cancel()
        self._timeout_handles.clear()
        self._completed_jobs.clear()
        self._critical_error = None
        self._job_event.clear()

        for i, sim in enumerate(request.simulations, start=1):
            await self._pending_jobs.put((i, sim))

        logger.info("All %d simulation(s) queued, waiting for completion", total)

        start_time = time.monotonic()

        _log_handle: list[asyncio.TimerHandle | None] = [None]

        def _log_progress() -> None:
            completed = len(self._completed_jobs)
            remaining = len(self._running_jobs) + self._pending_jobs.qsize()
            elapsed = time.monotonic() - start_time
            pct = completed / total * 100 if total else 0.0
            logger.info(
                "Progress: %.1f%% complete (%d/%d) - %.1fs elapsed",
                pct,
                completed,
                total,
                elapsed,
            )
            if completed > 0 and elapsed > 0:
                rate = completed / (elapsed / 60)
                est = remaining / rate if rate > 0 else 0.0
                logger.debug(
                    "Processing rate: %.1f jobs/min, est. remaining: %.1f min",
                    rate,
                    est,
                )
            if not self._all_done():
                _log_handle[0] = asyncio.get_running_loop().call_later(
                    self._log_interval_seconds, _log_progress
                )

        _log_handle[0] = asyncio.get_running_loop().call_later(
            self._log_interval_seconds, _log_progress
        )

        try:
            while not self._all_done():
                if self._critical_error is not None:
                    details = str(self._critical_error) or "Critical worker exception"
                    logger.critical(
                        "Critical worker exception; aborting RPC. details=%s", details
                    )
                    request_server_shutdown(timeout=2.0)
                    await context.abort(grpc.StatusCode.ABORTED, details)
                    return sm.Simulations(simulations=[])

                await self._job_event.wait()
                self._job_event.clear()
        finally:
            if _log_handle[0] is not None:
                _log_handle[0].cancel()

        total_time = time.monotonic() - start_time
        logger.info(
            "All simulations completed in %.2f seconds (%.2f minutes).",
            total_time,
            total_time / 60,
        )
        if total_time > 0:
            logger.debug(
                "Average processing rate: %.1f jobs/minute", total / (total_time / 60)
            )

        counts: collections.Counter = collections.Counter(
            getattr(job, "status", None) for job in self._completed_jobs.values()
        )
        success_count = counts[sm.JobStatus.SUCCESS]
        failed_count = counts[sm.JobStatus.FAILED]
        timeout_count = counts[sm.JobStatus.TIMEOUT]

        if failed_count or timeout_count:
            logger.warning(
                "Completed with %d successful, %d failed, %d timeout simulation(s)",
                success_count,
                failed_count,
                timeout_count,
            )
        else:
            logger.info("All %d simulation(s) completed successfully", success_count)

        sorted_jobs = sorted(self._completed_jobs.items(), key=lambda x: x[0])
        simulations = [job.simulation for _, job in sorted_jobs]
        logger.info("Returning %d completed simulation(s).", len(simulations))
        return sm.Simulations(simulations=simulations)

    async def RequestSimulationJob(self, request, context):
        try:
            worker_id = request.worker_id

            try:
                job_id, simulation = self._pending_jobs.get_nowait()
            except asyncio.QueueEmpty:
                logger.info("Worker %s: No jobs available in queue", worker_id)
                if self._running_jobs:
                    logger.info("Jobs still in-flight: %s", list(self._running_jobs))
                return sm.SimulationJob(
                    simulation=sm.Simulation(),
                    status=sm.JobStatus.NO_JOB_AVAILABLE,
                    worker_id=worker_id,
                    simulator=sm.Simulator.SIMULATOR_UNSPECIFIED,
                )

            self._running_jobs[job_id] = simulation
            self._schedule_timeout(job_id)
            logger.info("Worker %s: Assigned job %d", worker_id, job_id)

            return sm.SimulationJob(
                simulation=simulation,
                status=sm.JobStatus.NEW,
                worker_id=worker_id,
                simulator=sm.Simulator.OPENDARTS,
                job_id=job_id,
            )

        except asyncio.CancelledError:
            try:
                context.set_code(grpc.StatusCode.CANCELLED)
                context.set_details("Server shutting down")
            except Exception:
                pass
            return sm.SimulationJob(
                simulation=sm.Simulation(),
                status=sm.JobStatus.NO_JOB_AVAILABLE,
                worker_id=getattr(request, "worker_id", ""),
                simulator=sm.Simulator.SIMULATOR_UNSPECIFIED,
            )

    async def SubmitSimulationJob(self, request, context):
        job_id = request.job_id
        worker_id = request.worker_id

        if job_id not in self._running_jobs:
            logger.warning(
                "Worker %s: Attempted to submit invalid job ID %d", worker_id, job_id
            )
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return sm.Ack(message="Invalid job %d ID." % job_id)

        _VALID_TERMINAL = frozenset(
            {
                sm.JobStatus.SUCCESS,
                sm.JobStatus.FAILED,
                sm.JobStatus.TIMEOUT,
                sm.JobStatus.ERROR,
                sm.JobStatus.EXCEPTION,
            }
        )
        if request.status not in _VALID_TERMINAL:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid job %d status '%s'" % (job_id, request.status))
            logger.warning(
                "Worker %s: Invalid job %d status '%s'",
                worker_id,
                job_id,
                request.status,
            )
            return sm.Ack(
                message="Invalid job status from Worker %s, job %d status %s"
                % (worker_id, job_id, request.status)
            )

        status_name = (
            sm.JobStatus.Name(request.status)
            if hasattr(sm.JobStatus, "Name")
            else str(request.status)
        )
        logger.info(
            "Worker %s: Returned job %d with status %s", worker_id, job_id, status_name
        )

        if request.status == sm.JobStatus.EXCEPTION:
            msg = "Critical worker EXCEPTION from worker=%s job_id=%d" % (
                worker_id,
                job_id,
            )
            logger.critical(msg)
            if self._critical_error is None:
                self._critical_error = RuntimeError(msg)
            self._job_event.set()
            request_server_shutdown(timeout=2.0)

        self._cancel_timeout(job_id)
        self._completed_jobs[job_id] = request
        self._running_jobs.pop(job_id, None)
        self._job_event.set()

        return sm.Ack(
            message="Worker %s: Returned job %d with status %s"
            % (worker_id, job_id, status_name)
        )

    async def RequestSimulationModel(self, request, context):
        worker_id = request.worker_id
        logger.info("Worker %s: Requested a simulation model archive.", worker_id)
        if not self._simulation_model_archive:
            return sm.SimulationModel(
                package_archive=bytes(), status=sm.ModelStatus.NO_MODEL_AVAILABLE
            )
        return sm.SimulationModel(
            package_archive=self._simulation_model_archive,
            status=sm.ModelStatus.ON_SERVER,
        )


async def serve() -> None:
    global _SERVER_LOOP, _SERVER, _SERVER_READY

    logger.info("Initializing Async gRPC Server setup...")
    mode = os.getenv("OPEN_DARTS_RUNNER", "thread").lower()
    logger.info("Server starting with OPEN_DARTS_RUNNER=%s", mode)

    _SERVER_READY = asyncio.Event()

    try:
        loop = asyncio.get_running_loop()
        _warned = {"done": False}

        def _ignore_poller_eagain(loop, context):
            exc = context.get("exception")
            message = context.get("message", "")
            if isinstance(exc, BlockingIOError) and getattr(exc, "errno", None) == 11:
                if not _warned["done"]:
                    logger.debug(
                        "Suppressing gRPC AIO poller EAGAIN noise (BlockingIOError 11)"
                    )
                    _warned["done"] = True
                return
            if "PollerCompletionQueue._handle_events" in message:
                if not _warned["done"]:
                    logger.debug("Suppressing gRPC AIO poller _handle_events noise")
                    _warned["done"] = True
                return
            loop.default_exception_handler(context)

        loop.set_exception_handler(_ignore_poller_eagain)

        config = get_simulation_config()
        server_options = config.channel_options
        logger.debug("gRPC server options: %s", server_options)

        server = aio.server(options=server_options)
        sm_grpc.add_SimulationMessagingServicer_to_server(
            SimulationMessagingHandler(), server
        )

        manager_port = os.environ.get("SERVER_PORT", "50051")
        logger.info(
            "Binding server to port %s (from SERVER_PORT env var).", manager_port
        )
        server.add_insecure_port("[::]:%s" % manager_port)

        await server.start()
        logger.info("Async gRPC Server successfully started on port %s.", manager_port)

        _SERVER_LOOP = loop
        _SERVER = server
        _SERVER_READY.set()  # unblock any readiness waiters

        try:
            await server.wait_for_termination()
        except asyncio.CancelledError:
            logger.info("Received cancellation: shutting down server gracefully.")

    except Exception as e:
        logger.critical("Critical error during server startup or operation: %s", e)
        raise
    finally:
        _SERVER_LOOP = None
        _SERVER = None
        if _SERVER_READY is not None:
            _SERVER_READY.clear()
        logger.info("Server serve() routine completed or interrupted.")


if __name__ == "__main__":
    logger.info("Starting Simulation gRPC Server...")
    asyncio.run(serve())


def driver():
    asyncio.run(serve())


def request_server_shutdown(timeout: float | None = 2.0) -> None:
    end = time.monotonic() + (timeout or 0.0)
    while (_SERVER_LOOP is None or _SERVER is None) and time.monotonic() < end:
        time.sleep(0.05)

    if _SERVER_LOOP is None or _SERVER is None:
        logger.warning(
            "request_server_shutdown: server not initialized; nothing to stop."
        )
        return

    def _stop() -> None:
        try:
            if _SERVER is not None:
                logger.info(
                    "request_server_shutdown: initiating graceful server.stop()"
                )
                asyncio.ensure_future(_SERVER.stop(grace=0.0), loop=_SERVER_LOOP)
        except Exception:
            logger.exception("Error while requesting server shutdown")

    try:
        _SERVER_LOOP.call_soon_threadsafe(_stop)
    except Exception:
        logger.exception("Failed to schedule server shutdown on event loop")
