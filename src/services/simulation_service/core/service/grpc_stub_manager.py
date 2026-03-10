from contextlib import contextmanager
from threading import Lock

import grpc

from logger import get_logger
from services.simulation_service.core.config import get_simulation_config
from services.simulation_service.core.infrastructure.generated import (
    simulation_messaging_pb2_grpc as sm_grpc,
)

logger = get_logger(__name__)


class GrpcStubManager:
    """
    Manager class for creating and reusing gRPC stubs.
    """

    _channel = None
    _stub = None
    _lock = Lock()
    config = get_simulation_config()

    @classmethod
    @contextmanager
    def get_stub(cls, server_host, server_port):
        # ... existing code ...
        with cls._lock:
            if cls._channel is None or cls._stub is None:
                config = get_simulation_config()
                grpc_target = f"{server_host}:{server_port}"
                logger.info("Establishing gRPC connection to %s...", grpc_target)
                cls._channel = grpc.insecure_channel(
                    grpc_target, options=config.channel_options
                )
                cls._stub = sm_grpc.SimulationMessagingStub(cls._channel)

        try:
            yield cls._stub
        except grpc.RpcError as e:
            code = None
            details = None
            try:
                if hasattr(e, "code"):
                    code = e.code()
                if hasattr(e, "details"):
                    details = e.details()
            except Exception:
                pass

            # These can be "expected" in your design:
            # - ABORTED: server intentionally aborted due to critical worker exception
            # - CANCELLED: shutdown in progress / RPC cancelled
            if code in (grpc.StatusCode.ABORTED, grpc.StatusCode.CANCELLED):
                logger.info(
                    "gRPC call ended intentionally (code=%s, details=%s).",
                    code,
                    details,
                )
            else:
                logger.error(
                    "Error with gRPC communication (code=%s, details=%s): %s",
                    code,
                    details,
                    e,
                )

            # If there's a connection error, close and reset the channel and stub
            with cls._lock:
                if cls._channel is not None:
                    cls._channel.close()
                    cls._channel = None
                    cls._stub = None
            raise

    @classmethod
    def close(cls):
        """Close the channel if it exists."""
        with cls._lock:
            if cls._channel is not None:
                cls._channel.close()
                cls._channel = None
                cls._stub = None
                logger.info("Closed gRPC channel and stub.")
