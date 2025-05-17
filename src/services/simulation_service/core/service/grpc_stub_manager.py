from contextlib import contextmanager

import grpc

from logger import get_logger
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
    channel_options = [
        ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
    ]

    @classmethod
    @contextmanager
    def get_stub(cls, server_host, server_port):
        """
        Context manager that provides a reusable gRPC stub.

        Args:
            server_host (str): The server hostname.
            server_port (int): The server port.

        Yields:
            sm_grpc.SimulationMessagingStub: The gRPC stub for simulation messaging.
        """
        # Create channel and stub if they don't exist
        if cls._channel is None or cls._stub is None:
            grpc_target = f"{server_host}:{server_port}"
            logger.info("Establishing gRPC connection to %s...", grpc_target)
            cls._channel = grpc.insecure_channel(
                grpc_target, options=cls.channel_options
            )
            cls._stub = sm_grpc.SimulationMessagingStub(cls._channel)

        try:
            yield cls._stub
        except grpc.RpcError as e:
            # If there's a connection error, close and reset the channel and stub
            logger.error("Error with gRPC communication: %s", e)
            if cls._channel is not None:
                cls._channel.close()
                cls._channel = None
                cls._stub = None
            raise

    @classmethod
    def close(cls):
        """Close the channel if it exists."""
        if cls._channel is not None:
            cls._channel.close()
            cls._channel = None
            cls._stub = None
            logger.info("Closed gRPC channel and stub.")
