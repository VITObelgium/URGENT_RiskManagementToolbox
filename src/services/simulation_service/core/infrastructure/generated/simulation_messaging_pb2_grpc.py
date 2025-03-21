"""Client and server classes corresponding to protobuf-defined services."""

import grpc

from . import simulation_messaging_pb2 as simulation__messaging__pb2

GRPC_GENERATED_VERSION = "1.69.0"
GRPC_VERSION = grpc.__version__
_version_not_supported = False
try:
    from grpc._utilities import first_version_is_lower

    _version_not_supported = first_version_is_lower(
        GRPC_VERSION, GRPC_GENERATED_VERSION
    )
except ImportError:
    _version_not_supported = True
if _version_not_supported:
    raise RuntimeError(
        f"The grpc package installed is at version {GRPC_VERSION},"
        + " but the generated code in simulation_messaging_pb2_grpc.py depends on"
        + f" grpcio>={GRPC_GENERATED_VERSION}."
        + f" Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}"
        + f" or downgrade your generated code using grpcio-tools<={GRPC_VERSION}."
    )


class SimulationMessagingStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.TransferSimulationModel = channel.unary_unary(
            "/simulation_messaging.SimulationMessaging/TransferSimulationModel",
            request_serializer=simulation__messaging__pb2.SimulationModel.SerializeToString,
            response_deserializer=simulation__messaging__pb2.Ack.FromString,
            _registered_method=True,
        )
        self.PerformSimulations = channel.unary_unary(
            "/simulation_messaging.SimulationMessaging/PerformSimulations",
            request_serializer=simulation__messaging__pb2.Simulations.SerializeToString,
            response_deserializer=simulation__messaging__pb2.Simulations.FromString,
            _registered_method=True,
        )
        self.RequestSimulationJob = channel.unary_unary(
            "/simulation_messaging.SimulationMessaging/RequestSimulationJob",
            request_serializer=simulation__messaging__pb2.RequestJob.SerializeToString,
            response_deserializer=simulation__messaging__pb2.SimulationJob.FromString,
            _registered_method=True,
        )
        self.SubmitSimulationJob = channel.unary_unary(
            "/simulation_messaging.SimulationMessaging/SubmitSimulationJob",
            request_serializer=simulation__messaging__pb2.SimulationJob.SerializeToString,
            response_deserializer=simulation__messaging__pb2.Ack.FromString,
            _registered_method=True,
        )
        self.RequestSimulationModel = channel.unary_unary(
            "/simulation_messaging.SimulationMessaging/RequestSimulationModel",
            request_serializer=simulation__messaging__pb2.RequestModel.SerializeToString,
            response_deserializer=simulation__messaging__pb2.SimulationModel.FromString,
            _registered_method=True,
        )


class SimulationMessagingServicer(object):
    """Missing associated documentation comment in .proto file."""

    def TransferSimulationModel(self, request, context):
        """simulation service -> server"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def PerformSimulations(self, request, context):
        """simulation service -> server"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def RequestSimulationJob(self, request, context):
        """worker -> server"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def SubmitSimulationJob(self, request, context):
        """worker -> server"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def RequestSimulationModel(self, request, context):
        """worker -> server"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_SimulationMessagingServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "TransferSimulationModel": grpc.unary_unary_rpc_method_handler(
            servicer.TransferSimulationModel,
            request_deserializer=simulation__messaging__pb2.SimulationModel.FromString,
            response_serializer=simulation__messaging__pb2.Ack.SerializeToString,
        ),
        "PerformSimulations": grpc.unary_unary_rpc_method_handler(
            servicer.PerformSimulations,
            request_deserializer=simulation__messaging__pb2.Simulations.FromString,
            response_serializer=simulation__messaging__pb2.Simulations.SerializeToString,
        ),
        "RequestSimulationJob": grpc.unary_unary_rpc_method_handler(
            servicer.RequestSimulationJob,
            request_deserializer=simulation__messaging__pb2.RequestJob.FromString,
            response_serializer=simulation__messaging__pb2.SimulationJob.SerializeToString,
        ),
        "SubmitSimulationJob": grpc.unary_unary_rpc_method_handler(
            servicer.SubmitSimulationJob,
            request_deserializer=simulation__messaging__pb2.SimulationJob.FromString,
            response_serializer=simulation__messaging__pb2.Ack.SerializeToString,
        ),
        "RequestSimulationModel": grpc.unary_unary_rpc_method_handler(
            servicer.RequestSimulationModel,
            request_deserializer=simulation__messaging__pb2.RequestModel.FromString,
            response_serializer=simulation__messaging__pb2.SimulationModel.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "simulation_messaging.SimulationMessaging", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers(
        "simulation_messaging.SimulationMessaging", rpc_method_handlers
    )


class SimulationMessaging(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def TransferSimulationModel(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/simulation_messaging.SimulationMessaging/TransferSimulationModel",
            simulation__messaging__pb2.SimulationModel.SerializeToString,
            simulation__messaging__pb2.Ack.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def PerformSimulations(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/simulation_messaging.SimulationMessaging/PerformSimulations",
            simulation__messaging__pb2.Simulations.SerializeToString,
            simulation__messaging__pb2.Simulations.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def RequestSimulationJob(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/simulation_messaging.SimulationMessaging/RequestSimulationJob",
            simulation__messaging__pb2.RequestJob.SerializeToString,
            simulation__messaging__pb2.SimulationJob.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def SubmitSimulationJob(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/simulation_messaging.SimulationMessaging/SubmitSimulationJob",
            simulation__messaging__pb2.SimulationJob.SerializeToString,
            simulation__messaging__pb2.Ack.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def RequestSimulationModel(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/simulation_messaging.SimulationMessaging/RequestSimulationModel",
            simulation__messaging__pb2.RequestModel.SerializeToString,
            simulation__messaging__pb2.SimulationModel.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )
