"""Tests for GrpcStubManager — channel lifecycle and error handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import grpc
import pytest

from services.simulation_service.core.service.grpc_stub_manager import GrpcStubManager


class _FakeRpcError(grpc.RpcError):
    """Minimal gRPC error that carries a status code and details string."""

    def __init__(self, code: grpc.StatusCode, details: str = "") -> None:
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


@pytest.fixture(autouse=True)
def _reset_grpc_stub_manager():
    """Reset class-level channel/stub state before every test to prevent leakage."""
    GrpcStubManager._channel = None
    GrpcStubManager._stub = None
    yield
    with GrpcStubManager._lock:
        if GrpcStubManager._channel is not None:
            try:
                GrpcStubManager._channel.close()
            except Exception:
                pass
        GrpcStubManager._channel = None
        GrpcStubManager._stub = None


@pytest.fixture
def mock_grpc_channel():
    mock_channel = MagicMock()
    mock_stub = MagicMock()
    with (
        patch("grpc.insecure_channel", return_value=mock_channel) as mock_insecure,
        patch(
            "services.simulation_service.core.service.grpc_stub_manager.sm_grpc.SimulationMessagingStub",
            return_value=mock_stub,
        ) as mock_stub_cls,
    ):
        yield mock_insecure, mock_stub_cls, mock_channel, mock_stub


class TestGetStubCreation:
    def test_creates_channel_on_first_call(self, mock_grpc_channel):
        mock_insecure, _, mock_channel, mock_stub = mock_grpc_channel

        with GrpcStubManager.get_stub("localhost", 50051) as stub:
            assert stub is mock_stub

        mock_insecure.assert_called_once()
        call_args = mock_insecure.call_args
        assert call_args[0][0] == "localhost:50051"

    def test_reuses_channel_on_second_call(self, mock_grpc_channel):
        mock_insecure, _, _, _ = mock_grpc_channel

        with GrpcStubManager.get_stub("localhost", 50051):
            pass
        with GrpcStubManager.get_stub("localhost", 50051):
            pass

        assert mock_insecure.call_count == 1

    def test_stub_is_yielded(self, mock_grpc_channel):
        _, _, _, mock_stub = mock_grpc_channel

        with GrpcStubManager.get_stub("localhost", 50051) as stub:
            assert stub is mock_stub


class TestGetStubGrpcErrors:
    def test_aborted_error_resets_channel(self, mock_grpc_channel):
        _, _, mock_channel, _ = mock_grpc_channel
        err = _FakeRpcError(grpc.StatusCode.ABORTED, "aborted")

        with pytest.raises(grpc.RpcError):
            with GrpcStubManager.get_stub("localhost", 50051):
                raise err

        assert GrpcStubManager._channel is None
        assert GrpcStubManager._stub is None
        mock_channel.close.assert_called_once()

    def test_cancelled_error_resets_channel(self, mock_grpc_channel):
        _, _, mock_channel, _ = mock_grpc_channel
        err = _FakeRpcError(grpc.StatusCode.CANCELLED, "cancelled")

        with pytest.raises(grpc.RpcError):
            with GrpcStubManager.get_stub("localhost", 50051):
                raise err

        assert GrpcStubManager._channel is None

    def test_unavailable_server_shutdown_resets_channel(self, mock_grpc_channel):
        _, _, mock_channel, _ = mock_grpc_channel
        err = _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "Server shutting down")

        with pytest.raises(grpc.RpcError):
            with GrpcStubManager.get_stub("localhost", 50051):
                raise err

        assert GrpcStubManager._channel is None

    def test_unexpected_grpc_error_resets_channel(self, mock_grpc_channel):
        _, _, mock_channel, _ = mock_grpc_channel
        err = _FakeRpcError(grpc.StatusCode.INTERNAL, "unexpected failure")

        with pytest.raises(grpc.RpcError):
            with GrpcStubManager.get_stub("localhost", 50051):
                raise err

        assert GrpcStubManager._channel is None

    def test_unavailable_non_shutdown_detail_resets_channel(self, mock_grpc_channel):
        """UNAVAILABLE with a different detail string → logs error path, still resets."""
        _, _, mock_channel, _ = mock_grpc_channel
        err = _FakeRpcError(grpc.StatusCode.UNAVAILABLE, "connection refused")

        with pytest.raises(grpc.RpcError):
            with GrpcStubManager.get_stub("localhost", 50051):
                raise err

        assert GrpcStubManager._channel is None


class TestClose:
    def test_close_when_channel_open(self, mock_grpc_channel):
        _, _, mock_channel, _ = mock_grpc_channel

        with GrpcStubManager.get_stub("localhost", 50051):
            pass

        assert GrpcStubManager._channel is mock_channel
        GrpcStubManager.close()

        mock_channel.close.assert_called()
        assert GrpcStubManager._channel is None
        assert GrpcStubManager._stub is None

    def test_close_when_already_closed(self):
        """close() on an already-None channel should be a no-op."""
        assert GrpcStubManager._channel is None
        GrpcStubManager.close()  # should not raise
