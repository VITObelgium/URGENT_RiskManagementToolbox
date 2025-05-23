"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""

import builtins
import collections.abc
import sys
import typing

import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions
DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _JobStatus:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _JobStatusEnumTypeWrapper(
    google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_JobStatus.ValueType],
    builtins.type,
):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    JOBSTATUS_UNSPECIFIED: _JobStatus.ValueType
    "Default value; should not be used actively."
    NEW: _JobStatus.ValueType
    "A new job has been created."
    SUCCESS: _JobStatus.ValueType
    "Job has completed successfully."
    FAILED: _JobStatus.ValueType
    "Job has failed."
    NO_JOB_AVAILABLE: _JobStatus.ValueType
    "When no job is available for a worker."
    ERROR: _JobStatus.ValueType
    "Represents an internal server error or failure."
    TIMEOUT: _JobStatus.ValueType
    "Job has timeout"

class JobStatus(_JobStatus, metaclass=_JobStatusEnumTypeWrapper): ...

JOBSTATUS_UNSPECIFIED: JobStatus.ValueType
"Default value; should not be used actively."
NEW: JobStatus.ValueType
"A new job has been created."
SUCCESS: JobStatus.ValueType
"Job has completed successfully."
FAILED: JobStatus.ValueType
"Job has failed."
NO_JOB_AVAILABLE: JobStatus.ValueType
"When no job is available for a worker."
ERROR: JobStatus.ValueType
"Represents an internal server error or failure."
TIMEOUT: JobStatus.ValueType
"Job has timeout"
global___JobStatus = JobStatus

class _Simulator:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _SimulatorEnumTypeWrapper(
    google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_Simulator.ValueType],
    builtins.type,
):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    SIMULATOR_UNSPECIFIED: _Simulator.ValueType
    OPENDARTS: _Simulator.ValueType

class Simulator(_Simulator, metaclass=_SimulatorEnumTypeWrapper): ...

SIMULATOR_UNSPECIFIED: Simulator.ValueType
OPENDARTS: Simulator.ValueType
global___Simulator = Simulator

class _ModelStatus:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _ModelStatusEnumTypeWrapper(
    google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_ModelStatus.ValueType],
    builtins.type,
):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    MODELSTATUS_UNSPECIFIED: _ModelStatus.ValueType
    NO_MODEL_AVAILABLE: _ModelStatus.ValueType
    ON_SERVER: _ModelStatus.ValueType

class ModelStatus(_ModelStatus, metaclass=_ModelStatusEnumTypeWrapper): ...

MODELSTATUS_UNSPECIFIED: ModelStatus.ValueType
NO_MODEL_AVAILABLE: ModelStatus.ValueType
ON_SERVER: ModelStatus.ValueType
global___ModelStatus = ModelStatus

@typing.final
class SimulationInput(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    WELLS_FIELD_NUMBER: builtins.int
    wells: builtins.str

    def __init__(self, *, wells: builtins.str = ...) -> None: ...
    def ClearField(self, field_name: typing.Literal["wells", b"wells"]) -> None: ...

global___SimulationInput = SimulationInput

@typing.final
class SimulationControlVector(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    CONTENT_FIELD_NUMBER: builtins.int
    content: builtins.str

    def __init__(self, *, content: builtins.str = ...) -> None: ...
    def ClearField(self, field_name: typing.Literal["content", b"content"]) -> None: ...

global___SimulationControlVector = SimulationControlVector

@typing.final
class SimulationResult(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    RESULT_FIELD_NUMBER: builtins.int
    result: builtins.str

    def __init__(self, *, result: builtins.str = ...) -> None: ...
    def ClearField(self, field_name: typing.Literal["result", b"result"]) -> None: ...

global___SimulationResult = SimulationResult

@typing.final
class SimulationJob(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    SIMULATION_FIELD_NUMBER: builtins.int
    STATUS_FIELD_NUMBER: builtins.int
    WORKER_ID_FIELD_NUMBER: builtins.int
    SIMULATOR_FIELD_NUMBER: builtins.int
    JOB_ID_FIELD_NUMBER: builtins.int
    status: global___JobStatus.ValueType
    worker_id: builtins.str
    simulator: global___Simulator.ValueType
    job_id: builtins.int

    @property
    def simulation(self) -> global___Simulation: ...
    def __init__(
        self,
        *,
        simulation: global___Simulation | None = ...,
        status: global___JobStatus.ValueType = ...,
        worker_id: builtins.str = ...,
        simulator: global___Simulator.ValueType = ...,
        job_id: builtins.int = ...,
    ) -> None: ...
    def HasField(
        self, field_name: typing.Literal["simulation", b"simulation"]
    ) -> builtins.bool: ...
    def ClearField(
        self,
        field_name: typing.Literal[
            "job_id",
            b"job_id",
            "simulation",
            b"simulation",
            "simulator",
            b"simulator",
            "status",
            b"status",
            "worker_id",
            b"worker_id",
        ],
    ) -> None: ...

global___SimulationJob = SimulationJob

@typing.final
class RequestJob(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    WORKER_ID_FIELD_NUMBER: builtins.int
    worker_id: builtins.str

    def __init__(self, *, worker_id: builtins.str = ...) -> None: ...
    def ClearField(
        self, field_name: typing.Literal["worker_id", b"worker_id"]
    ) -> None: ...

global___RequestJob = RequestJob

@typing.final
class Ack(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    MESSAGE_FIELD_NUMBER: builtins.int
    message: builtins.str

    def __init__(self, *, message: builtins.str = ...) -> None: ...
    def ClearField(self, field_name: typing.Literal["message", b"message"]) -> None: ...

global___Ack = Ack

@typing.final
class Simulation(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    INPUT_FIELD_NUMBER: builtins.int
    RESULT_FIELD_NUMBER: builtins.int
    CONTROL_VECTOR_FIELD_NUMBER: builtins.int

    @property
    def input(self) -> global___SimulationInput: ...
    @property
    def result(self) -> global___SimulationResult: ...
    @property
    def control_vector(self) -> global___SimulationControlVector: ...
    def __init__(
        self,
        *,
        input: global___SimulationInput | None = ...,
        result: global___SimulationResult | None = ...,
        control_vector: global___SimulationControlVector | None = ...,
    ) -> None: ...
    def HasField(
        self,
        field_name: typing.Literal[
            "control_vector", b"control_vector", "input", b"input", "result", b"result"
        ],
    ) -> builtins.bool: ...
    def ClearField(
        self,
        field_name: typing.Literal[
            "control_vector", b"control_vector", "input", b"input", "result", b"result"
        ],
    ) -> None: ...

global___Simulation = Simulation

@typing.final
class Simulations(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    SIMULATIONS_FIELD_NUMBER: builtins.int

    @property
    def simulations(
        self,
    ) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[
        global___Simulation
    ]: ...
    def __init__(
        self, *, simulations: collections.abc.Iterable[global___Simulation] | None = ...
    ) -> None: ...
    def ClearField(
        self, field_name: typing.Literal["simulations", b"simulations"]
    ) -> None: ...

global___Simulations = Simulations

@typing.final
class RequestModel(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    WORKER_ID_FIELD_NUMBER: builtins.int
    worker_id: builtins.str

    def __init__(self, *, worker_id: builtins.str = ...) -> None: ...
    def ClearField(
        self, field_name: typing.Literal["worker_id", b"worker_id"]
    ) -> None: ...

global___RequestModel = RequestModel

@typing.final
class SimulationModel(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    PACKAGE_ARCHIVE_FIELD_NUMBER: builtins.int
    STATUS_FIELD_NUMBER: builtins.int
    package_archive: builtins.bytes
    status: global___ModelStatus.ValueType

    def __init__(
        self,
        *,
        package_archive: builtins.bytes = ...,
        status: global___ModelStatus.ValueType = ...,
    ) -> None: ...
    def ClearField(
        self,
        field_name: typing.Literal[
            "package_archive", b"package_archive", "status", b"status"
        ],
    ) -> None: ...

global___SimulationModel = SimulationModel
