"""Generated protocol buffer code."""

from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC, 5, 29, 0, "", "simulation_messaging.proto"
)
_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x1asimulation_messaging.proto\x12\x14simulation_messaging" \n\x0fSimulationInput\x12\r\n\x05wells\x18\x01 \x01(\t"*\n\x17SimulationControlVector\x12\x0f\n\x07content\x18\x01 \x01(\t""\n\x10SimulationResult\x12\x0e\n\x06result\x18\x01 \x01(\t"\xcd\x01\n\rSimulationJob\x124\n\nsimulation\x18\x01 \x01(\x0b2 .simulation_messaging.Simulation\x12/\n\x06status\x18\x02 \x01(\x0e2\x1f.simulation_messaging.JobStatus\x12\x11\n\tworker_id\x18\x03 \x01(\t\x122\n\tsimulator\x18\x04 \x01(\x0e2\x1f.simulation_messaging.Simulator\x12\x0e\n\x06job_id\x18\x05 \x01(\x05"\x1f\n\nRequestJob\x12\x11\n\tworker_id\x18\x01 \x01(\t"\x16\n\x03Ack\x12\x0f\n\x07message\x18\x01 \x01(\t"\xc1\x01\n\nSimulation\x124\n\x05input\x18\x01 \x01(\x0b2%.simulation_messaging.SimulationInput\x126\n\x06result\x18\x02 \x01(\x0b2&.simulation_messaging.SimulationResult\x12E\n\x0econtrol_vector\x18\x03 \x01(\x0b2-.simulation_messaging.SimulationControlVector"D\n\x0bSimulations\x125\n\x0bsimulations\x18\x01 \x03(\x0b2 .simulation_messaging.Simulation"!\n\x0cRequestModel\x12\x11\n\tworker_id\x18\x01 \x01(\t"]\n\x0fSimulationModel\x12\x17\n\x0fpackage_archive\x18\x01 \x01(\x0c\x121\n\x06status\x18\x02 \x01(\x0e2!.simulation_messaging.ModelStatus*i\n\tJobStatus\x12\x19\n\x15JOBSTATUS_UNSPECIFIED\x10\x00\x12\x07\n\x03NEW\x10\x01\x12\x0b\n\x07SUCCESS\x10\x02\x12\n\n\x06FAILED\x10\x03\x12\x14\n\x10NO_JOB_AVAILABLE\x10\x04\x12\t\n\x05ERROR\x10\x05*5\n\tSimulator\x12\x19\n\x15SIMULATOR_UNSPECIFIED\x10\x00\x12\r\n\tOPENDARTS\x10\x01*Q\n\x0bModelStatus\x12\x1b\n\x17MODELSTATUS_UNSPECIFIED\x10\x00\x12\x16\n\x12NO_MODEL_AVAILABLE\x10\x01\x12\r\n\tON_SERVER\x10\x022\xe9\x03\n\x13SimulationMessaging\x12[\n\x17TransferSimulationModel\x12%.simulation_messaging.SimulationModel\x1a\x19.simulation_messaging.Ack\x12Z\n\x12PerformSimulations\x12!.simulation_messaging.Simulations\x1a!.simulation_messaging.Simulations\x12]\n\x14RequestSimulationJob\x12 .simulation_messaging.RequestJob\x1a#.simulation_messaging.SimulationJob\x12U\n\x13SubmitSimulationJob\x12#.simulation_messaging.SimulationJob\x1a\x19.simulation_messaging.Ack\x12c\n\x16RequestSimulationModel\x12".simulation_messaging.RequestModel\x1a%.simulation_messaging.SimulationModelb\x06proto3'
)
_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, "simulation_messaging_pb2", _globals
)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals["_JOBSTATUS"]._serialized_start = 827
    _globals["_JOBSTATUS"]._serialized_end = 932
    _globals["_SIMULATOR"]._serialized_start = 934
    _globals["_SIMULATOR"]._serialized_end = 987
    _globals["_MODELSTATUS"]._serialized_start = 989
    _globals["_MODELSTATUS"]._serialized_end = 1070
    _globals["_SIMULATIONINPUT"]._serialized_start = 52
    _globals["_SIMULATIONINPUT"]._serialized_end = 84
    _globals["_SIMULATIONCONTROLVECTOR"]._serialized_start = 86
    _globals["_SIMULATIONCONTROLVECTOR"]._serialized_end = 128
    _globals["_SIMULATIONRESULT"]._serialized_start = 130
    _globals["_SIMULATIONRESULT"]._serialized_end = 164
    _globals["_SIMULATIONJOB"]._serialized_start = 167
    _globals["_SIMULATIONJOB"]._serialized_end = 372
    _globals["_REQUESTJOB"]._serialized_start = 374
    _globals["_REQUESTJOB"]._serialized_end = 405
    _globals["_ACK"]._serialized_start = 407
    _globals["_ACK"]._serialized_end = 429
    _globals["_SIMULATION"]._serialized_start = 432
    _globals["_SIMULATION"]._serialized_end = 625
    _globals["_SIMULATIONS"]._serialized_start = 627
    _globals["_SIMULATIONS"]._serialized_end = 695
    _globals["_REQUESTMODEL"]._serialized_start = 697
    _globals["_REQUESTMODEL"]._serialized_end = 730
    _globals["_SIMULATIONMODEL"]._serialized_start = 732
    _globals["_SIMULATIONMODEL"]._serialized_end = 825
    _globals["_SIMULATIONMESSAGING"]._serialized_start = 1073
    _globals["_SIMULATIONMESSAGING"]._serialized_end = 1562
