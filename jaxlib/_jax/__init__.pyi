# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable, Iterator, Mapping, Sequence
import enum
import inspect
import traceback
import types
from typing import Any, overload

import numpy
import typing_extensions

import jaxlib._xla

from . import (
    config as config,
    ffi as ffi,
    guard_lib as guard_lib,
    hlo_sharding_util as hlo_sharding_util,
    ifrt_programs as ifrt_programs,
    jax_jit as jax_jit,
    mlir as mlir,
    pmap_lib as pmap_lib,
    pytree as pytree,
)
from .pmap_lib import PmapFunction as PmapFunction
from .pytree import (PyTreeDef as PyTreeDef, PyTreeRegistry as _PyTreeRegistry)

class JaxRuntimeError(RuntimeError):
  """Runtime errors thrown by the JAX runtime.

  While the JAX runtime may raise other exceptions as well, most exceptions
  thrown by the runtime are instances of this class.
  """

class Device:
  """A descriptor of an available device.

  Subclasses are used to represent specific types of devices, e.g. CPUs, GPUs.
  Subclasses may have additional properties specific to that device type.
  """

  @property
  def id(self) -> int:
    """Integer ID of this device.

    Unique across all available devices of this type, including remote devices
    on multi-host platforms.
    """

  @property
  def process_index(self) -> int:
    """Integer index of this device's process.

    This is always 0 except on multi-process platforms.
    """

  @property
  def host_id(self) -> int:
    """Deprecated; please use process_index"""

  @property
  def task_id(self) -> int:
    """Deprecated; please use process_index"""

  @property
  def platform(self) -> str: ...
  @property
  def device_kind(self) -> str: ...
  @property
  def client(self) -> Client: ...
  @property
  def local_hardware_id(self) -> int | None:
    """Opaque hardware ID, e.g., the CUDA device number.

    In general, not guaranteed to be dense, and not guaranteed to be defined on
    all platforms.
    """

  def __str__(self) -> str: ...
  def __repr__(self) -> str: ...
  def memory(self, kind: str) -> Memory: ...
  def default_memory(self) -> Memory:
    """Returns the default memory of a device."""

  def addressable_memories(self) -> list[Memory]:
    """Returns all the memories that a device can address."""

  def live_buffers(self) -> list: ...
  def memory_stats(self) -> dict[str, int] | None:
    """Returns memory statistics for this device keyed by name.

    May not be implemented on all platforms, and different platforms may return
    different stats, or -1 for unavailable stats. 'bytes_in_use' is usually
    available. Intended for diagnostic use.
    """

  def get_stream_for_external_ready_events(self) -> int: ...

  __getattr__: types.MethodDescriptorType = ...

class Memory:
  @property
  def process_index(self) -> int: ...
  @property
  def platform(self) -> str: ...
  @property
  def kind(self) -> str: ...
  def __str__(self) -> str: ...
  def __repr__(self) -> str: ...
  def addressable_by_devices(self) -> list[Device]:
    """Returns devices that can address this memory."""

class HostBufferSemantics(enum.Enum):
  IMMUTABLE_ONLY_DURING_CALL = 0

  IMMUTABLE_UNTIL_TRANSFER_COMPLETES = 1

  ZERO_COPY = 2

class Client:
  @property
  def platform(self) -> str: ...
  @property
  def _raw_platform(self) -> str: ...
  @property
  def platform_version(self) -> str: ...
  @property
  def runtime_type(self) -> str: ...
  def device_count(self) -> int: ...
  def local_device_count(self) -> int: ...
  def devices(self) -> list[Device]: ...
  def local_devices(self) -> list[Device]: ...
  def _get_all_devices(self) -> list[Device]: ...
  def device_from_local_hardware_id(self, arg: int, /) -> Device: ...
  def live_executables(self) -> list[LoadedExecutable]: ...
  def live_arrays(self) -> list[Array]: ...
  def live_buffers(self) -> list[Array]: ...
  def process_index(self) -> int: ...
  def host_id(self) -> int: ...
  def task_id(self) -> int: ...
  def buffer_from_pyval(
      self,
      argument: object,
      device: Device | None = ...,
      force_copy: bool = ...,
      host_buffer_semantics: HostBufferSemantics = HostBufferSemantics.ZERO_COPY,
  ) -> object: ...
  def compile(
      self,
      computation: object,
      executable_devices: DeviceList,
      compile_options: jaxlib._xla.CompileOptions = ...,
  ) -> Executable: ...
  @overload
  def compile_and_load(
      self,
      computation: object,
      executable_devices: DeviceList,
      compile_options: jaxlib._xla.CompileOptions = ...,
      host_callbacks: Sequence[typing_extensions.CapsuleType] = ...,
  ) -> LoadedExecutable: ...
  @overload
  def compile_and_load(
      self,
      computation: object,
      executable_devices: DeviceList,
      compile_options: jaxlib._xla.CompileOptions = ...,
      host_callbacks: Sequence[Callable[..., Any]] = ...,
  ) -> LoadedExecutable: ...
  @overload
  def compile_and_load(
      self,
      computation: bytes,
      executable_devices: Sequence,
      compile_options: jaxlib._xla.CompileOptions = ...,
  ) -> LoadedExecutable: ...
  @overload
  def compile_and_load(
      self,
      computation: str,
      executable_devices: Sequence,
      compile_options: jaxlib._xla.CompileOptions = ...,
  ) -> LoadedExecutable: ...
  def compile_ifrt_program(
      self, arg0: ifrt_programs.Program, arg1: ifrt_programs.CompileOptions, /
  ) -> LoadedExecutable: ...
  def compile_and_load_ifrt_program(
      self, arg0: ifrt_programs.Program, arg1: ifrt_programs.CompileOptions, /
  ) -> LoadedExecutable: ...
  def serialize_executable(self, arg: LoadedExecutable, /) -> bytes: ...
  @overload
  def deserialize_executable(
      self,
      serialized: bytes,
      executable_devices: DeviceList,
      compile_options: (
          jaxlib._xla.CompileOptions
          | None
      ) = ...,
      host_callbacks: Sequence[typing_extensions.CapsuleType] = ...,
  ) -> LoadedExecutable: ...
  @overload
  def deserialize_executable(
      self,
      serialized: bytes,
      executable_devices: DeviceList,
      compile_options: (
          jaxlib._xla.CompileOptions
          | None
      ) = ...,
      host_callbacks: Sequence[Callable] = ...,
  ) -> LoadedExecutable: ...
  @overload
  def deserialize_executable(
      self,
      serialized: bytes,
      executable_devices: Sequence,
      compile_options: (
          jaxlib._xla.CompileOptions
          | None
      ) = ...,
  ) -> LoadedExecutable: ...
  def heap_profile(self) -> bytes: ...
  def defragment(self) -> None: ...
  def make_python_callback_from_host_send_and_recv(
      self,
      callable: Callable,
      operand_shapes: Sequence[
          jaxlib._xla.Shape
      ],
      result_shapes: Sequence[
          jaxlib._xla.Shape
      ],
      send_channel_ids: Sequence[int],
      recv_channel_ids: Sequence[int],
      serializer: Callable | None = ...,
  ) -> object: ...
  def get_default_layout(
      self, dtype: numpy.dtype, shard_shape: Sequence, device: Device
  ) -> PjRtLayout: ...
  def __getattr__(self, arg: str, /) -> object: ...

class ArrayCopySemantics(enum.IntEnum):
  ALWAYS_COPY = 0

  REUSE_INPUT = 1

  DONATE_INPUT = 2

class PjRtLayout:
  def __str__(self) -> str: ...
  def __eq__(self, arg: object, /) -> bool: ...
  def __hash__(self) -> int: ...
  def _xla_layout(
      self,
  ) -> jaxlib._xla.Layout: ...
  def __getstate__(self) -> tuple: ...
  def __setstate__(self, arg: tuple, /) -> None: ...

class CpuCollectives:
  def Init(self) -> None: ...
  def Finalize(self) -> None: ...

def make_gloo_tcp_collectives(
    distributed_client: DistributedRuntimeClient,
    hostname: str | None = ...,
    interface: str | None = ...,
) -> CpuCollectives: ...
def make_mpi_collectives() -> CpuCollectives: ...
def get_tfrt_cpu_client(
    asynchronous: bool = ...,
    distributed_client: DistributedRuntimeClient | None = ...,
    node_id: int = ...,
    num_nodes: int = ...,
    collectives: CpuCollectives | None = ...,
    num_devices: int | None = ...,
    get_local_topology_timeout_minutes: int | None = ...,
    get_global_topology_timeout_minutes: int | None = ...,
    transfer_server_factory: TransferServerInterfaceFactory | None = ...,
) -> Client: ...
def pjrt_plugin_loaded(arg: str, /) -> bool: ...
def load_pjrt_plugin(
    platform_name: str,
    library_path: str | None = ...,
    c_api: typing_extensions.CapsuleType | None = ...,
) -> typing_extensions.CapsuleType: ...
def pjrt_plugin_initialized(arg: str, /) -> bool: ...
def initialize_pjrt_plugin(arg: str, /) -> None: ...
def get_c_api_client(
    platform_name: str,
    options: Mapping[str, str | bool | int | Sequence[int] | float] = ...,
    distributed_client: DistributedRuntimeClient | None = ...,
    transfer_server_factory: TransferServerInterfaceFactory | None = ...,
) -> Client: ...
def get_default_c_api_topology(
    arg0: str,
    arg1: str,
    arg2: Mapping[str, str | bool | int | Sequence[int] | float],
    /,
) -> DeviceTopology: ...
def get_c_api_topology(
    arg0: typing_extensions.CapsuleType,
    arg1: str,
    arg2: Mapping[str, str | bool | int | Sequence[int] | float],
    /,
) -> DeviceTopology: ...
def get_topology_for_devices(arg: Sequence[Device], /) -> DeviceTopology: ...

class ArrayMeta(type):
  def __instancecheck__(self, x: object | None) -> bool: ...

Array: Any

def set_tracer_class(arg: object, /) -> None: ...

ArrayImpl: Any

def batched_copy_array_to_devices_with_sharding(
    arg0: Sequence[Array],
    arg1: Sequence[DeviceList],
    arg2: Sequence[object],
    arg3: Sequence[ArrayCopySemantics],
    /,
) -> list[Array]: ...
def array_result_handler(
    aval: object, sharding: object, committed: bool, _skip_checks: bool = ...
) -> ResultHandler: ...

class ResultHandler:
  def __call__(self, arg: Array | Sequence[Array], /) -> Array: ...

class DeviceList:
  def __init__(self, arg: tuple[Device, ...], /) -> None: ...
  def __hash__(self) -> int: ...
  def __eq__(self, arg: object, /) -> bool: ...
  def __ne__(self, arg: object, /) -> bool: ...
  def __len__(self) -> int: ...
  @overload
  def __getitem__(self, index: int, /) -> Device: ...
  @overload
  def __getitem__(self, slice: slice, /) -> Sequence[Device]: ...
  def __iter__(self) -> Iterator: ...
  def __str__(self) -> str: ...
  def __repr__(self) -> str: ...
  def __getstate__(self) -> tuple: ...
  def __setstate__(self, arg: tuple, /) -> None: ...
  @property
  def is_fully_addressable(self) -> bool: ...
  @property
  def addressable_device_list(self) -> DeviceList: ...
  @property
  def process_indices(self) -> set[int]: ...
  @property
  def default_memory_kind(self) -> str | None: ...
  @property
  def memory_kinds(self) -> tuple[str, ...]: ...
  @property
  def device_kind(self) -> str: ...

class Sharding:
  def __init__(self) -> None: ...

class NamedSharding(Sharding):
  def __init__(
      self,
      mesh: object,
      spec: PartitionSpec,
      memory_kind: object | None = ...,
      _logical_device_ids: object | None = ...,
  ) -> None: ...
  @property
  def mesh(self) -> object: ...
  @property
  def spec(self) -> PartitionSpec: ...
  @property
  def _memory_kind(self) -> object: ...
  @property
  def _logical_device_ids(self) -> object: ...
  @property
  def _internal_device_list(self) -> DeviceList: ...
  def __eq__(self, arg: object) -> bool: ...
  def __hash__(self) -> int: ...

class SingleDeviceSharding(Sharding):
  def __init__(
      self, device: object, memory_kind: object | None = ...
  ) -> None: ...
  @property
  def _device(self) -> object: ...
  @property
  def _memory_kind(self) -> object: ...
  @property
  def _internal_device_list(self) -> DeviceList: ...

class PmapSharding(Sharding):
  def __init__(
      self, devices: object, sharding_spec: pmap_lib.ShardingSpec
  ) -> None: ...
  @property
  def devices(self) -> numpy.ndarray: ...
  @property
  def sharding_spec(self) -> pmap_lib.ShardingSpec: ...
  @property
  def _internal_device_list(self) -> DeviceList: ...

class GSPMDSharding(Sharding):
  @overload
  def __init__(
      self,
      devices: DeviceList,
      op_sharding: jaxlib._xla.OpSharding,
      memory_kind: object | None = ...,
  ) -> None: ...
  @overload
  def __init__(
      self,
      devices: DeviceList,
      op_sharding: jaxlib._xla.HloSharding,
      memory_kind: object | None = ...,
  ) -> None: ...
  @overload
  def __init__(
      self,
      devices: Sequence[Device],
      op_sharding: jaxlib._xla.OpSharding,
      memory_kind: object | None = ...,
  ) -> None: ...
  @overload
  def __init__(
      self,
      devices: Sequence[Device],
      op_sharding: jaxlib._xla.HloSharding,
      memory_kind: object | None = ...,
  ) -> None: ...
  @property
  def _devices(self) -> DeviceList: ...
  @property
  def _hlo_sharding(
      self,
  ) -> jaxlib._xla.HloSharding: ...
  @property
  def _memory_kind(self) -> object: ...
  @property
  def _internal_device_list(self) -> DeviceList: ...

class CompiledMemoryStats:
  @property
  def generated_code_size_in_bytes(self) -> int: ...
  @generated_code_size_in_bytes.setter
  def generated_code_size_in_bytes(self, arg: int, /) -> None: ...
  @property
  def argument_size_in_bytes(self) -> int: ...
  @argument_size_in_bytes.setter
  def argument_size_in_bytes(self, arg: int, /) -> None: ...
  @property
  def output_size_in_bytes(self) -> int: ...
  @output_size_in_bytes.setter
  def output_size_in_bytes(self, arg: int, /) -> None: ...
  @property
  def alias_size_in_bytes(self) -> int: ...
  @alias_size_in_bytes.setter
  def alias_size_in_bytes(self, arg: int, /) -> None: ...
  @property
  def temp_size_in_bytes(self) -> int: ...
  @temp_size_in_bytes.setter
  def temp_size_in_bytes(self, arg: int, /) -> None: ...
  @property
  def host_generated_code_size_in_bytes(self) -> int: ...
  @host_generated_code_size_in_bytes.setter
  def host_generated_code_size_in_bytes(self, arg: int, /) -> None: ...
  @property
  def host_argument_size_in_bytes(self) -> int: ...
  @host_argument_size_in_bytes.setter
  def host_argument_size_in_bytes(self, arg: int, /) -> None: ...
  @property
  def host_output_size_in_bytes(self) -> int: ...
  @host_output_size_in_bytes.setter
  def host_output_size_in_bytes(self, arg: int, /) -> None: ...
  @property
  def host_alias_size_in_bytes(self) -> int: ...
  @host_alias_size_in_bytes.setter
  def host_alias_size_in_bytes(self, arg: int, /) -> None: ...
  @property
  def host_temp_size_in_bytes(self) -> int: ...
  @host_temp_size_in_bytes.setter
  def host_temp_size_in_bytes(self, arg: int, /) -> None: ...
  @property
  def serialized_buffer_assignment_proto(self) -> bytes: ...
  @property
  def peak_memory_in_bytes(self) -> int: ...
  @peak_memory_in_bytes.setter
  def peak_memory_in_bytes(self, arg: int, /) -> None: ...
  def __str__(self) -> str: ...

def get_execution_stream_id() -> int: ...
def set_execution_stream_id(arg: int, /) -> None: ...

class LoadedExecutable:
  @property
  def client(self) -> Client: ...
  def local_devices(self) -> list[Device]: ...
  def get_hlo_text(self) -> str: ...
  def size_of_generated_code_in_bytes(self) -> int: ...
  def get_compiled_memory_stats(self) -> CompiledMemoryStats: ...
  def execute_sharded(
      self, arguments: Sequence[Array], with_tokens: bool = ...
  ) -> ExecuteResults: ...
  def hlo_modules(
      self,
  ) -> list[
      jaxlib._xla.HloModule
  ]: ...
  def get_output_memory_kinds(self) -> list[list[str]]: ...
  def get_output_shardings(
      self,
  ) -> (
      list[jaxlib._xla.OpSharding]
      | None
  ): ...
  def get_parameter_layouts(self) -> list[PjRtLayout]: ...
  def get_output_layouts(self) -> list[PjRtLayout]: ...
  def get_parameter_shardings(
      self,
  ) -> (
      list[jaxlib._xla.OpSharding]
      | None
  ): ...
  def keep_alive(self, arg: object, /) -> None: ...
  def cost_analysis(
      self,
  ) -> dict[str, str | bool | int | list[int] | float]: ...
  @property
  def traceback(self) -> Traceback | None: ...
  @property
  def fingerprint(self) -> object: ...

class ExecuteResults:
  def __len__(self) -> int: ...
  def disassemble_into_single_device_arrays(self) -> list[list[Array]]: ...
  def disassemble_prefix_into_single_device_arrays(
      self, arg: int, /
  ) -> list[list[Array]]: ...
  def consume_with_handlers(
      self, arg: Sequence[ResultHandler | object], /
  ) -> list[object]: ...
  def consume_token(self) -> ShardedToken: ...

class Token:
  def block_until_ready(self) -> None: ...

class ShardedToken:
  def block_until_ready(self) -> None: ...
  def get_token(self, arg: int, /) -> Token: ...

class Executable:
  def hlo_modules(
      self,
  ) -> list[
      jaxlib._xla.HloModule
  ]: ...
  def get_output_memory_kinds(self) -> list[list[str]]: ...
  def get_output_shardings(
      self,
  ) -> (
      list[jaxlib._xla.OpSharding]
      | None
  ): ...
  def get_parameter_layouts(self) -> list[PjRtLayout]: ...
  def get_output_layouts(self) -> list[PjRtLayout]: ...
  def get_parameter_shardings(
      self,
  ) -> (
      list[jaxlib._xla.OpSharding]
      | None
  ): ...
  def get_compiled_memory_stats(self) -> CompiledMemoryStats: ...
  def serialize(self) -> bytes: ...
  def cost_analysis(
      self,
  ) -> dict[str, str | bool | int | list[int] | float]: ...

def buffer_to_dlpack_managed_tensor(
    buffer: object, stream: int | None = ...
) -> typing_extensions.CapsuleType: ...
def dlpack_managed_tensor_to_buffer(
    dlpack: typing_extensions.CapsuleType,
    device: Device,
    stream: int | None,
    copy: bool | None = ...,
) -> ArrayImpl: ...
def cuda_array_interface_to_buffer(
    cai: dict, gpu_backend: Client | None = ..., device_id: int | None = ...
) -> object: ...

class RuntimeTracebackMode(enum.Enum):
  OFF = 0

  ON = 1

  FULL = 2

def add_exclude_path(arg: str, /) -> None:
  """Adds a path to exclude from tracebacks."""

def set_send_traceback_to_runtime_global(
    arg: RuntimeTracebackMode, /
) -> None: ...
def set_send_traceback_to_runtime_thread_local(
    mode: RuntimeTracebackMode | None,
) -> None: ...

class PjitFunctionCache:
  def __init__(self, capacity: int = ...) -> None: ...
  def size(self) -> int: ...
  def capacity(self) -> int: ...
  def clear(self) -> None: ...
  @staticmethod
  def clear_all() -> None: ...
  def __getstate__(self) -> dict: ...
  def __setstate__(self, arg: dict, /) -> None: ...

class PjitFunction:
  def __repr__(self, /):
    """Return repr(self)."""

  def __call__(self, /, *args, **kwargs):
    """Call self as a function."""

  def __get__(self, instance, owner=..., /):
    """Return an attribute of instance, which is of type owner."""
  __vectorcalloffset__: types.MemberDescriptorType = ...

  def __getstate__(self) -> dict: ...
  def __setstate__(self, arg: dict, /) -> None: ...
  @property
  def __signature__(self) -> inspect.Signature: ...
  @property
  def _cache_miss(self) -> Callable: ...
  def _cache_size(self) -> int: ...
  def _clear_cache(self) -> None: ...

def pjit(
    function_name: str,
    fun: Callable[..., Any] | None,
    cache_miss: Callable[..., Any],
    static_argnums: Sequence[int],
    static_argnames: Sequence[str],
    global_cache_key: Any,
    pytree_registry: _PyTreeRegistry,
    shard_arg_fallback: Callable[..., Any],
    cache: PjitFunctionCache | None = ...,
) -> PjitFunction: ...

class Frame:
  def __init__(self, arg0: str, arg1: str, arg2: int, arg3: int, /) -> None: ...
  @property
  def file_name(self) -> str: ...
  @property
  def function_name(self) -> str: ...
  @property
  def function_start_line(self) -> int: ...
  @property
  def line_num(self) -> int: ...
  def __repr__(self) -> str: ...

class Traceback:
  def __hash__(self, /):
    """Return hash(self)."""

  def __str__(self, /):
    """Return str(self)."""

  def __lt__(self, value, /):
    """Return self<value."""

  def __le__(self, value, /):
    """Return self<=value."""

  def __eq__(self, value, /):
    """Return self==value."""

  def __ne__(self, value, /):
    """Return self!=value."""

  def __gt__(self, value, /):
    """Return self>value."""

  def __ge__(self, value, /):
    """Return self>=value."""

  @staticmethod
  def get_traceback() -> Traceback | None:
    """Returns a :class:`Traceback` for the current thread.

    If ``Traceback.enabled`` is ``True``, returns a :class:`Traceback`
    object that describes the Python stack of the calling thread. Stack
    trace collection has a small overhead, so it is disabled by default. If
    traceback collection is disabled, returns ``None``.
    """

  @property
  def frames(self) -> list[Frame]: ...
  def raw_frames(self) -> tuple[list[types.CodeType], list[int]]: ...
  def as_python_traceback(self) -> traceback.TracebackType: ...
  @staticmethod
  def traceback_from_frames(frames: list[Frame]) -> traceback.TracebackType:
    """Creates a traceback from a list of frames."""

  @staticmethod
  def code_addr2line(code: types.CodeType, lasti: int) -> int:
    """Python wrapper around the Python C API function PyCode_Addr2Line"""

  @staticmethod
  def code_addr2location(
      code: types.CodeType, lasti: int
  ) -> tuple[int, int, int, int]:
    """Python wrapper around the Python C API function PyCode_Addr2Location"""

def tracebacks_enabled() -> bool: ...
def set_tracebacks_enabled(arg: bool, /) -> None: ...
def register_custom_call_partitioner(
    name: str,
    prop_user_sharding: object,
    partition: object,
    infer_sharding_from_operands: object,
    can_side_effecting_have_replicated_sharding: bool = ...,
    c_api: typing_extensions.CapsuleType | None = ...,
) -> None:
  """Registers a partitioner for a custom-call operation.

  Args:
    name: custom_call_target to match.
    prop_user_sharding: Custom backwards sharding propagation rule. Takes result
      sharding and returns the instruction sharding.
    partition: Lowering rule. Takes operand and result shardings and returns a
      generated HLO and sharding specs. The spmd lowerer first reshards to match
      the returned sharding specs and then inserts the generated hlo.
    infer_sharding_from_operands: Custom forwards sharding propagation rule.
      Takes operand sharding and returns the instruction sharding.
    can_side_effecting_have_replicated_sharding: Side effecting ops are not
      allowed to have replicated sharding. Pass true to disable this check.
    c_api: Optional `PJRT_Api*` if it is called with a plugin. This is safe to
      call on plugins that do not implement the custom partitioner extension
  """

def encode_inspect_sharding_callback(arg: object, /) -> bytes: ...
def register_custom_call_as_batch_partitionable(
    target_name: str, c_api: typing_extensions.CapsuleType | None = ...
) -> None:
  """Registers a custom call as batch partitionable.

  If a custom call is "batch partitionable", it means that it can be trivially
  partitioned on some number of (leading) dimensions, with the same call being
  executed independently on each shard of data. If the data are sharded on
  non-batch dimensions, partitioning will re-shard the data to be replicated on
  the non-batch dimensions.

  Args:
    target_name: the target name of the batch partitionable custom call.
    c_api: optional `PJRT_Api*` to support registration via a PJRT plugin.
  """

def register_custom_call_target(
    fn_name: object,
    fn: object,
    platform: str,
    api_version: int = ...,
    traits: int = ...,
) -> None: ...
def custom_call_targets(platform: str) -> dict: ...
def register_custom_type(type_name: str, type_id: object) -> None: ...

class TransferConnection:
  def _testonly_inject_failure(self) -> None: ...
  def _poison_connection(self) -> None: ...
  def _pull_flat(
      self, arg0: int, arg1: Client, arg2: Sequence[object], /
  ) -> list[Array]: ...
  def _pull_into_flat(
      self, arg0: int, arg1: Sequence[Array], arg2: Sequence[slice], /
  ) -> list[Token]: ...

class TransferServer:
  def address(self) -> str: ...
  def _await_pull_flat(self, arg0: int, arg1: Sequence[Array], /) -> None: ...
  def _reset_rendevous_table(self) -> None: ...
  def connect(self, arg: str, /) -> TransferConnection: ...

def _make_error_array(arg0: Client, arg1: object, arg2: str, /) -> Array: ...
def start_transfer_server(
    client: Client,
    address: str = ...,
    transport_addresses: Sequence[str] = ...,
    max_num_parallel_copies: int = ...,
    transfer_size: int = ...,
    supports_pinned_allocator: bool = ...,
    use_raw_buffers: bool = ...,
) -> TransferServer: ...
def make_transfer_server_interface_factory(
    transfer_size: int = ...,
    cross_host_transfer_timeout_seconds: int = ...,
    distributed_client: DistributedRuntimeClient | None = ...,
    socket_address: str = ...,
    transport_addresses: Sequence[str] = ...,
) -> TransferServerInterfaceFactory: ...

class PreemptionSyncManager:
  def initialize(
      self, distributed_client: DistributedRuntimeClient
  ) -> None: ...
  def reached_sync_point(self, arg: int, /) -> bool: ...
  def shutdown(self) -> None: ...

def create_preemption_sync_manager() -> PreemptionSyncManager: ...

class DistributedRuntimeService:
  def shutdown(self) -> None: ...

class DistributedRuntimeClient:
  def connect(self) -> None: ...
  def shutdown(self) -> None: ...
  def blocking_key_value_get(self, key: str, timeout_in_ms: int) -> str: ...
  def blocking_key_value_get_bytes(
      self, key: str, timeout_in_ms: int
  ) -> bytes: ...
  def key_value_try_get(self, key: str) -> str: ...
  def key_value_try_get_bytes(self, key: str) -> bytes: ...
  def key_value_increment(self, key: str, increment: int) -> int: ...
  def wait_at_barrier(
      self,
      barrier_id: str,
      timeout_in_ms: int,
      process_ids: Sequence[int] | None = ...,
  ) -> None: ...
  def get_live_nodes(self, process_ids: Sequence[int]) -> dict[int, int]: ...
  def key_value_set(
      self, key: str, value: str, allow_overwrite: bool = ...
  ) -> None: ...
  def key_value_set_bytes(
      self, key: str, value: bytes, allow_overwrite: bool = ...
  ) -> None: ...
  def key_value_dir_get(self, key: str) -> list[tuple[str, str]]: ...
  def key_value_dir_get_bytes(self, key: str) -> list[tuple[str, bytes]]: ...
  def key_value_delete(self, key: str) -> None: ...

def get_distributed_runtime_service(
    address: str,
    num_nodes: int,
    heartbeat_timeout: int | None = ...,
    cluster_register_timeout: int | None = ...,
    shutdown_timeout: int | None = ...,
) -> DistributedRuntimeService: ...
def get_distributed_runtime_client(
    address: str,
    node_id: int,
    rpc_timeout: int | None = ...,
    init_timeout: int | None = ...,
    shutdown_timeout: int | None = ...,
    heartbeat_timeout: int | None = ...,
    missed_heartbeat_callback: Callable | None = ...,
    shutdown_on_destruction: bool | None = ...,
    use_compression: bool | None = ...,
    recoverable: bool | None = ...,
) -> DistributedRuntimeClient: ...
def collect_garbage() -> None: ...
def is_optimized_build() -> bool: ...
def json_to_pprof_profile(arg: str, /) -> bytes:
  """Encodes the JSON representation of a pprof Profile into its binary protocol buffer encoding."""

def pprof_profile_to_json(arg: bytes, /) -> str:
  """Decodes an uncompressed pprof Profile protocol buffer into a JSON representation"""

class CompileOnlyPyClient(Client):
  def compile(
      self,
      computation: object,
      executable_devices: DeviceList,
      compile_options: jaxlib._xla.CompileOptions = ...,
      host_callbacks: Sequence[typing_extensions.CapsuleType] = ...,
  ) -> Executable: ...

class DeviceTopology:
  def _make_compile_only_devices(self) -> list[Device]: ...
  @property
  def platform(self) -> str: ...
  @property
  def platform_version(self) -> str: ...
  def serialize(self) -> bytes: ...
  def __getattr__(self, arg: str, /) -> object: ...

class TransferServerInterfaceFactory:
  pass

def is_asan() -> bool: ...
def is_msan() -> bool: ...
def is_tsan() -> bool: ...
def is_sanitized() -> bool: ...
def batched_device_put(
    aval: object,
    sharding: object,
    xs: Sequence[object],
    devices: Sequence[Device],
    committed: bool = ...,
    force_copy: bool = ...,
    host_buffer_semantics: HostBufferSemantics = HostBufferSemantics.ZERO_COPY,
    enable_x64: bool | None = ...,
) -> object: ...
def reorder_shards(
    x: Array, dst_sharding: object, array_copy_semantics: ArrayCopySemantics
) -> Array: ...
def batched_block_until_ready(arg: Sequence[object], /) -> None: ...
def check_and_canonicalize_memory_kind(
    memory_kind: object | None, device_list: DeviceList
) -> object: ...

ifrt_version_number: int = ...

def approx_top_k_reduction_output_size(
    input_size: int,
    rank: int,
    top_k: int,
    recall_target: float,
    aggregate_to_topk: bool = ...,
    input_size_override: int = ...,
) -> tuple[int, int]: ...
def get_internal_device_put_info() -> dict[str, int]: ...

class UnconstrainedSingleton:
  def __repr__(self) -> str: ...
  def __reduce__(self) -> str: ...

UNCONSTRAINED_PARTITION: UnconstrainedSingleton = ...

def canonicalize_partition(arg: object, /) -> object: ...

class PartitionSpec(Any):
  def __init__(
      self, *partitions, unreduced: object = ..., reduced: object = ...
  ) -> None: ...
  @property
  def _partitions(self) -> tuple: ...
  @property
  def unreduced(self) -> frozenset: ...
  @property
  def reduced(self) -> frozenset: ...
  def __eq__(self, arg: object) -> bool: ...
  def __hash__(self) -> int: ...

def set_typed_int_type(arg: object, /) -> None: ...
def set_typed_float_type(arg: object, /) -> None: ...
def set_typed_complex_type(arg: object, /) -> None: ...
def set_typed_ndarray_type(arg: object, /) -> None: ...
def hlo_module_cost_analysis(
    arg0: Client,
    arg1: jaxlib._xla.HloModule,
    /,
) -> dict: ...
