# Copyright 2021 The JAX Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import enum
from typing import Any, Union

from jaxlib import _jax as _xla
from jaxlib._jax import ArrayCopySemantics as ArrayCopySemantics
from jaxlib._jax import ArrayImpl as ArrayImpl
from jaxlib._jax import AutotuneCacheMode as AutotuneCacheMode
from jaxlib._jax import Client as Client
from jaxlib._jax import CompileOptions as CompileOptions
from jaxlib._jax import Device as Device
from jaxlib._jax import DeviceAssignment as DeviceAssignment
from jaxlib._jax import DeviceList as DeviceList
from jaxlib._jax import DeviceTopology as DeviceTopology
from jaxlib._jax import DistributedRuntimeClient as DistributedRuntimeClient
from jaxlib._jax import Frame as Frame
from jaxlib._jax import GSPMDSharding as GSPMDSharding
from jaxlib._jax import HloSharding as HloSharding
from jaxlib._jax import HostBufferSemantics as HostBufferSemantics
from jaxlib._jax import ifrt_programs as ifrt_programs
from jaxlib._jax import Layout as Layout
from jaxlib._jax import LoadedExecutable as LoadedExecutable
from jaxlib._jax import Executable as Executable
from jaxlib._jax import Memory as Memory
from jaxlib._jax import NamedSharding as NamedSharding
from jaxlib._jax import OpSharding as OpSharding
from jaxlib._jax import PjRtLayout as PjRtLayout
from jaxlib._jax import PmapSharding as PmapSharding
from jaxlib._jax import PrimitiveType as PrimitiveType
from jaxlib._jax import Shape as Shape
from jaxlib._jax import Sharding as Sharding
from jaxlib._jax import SingleDeviceSharding as SingleDeviceSharding
from jaxlib._jax import Traceback as Traceback
from jaxlib._jax import XlaComputation as XlaComputation

_version: int
_ifrt_version: int

_NameValueMapping = Mapping[str, Union[str, int, list[int], float, bool]]

XlaRuntimeError = _xla.XlaRuntimeError

def make_cpu_client(
    asynchronous: bool = ...,
    distributed_client: DistributedRuntimeClient | None = ...,
    node_id: int = ...,
    num_nodes: int = ...,
    collectives: _xla.CpuCollectives | None = ...,
    num_devices: int | None = ...,
) -> Client: ...
def make_gpu_client(
    distributed_client: DistributedRuntimeClient | None = ...,
    node_id: int = ...,
    num_nodes: int = ...,
    platform_name: str | None = ...,
    allowed_devices: set[int] | None = ...,
    mock: bool | None = ...,
    mock_gpu_topology: str | None = ...,
) -> Client: ...
def make_tfrt_tpu_c_api_device_topology(
    topology_name: str | None = None, **kwargs
) -> DeviceTopology: ...
def make_c_api_device_topology(
    c_api: Any, topology_name: str = '', **kwargs
) -> DeviceTopology: ...
def get_topology_for_devices(devices: list[Device]) -> DeviceTopology: ...
def make_c_api_client(
    plugin_name: str,
    options: _NameValueMapping | None = None,
    distributed_client: DistributedRuntimeClient | None = None,
) -> Client: ...
def pjrt_plugin_loaded(plugin_name: str) -> bool: ...
def load_pjrt_plugin_dynamically(
    plugin_name: str, library_path: str
) -> Any: ...
def load_pjrt_plugin_with_c_api(plugin_name: str, c_api: Any) -> None: ...
def pjrt_plugin_initialized(plugin_name: str) -> bool: ...
def initialize_pjrt_plugin(plugin_name: str) -> None: ...
def generate_pjrt_gpu_plugin_options() -> _NameValueMapping: ...
def batched_copy_array_to_devices_with_sharding(
    arrays: Sequence[ArrayImpl],
    devices: Sequence[list[Device]],
    sharding: Sequence[Any],
    array_copy_semantics: Sequence[ArrayCopySemantics],
) -> Sequence[ArrayImpl]: ...
def batched_device_put(
    aval: Any,
    sharding: Any,
    shards: Sequence[Any],
    devices: list[Device],
    committed: bool = ...,
    force_copy: bool = ...,
    host_buffer_semantics: Any = ...,
) -> ArrayImpl: ...
def reorder_shards(
    x: ArrayImpl,
    dst_sharding: Any,
    array_copy_semantics: ArrayCopySemantics,
) -> ArrayImpl: ...
def batched_block_until_ready(x: Sequence[ArrayImpl]) -> None: ...
def check_and_canonicalize_memory_kind(
    memory_kind: str | None, device_list: DeviceList
) -> str | None: ...
def array_result_handler(
    aval: Any, sharding: Any, committed: bool, _skip_checks: bool = ...
) -> Callable: ...

class CustomCallTargetTraits(enum.IntFlag):
  DEFAULT = 0
  COMMAND_BUFFER_COMPATIBLE = 1

def register_custom_call_target(
    name: str,
    fn: Any,
    platform: str = ...,
    api_version: int = ...,
    traits: CustomCallTargetTraits = ...,
) -> None: ...
def register_custom_call_handler(
    xla_platform_name: str, handler: Any
) -> None: ...
def custom_call_targets(platform: str) -> dict[str, Any]: ...
def register_custom_type_id(
    type_name: str,
    type_id: Any,
    platform: str = ...,
) -> None: ...
def register_custom_type_id_handler(platform: str, handler: Any) -> None: ...
def encode_inspect_sharding_callback(handler: Any) -> bytes: ...

register_custom_call_partitioner = _xla.register_custom_call_partitioner
register_custom_call_as_batch_partitionable = (
    _xla.register_custom_call_as_batch_partitionable
)
