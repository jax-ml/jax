# Copyright 2021 Google LLC
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

"""Experimental IREE backend for JAX.

This backend is quite incomplete, but exists to allow experimenting with
using IREE to compile and run JAX computations instead of XLA.
"""

# pytype: skip-file

from __future__ import annotations

from typing import Any, List, Sequence, Optional

import iree.compiler
from iree import runtime as iree_runtime

from jax._src.lib import xla_client
import numpy as np


class IreeDevice:

  def __init__(self, client):
    self.id = 0
    self.host_id = 0
    self.process_index = 0
    self.platform = "iree"
    self.device_kind = "IREE device"
    self.client = client

  def __str__(self) -> str:
    return "IreeDevice"

  def transfer_to_infeed(self, literal: Any):
    raise NotImplementedError("transfer_to_infeed")

  def transfer_from_outfeed(self, shape: xla_client.Shape):
    raise NotImplementedError("transfer_to_outfeed")

  def live_buffers(self) -> List[IreeBuffer]:
    raise NotImplementedError("live_buffers")


class IreeBuffer(xla_client.DeviceArrayBase):

  def __init__(self, client, device, npy_value):
    self.client = client
    self._device = device
    assert device is not None
    self._npy_value = np.asarray(npy_value)

  def copy_to_device(self, device):
    return self

  def to_py(self) -> np.ndarray:
    return self._npy_value

  def to_iree(self):
    return self._npy_value

  def platform(self):
    return self.client.platform

  def device(self):
    return self._device

  def block_until_ready(self) -> IreeBuffer:
    return self  # no async

class IreeExecutable:

  def __init__(self, client, devices, module_object, function_name):
    self.client = client
    self.traceback = None
    self.fingerprint = None
    self._devices = devices
    self.module_object = module_object
    self.function_name = function_name

  def local_devices(self) -> List[IreeDevice]:
    return self._devices

  def execute(self, arguments: Sequence[IreeBuffer]) -> List[IreeBuffer]:
    inputs = [arg.to_iree() for arg in arguments]
    outputs = self.module_object[self.function_name](*inputs)
    # TODO(phawkins): Have a way to just have it always return the list,
    # regardless of arity.
    if not isinstance(outputs, list) and not isinstance(outputs, tuple):
      outputs = [outputs]
    return [
        IreeBuffer(self.client, self._devices[0], output) for output in outputs
    ]


class IreeClient:

  def __init__(self,
               *,
               compile_target_backends: Sequence[str] = ("cpu",),
               runtime_driver: str = "dylib"):
    self.platform = "iree"
    self.platform_version = "0.0.1"
    self.runtime_type = "iree"
    self.iree_config = iree_runtime.system_api.Config(runtime_driver)
    self._devices = [IreeDevice(self)]

  def process_index(self) -> int:
    return 0

  def device_count(self) -> int:
    return len(self._devices)

  def devices(self) -> List[IreeDevice]:
    return self._devices

  def local_devices(self) -> List[IreeDevice]:
    return self._devices

  def local_device_count(self) -> int:
    return len(self._devices)

  def get_default_device_assignment(
      self,
      num_replicas: int) -> List[IreeDevice]:
    if num_replicas != 1:
      raise NotImplementedError("Only single-device computations implemented")
    return [self._devices[0]]


  def compile(self, computation: str,
              compile_options: xla_client.CompileOptions) -> IreeExecutable:
    del compile_options  # Ignored.
    iree_binary = iree.compiler.compile_str(
        computation, target_backends=["dylib"], input_type="mhlo",
        # extra_args=["--print-ir-after-all"],
        # extended_diagnostics=True,
    )
    # Load it into the runtime.
    vm_module = iree_runtime.binding.VmModule.from_flatbuffer(iree_binary)
    module_object = iree_runtime.load_vm_module(vm_module, self.iree_config)
    return IreeExecutable(self, self._devices, module_object, "main")

  def buffer_from_pyval(
      self,
      argument: Any,
      device: Optional[IreeDevice],
      force_copy: bool = True,
      host_buffer_semantics: xla_client.HostBufferSemantics = xla_client
      .HostBufferSemantics.ZERO_COPY
  ) -> IreeBuffer:
    # TODO(phawkins): IREE's python API will accept a numpy array directly but
    # may want to explicitly construct a lower level BufferView to avoid copies.
    if device is None:
      assert type(argument) is np.ndarray
      device = self._devices[0]
    return IreeBuffer(self, device, np.array(argument, copy=True))


def iree_client_factory():
  return IreeClient()
