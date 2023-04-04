# Copyright 2021 The JAX Authors.
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

import os
import platform
from typing import Any, List, Sequence, Optional

import iree.compiler
import iree.runtime

from jax._src.config import flags
from jax._src.lib import xla_client
import numpy as np

FLAGS = flags.FLAGS


flags.DEFINE_string(
    'jax_iree_backend', os.getenv('JAX_IREE_BACKEND', 'cpu'),
    'IREE compiler backend to use.')

iree_compiler_map = {
  "cpu" : "llvm-cpu",
  "cuda" : "cuda",
  "vmvx" : "vmvx",
  "vulkan" : "vulkan-spirv"
}

iree_runtime_map = {
  "cpu" : "local-task",
  "cuda" : "cuda",
  "vmvx" : "local-task",
  "vulkan" : "vulkan"
}

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


class IreeBuffer:

  def __init__(self, client, device, buffer):
    self.client = client
    self._device = device
    assert device is not None
    self._buffer = buffer

  def copy_to_device(self, device):
    return self

  def __array__(self, dtype=None, context=None):
    return np.asarray(self._buffer)

  def to_iree(self):
    return self._buffer

  def platform(self):
    return self.client.platform

  def device(self):
    return self._device

  def block_until_ready(self) -> IreeBuffer:
    return self  # no async

  # overrides repr on base class which expects _value and aval attributes
  def __repr__(self): return f'IreeBuffer({np.asarray(self)})'

  @property
  def _value(self):
    return np.asarray(self)

  def copy_to_host_async(self):
    return self

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
               iree_backend: Optional[str] = None):
    self.platform = "iree"
    self.platform_version = "0.0.1"
    self.runtime_type = "iree"
    self.iree_backend = (FLAGS.jax_iree_backend if iree_backend is None
                        else iree_backend)
    self.compiler_driver = iree_compiler_map[self.iree_backend]
    self.runtime_driver = iree_runtime_map[self.iree_backend]
    self.iree_config = iree.runtime.system_api.Config(self.runtime_driver)
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
    extra_args = []
    # extra_args=["--mlir-print-ir-after-all"]
    if platform.system() == "Darwin" and platform.machine() == "arm64":
      extra_args += ["--iree-llvm-target-triple=arm64-apple-darwin21.5.0"]
    iree_binary = iree.compiler.compile_str(
        computation, target_backends=[self.compiler_driver], input_type="mhlo",
        # extended_diagnostics=True,
        extra_args=extra_args,
    )
    # Load it into the runtime.
    vm_module = iree.runtime.VmModule.from_flatbuffer(
      self.iree_config.vm_instance, iree_binary)
    module_object = iree.runtime.load_vm_module(vm_module, self.iree_config)
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
