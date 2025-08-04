# Copyright 2018 The JAX Authors.
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
"""Pickling support for precompiled binaries."""

from __future__ import annotations

import pickle
import io

import jax
from jax._src.lib import xla_client as xc
from collections.abc import Sequence


def serialize(compiled: jax.stages.Compiled):
  """Serializes a compiled binary.

  Because pytrees are not serializable, they are returned so that
  the user can handle them properly.
  """
  unloaded_executable = getattr(compiled._executable,
                                '_unloaded_executable', None)
  if unloaded_executable is None:
    raise ValueError("Compilation does not support serialization")
  if getattr(unloaded_executable, 'mut', None) and unloaded_executable.mut.in_mut:
    raise ValueError("can't serialize with a closed-over mutable array ref")
  args_info_flat, in_tree = jax.tree_util.tree_flatten(compiled.args_info)
  # TODO(necula): deal with constants in serialized executables
  if compiled._params.const_args:
    raise NotImplementedError("serialize_executables with const_args")
  with io.BytesIO() as file:
    _JaxPjrtPickler(file).dump(
        (unloaded_executable, args_info_flat, compiled._no_kwargs))
    return file.getvalue(), in_tree, compiled.out_tree


def deserialize_and_load(serialized,
                         in_tree,
                         out_tree,
                         backend: str | xc.Client | None = None,
                         execution_devices: Sequence[xc.Device] | None = None):
  """Constructs a jax.stages.Compiled from a serialized executable."""

  if backend is None or isinstance(backend, str):
    backend = jax.devices(backend)[0].client

  if execution_devices is None:
    execution_devices = backend.devices()
  else:
    device_backend = execution_devices[0].client
    if device_backend != backend:
      raise ValueError(
          'Execution devices belong to a client other than `backend`. Got '
          f'backend client: {(backend.platform, backend.platform_version)} and '
          'execution devices client: '
          f'{(device_backend.platform, device_backend.platform_version)}')

  (unloaded_executable, args_info_flat,
   no_kwargs) = _JaxPjrtUnpickler(
       io.BytesIO(serialized), backend, execution_devices).load()

  args_info = in_tree.unflatten(args_info_flat)

  loaded_compiled_obj = unloaded_executable.load()
  # TODO(necula): deal with constants in serialized executables
  return jax.stages.Compiled(
      loaded_compiled_obj, [], args_info, out_tree, no_kwargs=no_kwargs)


class _JaxPjrtPickler(pickle.Pickler):
  device_types = (xc.Device,)
  client_types = (xc.Client,)

  def persistent_id(self, obj):
    if isinstance(obj, xc.LoadedExecutable):
      return ('exec', obj.client.serialize_executable(obj))
    if isinstance(obj, xc._xla.Executable):
      return ('exec', obj.serialize())
    if isinstance(obj, self.device_types):
      return ('device', obj.id)
    if isinstance(obj, self.client_types):
      return ('client',)


class _JaxPjrtUnpickler(pickle.Unpickler):

  def __init__(self, file, backend, execution_devices=None):
    super().__init__(file)
    self.backend = backend
    if execution_devices is None:
      execution_devices = backend.devices()
    else:
      device_backend = execution_devices[0].client
      if device_backend != backend:
        raise ValueError(
            'Execution devices belong to a client other than `backend`. Got '
            f'backend client: {(backend.platform, backend.platform_version)} '
            'and execution devices client: '
            f'{(device_backend.platform, device_backend.platform_version)}')
    self.devices_by_id = {d.id: d for d in execution_devices}
    self.execution_devices = xc.DeviceList(tuple(execution_devices))

  def persistent_load(self, pid):
    if pid[0] == 'exec':
      return self.backend.deserialize_executable(
          pid[1], executable_devices=self.execution_devices)
    if pid[0] == 'device':
      return self.devices_by_id[pid[1]]
    if pid[0] == 'client':
      return self.backend
    raise pickle.UnpicklingError
