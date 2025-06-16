# Copyright 2024 The JAX Authors.
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
"""Colocated Python top-level API."""

from __future__ import annotations

import collections
from typing import Any, Callable, Sequence, Type, overload

import jax
from jax._src import api_util
from jax._src import util
from jax.experimental.colocated_python.func import make_callable
from jax.experimental.colocated_python.obj import wrap_class
import numpy as np


@overload
def colocated_cpu_devices(
    devices_or_mesh: Sequence[jax.Device],
) -> Sequence[jax.Device]:
  ...


@overload
def colocated_cpu_devices(
    devices_or_mesh: jax.sharding.Mesh,
) -> jax.sharding.Mesh:
  ...


def colocated_cpu_devices(devices_or_mesh):
  """Finds devices or a mesh that has CPU devices colocated with the given devices or mesh."""
  if isinstance(devices_or_mesh, jax.sharding.Mesh):
    return _colocated_cpu_mesh_cached(devices_or_mesh)

  if not isinstance(devices_or_mesh, tuple):
    devices_or_mesh = tuple(devices_or_mesh)
  try:
    return _colocated_cpu_devices_cached(devices_or_mesh)
  except (ValueError, AttributeError):
    return _colocated_cpu_devices_cached_fallback_to_cpu_backend(
        devices_or_mesh
    )


@util.cache(max_size=1024, trace_context_in_key=False)
def _colocated_cpu_devices_cached(
    devices: tuple[jax.Device, ...],
) -> Sequence[jax.Device]:
  cpu_devices_by_colocation_id = collections.defaultdict(list)
  for device in devices[0].client._get_all_devices():  # pylint: disable=protected-access
    if device.device_kind == "cpu":
      cpu_devices_by_colocation_id[device.colocation_id].append(device)
  if not cpu_devices_by_colocation_id:
    raise ValueError("No CPU devices found")

  colocated_cpu_devices = []
  for device in devices:
    matches = cpu_devices_by_colocation_id[device.colocation_id]
    if not matches:
      raise ValueError(f"Device {device} has no colocated devices")
    elif len(matches) > 1:
      raise ValueError(
          f"Ambiguous colocated devices; device {device} has"
          f" {len(matches)} colocated devices: f{matches}"
      )
    colocated_cpu_devices.append(matches[0])
  return colocated_cpu_devices


@util.cache(max_size=1024, trace_context_in_key=False)
def _colocated_cpu_devices_cached_fallback_to_cpu_backend(
    devices: tuple[jax.Device, ...],
) -> Sequence[jax.Device]:
  # PjRt-IFRT currently defines CPU devices by using a CPU backend.
  # TODO(hyeontaek): Remove this fallback path once a PjRt-IFRT backend defines
  # CPU devices by its own instead of using a separate CPU backend.
  cpu_backend_devices = jax.local_devices(backend="cpu")
  device_index_map = {device.id: i for i, device in enumerate(jax.devices())}

  available_devices = devices[: min(len(cpu_backend_devices), len(devices))]
  return [
      cpu_backend_devices[device_index_map[d.id]] for d in available_devices
  ]


@util.cache(max_size=1024, trace_context_in_key=False)
def _colocated_cpu_mesh_cached(mesh: jax.sharding.Mesh) -> jax.sharding.Mesh:
  """Returns a CPU mesh that is similar to the given mesh but has colocated CPU devices."""
  # Finding colocated CPU devices reuses the cache of `colocated_cpu_devices`
  # called with devices. `_colocated_cpu_mesh` itself is also cached to avoid
  # creating a new `Mesh` object repeatedly.
  flat_cpu_devices = colocated_cpu_devices(tuple(mesh.devices.flat))
  return jax.sharding.Mesh(
      np.array(flat_cpu_devices).reshape(mesh.axis_sizes),
      mesh.axis_names,
      axis_types=mesh.axis_types,
  )


def colocated_python(fun: Callable[..., Any]):
  """Executes the given Python function on the same devices as the arguments."""
  return make_callable(
      fun, api_util.fun_sourceinfo(fun), api_util.fun_signature(fun)
  )


def colocated_python_class(cls: Type[object]) -> Type[object]:
  """Executes the given Python class methods on the same devices as the arguments."""
  return wrap_class(cls, api_util.fun_sourceinfo(cls))
