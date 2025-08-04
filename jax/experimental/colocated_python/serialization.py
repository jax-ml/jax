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
"""Colocated Python serialization utilities."""

from __future__ import annotations

import base64
import collections
from collections.abc import Callable, Sequence
import functools
import io
from typing import Any

try:
  import cloudpickle  # type: ignore[import-not-found]
except ImportError:
  cloudpickle = None

import jax
from jax._src import api
from jax._src import tree_util
from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc
import numpy as np

DeviceList = xc.DeviceList


@jax._src.util.cache(max_size=None)
def _get_cpu_device_map() -> dict[int, jax.Device]:
  """Returns a map from a device id to a matching device."""
  cpu_device_map: dict[int, jax.Device] = {}
  # TODO(hyeontaek): We should look up CPU devices for a specific CPU backend.
  # When deserializing a device on the controller, the backend should be the one
  # associated with colocated_python. When deserializing on the colocated_python
  # executor, it should be the CPU backend visible to the user function running
  # under colocated_python.

  # Look for CPU devices in the default backend.
  for d in xb.local_devices()[0].client._get_all_devices():  # pylint: disable=protected-access
    if d.device_kind == "cpu":
      if d.id in cpu_device_map:
        raise ValueError(
            f"Multiple CPU devices with id {d.id} found:"
            f" {cpu_device_map[d.id]} and {d}"
        )
      cpu_device_map[d.id] = d
  if cpu_device_map:
    return cpu_device_map

  # Fall back to searching CPU devices in all backends.
  for backend in xb.backends().values():
    for d in backend._get_all_devices():  # pylint: disable=protected-access
      if d.device_kind == "cpu":
        if d.id in cpu_device_map:
          raise ValueError(
              f"Multiple CPU devices with id {d.id} found:"
              f" {cpu_device_map[d.id]} and {d}"
          )
        cpu_device_map[d.id] = d
  return cpu_device_map


def _lookup_cpu_device(
    cpu_device_map: dict[int, jax.Device], device_id: int
) -> jax.Device:
  """Returns a CPU device with the given device ID."""
  d = cpu_device_map.get(device_id)
  if d is None:
    raise ValueError(
        f"Invalid device ID {device_id}. Device list must contain only CPU"
        " devices."
    )
  return d


def _reduce_mesh(
    mesh: jax.sharding.Mesh,
) -> tuple[Callable[..., jax.sharding.Mesh], Any]:
  def make_mesh(
      mesh_device_ids: np.ndarray, axis_names: Any
  ) -> jax.sharding.Mesh:
    cpu_device_map = _get_cpu_device_map()
    mesh_devices = np.vectorize(
        functools.partial(_lookup_cpu_device, cpu_device_map)
    )(mesh_device_ids)
    return jax.sharding.Mesh(mesh_devices, axis_names)

  mesh_device_ids = np.vectorize(lambda d: d.id, otypes=[int])(mesh.devices)
  return make_mesh, (mesh_device_ids, mesh.axis_names)


def _reduce_named_sharding(
    sharding: jax.sharding.NamedSharding,
) -> tuple[Callable[..., jax.sharding.NamedSharding], Any]:

  def make_named_sharding(
      mesh: jax.sharding.Mesh,
      spec: jax.sharding.PartitionSpec,
      memory_kind: str | None,
  ) -> jax.sharding.NamedSharding:
    if jax._src.lib.ifrt_version < 19:
      memory_kind = None
    return jax.sharding.NamedSharding(mesh, spec, memory_kind=memory_kind)

  return make_named_sharding, (
      sharding.mesh,
      sharding.spec,
      sharding.memory_kind,
  )


def _reduce_device_list(
    device_list: DeviceList,
) -> tuple[Callable[..., DeviceList], Any]:
  def make_device_list(device_ids: Sequence[int]) -> DeviceList:
    cpu_device_map = _get_cpu_device_map()
    devices = np.vectorize(
        functools.partial(_lookup_cpu_device, cpu_device_map)
    )(device_ids)
    return DeviceList(tuple(devices))

  device_ids = [d.id for d in device_list]
  return make_device_list, (device_ids,)


def _reduce_single_device_sharding(
    sharding: jax.sharding.SingleDeviceSharding,
) -> tuple[Callable[..., jax.sharding.SingleDeviceSharding], Any]:

  def make_single_device_sharding(
      device_id: int, memory_kind: str | None
  ) -> jax.sharding.SingleDeviceSharding:
    if jax._src.lib.ifrt_version < 19:
      memory_kind = None
    cpu_device_map = _get_cpu_device_map()
    device = _lookup_cpu_device(cpu_device_map, device_id)
    return jax.sharding.SingleDeviceSharding(device, memory_kind=memory_kind)

  return make_single_device_sharding, (
      sharding.device_set.pop().id,
      sharding.memory_kind,
  )


def _serialize(obj: Any) -> bytes:
  """Serializes callables and input/output spec objects.

  DO NOT USE THIS FUNCTION EXCEPT FOR THE INTERNAL IMPLEMENTATION OF
  colocated_python.

  This module contains utility functions used internally for implementiong
  `colocated_python` when it ships callables and input/output specs through
  IFRT. The pickled data is produced and consumed in an ephermeral fashion
  without any persistence, and it does not expect any version compatibility
  (which cloudpickle does not guarantee). Furthermore, serialization and
  deserialization is expected to be done on machine(s) that are controlled by a
  single tenant, which allows unpickling done during deserialization to be
  trusted.

  Raises:
    ModuleNotFoundError: If cloudpickle is not available.
  """
  if cloudpickle is None:
    raise ModuleNotFoundError('No module named "cloudpickle"')

  class _CustomPickler(cloudpickle.Pickler):
    dispatch_table = collections.ChainMap(
        {jax.sharding.Mesh: _reduce_mesh},
        {jax.sharding.NamedSharding: _reduce_named_sharding},
        {DeviceList: _reduce_device_list},
        {jax.sharding.SingleDeviceSharding: _reduce_single_device_sharding},
        cloudpickle.CloudPickler.dispatch_table,  # pylint: disable=attribute-error
    )
    dispatch = dispatch_table

  with io.BytesIO() as file:
    _CustomPickler(file).dump(obj)
    return file.getvalue()


def _deserialize(serialized: bytes) -> Any:
  """Deserializes callables and input/output spec objects.

  DO NOT USE THIS FUNCTION EXCEPT FOR THE INTERNAL IMPLEMENTATION OF
  colocated_python. See serialize() for details.

  Raises:
    ModuleNotFoundError: If cloudpickle is not available.
  """
  if cloudpickle is None:
    raise ModuleNotFoundError('No module named "cloudpickle"')

  return cloudpickle.loads(serialized)


def _make_specs_for_serialized_specs(
    devices: DeviceList,
) -> api.ShapeDtypeStruct:
  """Makes output specs for serialized specs."""
  mesh = jax.sharding.Mesh(tuple(devices), ("x",))
  replicated_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec()
  )
  return api.ShapeDtypeStruct(
      shape=(), dtype=np.dtypes.StringDType(), sharding=replicated_sharding  # type: ignore
  )


def _serialize_specs(
    specs_treedef: tree_util.PyTreeDef,
    specs_leaves: tuple[api.ShapeDtypeStruct, ...],
    devices: DeviceList,
) -> jax.Array:
  """Serializes the output specs into a jax.Array of string type.

  DO NOT USE THIS FUNCTION EXCEPT FOR THE INTERNAL IMPLEMENTATION OF
  colocated_python. See serialize() for details.
  """
  if not hasattr(np.dtypes, "StringDType"):
    raise TypeError(
        "Serializing Colocated Python requires StringDType. Please use"
        " numpy to 2.0.0 or later, or explicitly provide an output spec"
        " function."
    )

  s_bytes = _serialize((specs_treedef, specs_leaves))
  s_str = base64.b64encode(s_bytes).decode("ascii")
  s_np_array = np.array(s_str, dtype=np.dtypes.StringDType())  # type: ignore

  # TODO(jmudigonda): Revisit this when JAX supports HLO sharding for making
  # jax.Array via make_array_from_single_device_arrays. We should then use a
  # sharding that spans all the execution devices - not just the addressable
  # ones.
  addressable_devices = devices.addressable_device_list
  mesh = jax.sharding.Mesh(tuple(addressable_devices), ("x",))
  replicated_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec()
  )

  out_arrays = [
      jax.device_put(s_np_array, device) for device in addressable_devices
  ]
  return jax.make_array_from_single_device_arrays(
      arrays=out_arrays,
      sharding=replicated_sharding,
      shape=(),
  )


def _deserialize_specs(
    serialized_specs: jax.Array,
) -> tuple[tree_util.PyTreeDef, tuple[api.ShapeDtypeStruct, ...]]:
  """Deserializes the specs from the serialized specs.

  DO NOT USE THIS FUNCTION EXCEPT FOR THE INTERNAL IMPLEMENTATION OF
  colocated_python. See serialize() for details.
  """
  data_array = serialized_specs.addressable_shards[0].data
  data = base64.b64decode(data_array.item().encode("ascii"))
  return _deserialize(data)
