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

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import functools
import math
from typing import Any, NamedTuple, cast

from jax._src import core
from jax._src import mesh as mesh_lib
from jax._src import sharding as jsharding
from jax._src import sharding_specs
from jax._src import tree_util
from jax._src import util
from jax._src import source_info_util
from jax._src import xla_bridge
from jax._src import mesh_utils
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir.dialects import sdy
from jax._src.named_sharding import (  # noqa: F401
    SdyArraySharding, SdyDimSharding, UnspecifiedValue, AUTO,
    ParsedPartitionSpec, _check_unique_resources, NamedSharding, UNSPECIFIED,
    ArrayMapping, ArrayMappingOrAutoOrUnspecified, get_array_mapping,
    array_mapping_to_axis_resources, get_single_pspec, preprocess,
    named_sharding_to_xla_hlo_sharding)
from jax._src.op_shardings import (
    are_op_shardings_equal, get_num_ways_dim_sharded, is_op_sharding_replicated)
from jax._src.partition_spec import PartitionSpec
from jax._src.util import safe_map, safe_zip, use_cpp_class, use_cpp_method
import numpy as np


Shape = tuple[int, ...]
Device = xc.Device
Index = tuple[slice, ...]
XLADeviceAssignment = tuple[Device, ...]
# TODO(yashkatariya): Remove this after 3 months of deprecation.
XLACompatibleSharding = jsharding.Sharding

@dataclasses.dataclass(frozen=True)
class TransferToMemoryKind:
  memory_kind: str


def hashed_index(x) -> int:
  # This works for both `pjit` indices and `pmap` indices (which might
  # have an integer instead of a slice).
  assert all(v.step is None for v in x if isinstance(v, slice))
  return hash(tuple((v.start, v.stop) if isinstance(v, slice) else v for v in x))


@util.cache(max_size=4096, trace_context_in_key=False)
def device_replica_id_map(sharding, global_shape: Shape) -> Mapping[Device, int]:
  try:
    device_indices_map_fn = sharding.devices_indices_map
  except AttributeError:
    raise ValueError(
        f'Cannot calculate replica ids from sharding: {sharding}. Please '
        'create a device to index mapping for your sharding from which replica '
        'ids will be calculated.') from None

  index_to_replica: dict[int, int] = collections.Counter()
  out = {}
  for device, index in device_indices_map_fn(global_shape).items():
    h_index = hashed_index(index)
    replica_id = index_to_replica[h_index]
    index_to_replica[h_index] += 1
    out[device] = replica_id
  return out


@dataclasses.dataclass
class SdyArrayShardingList:
  shardings: Sequence[SdyArraySharding]

  def build(self) -> sdy.TensorShardingPerValueAttr:
    return sdy.TensorShardingPerValueAttr.get(
        [sharding.build() for sharding in self.shardings])


# TODO(yashkatariya): Upstream this into `_to_sdy_sharding` maybe with an extra
# parameter to it `_to_sdy_sharding(self, ndim, modify_wrt_axis_types=False)`
def modify_sdy_sharding_wrt_axis_types(sdy_sharding: SdyArraySharding, mesh):
  if mesh._any_axis_auto:
    dim_shardings, used_axes = [], []  # type: ignore
    for d in sdy_sharding.dimension_shardings:
      # TODO(yashkatariya): Maybe if any mesh axis is auto, mark all axes as open?
      dim_shardings.append(SdyDimSharding(axes=[], is_closed=False)
                           if not d.axes and d.is_closed else d)
      used_axes.extend(d.axes)
    remaining_axes = set(mesh.axis_names) - set(used_axes)
    replicated_axes = tuple(r for r in remaining_axes
                            if mesh._name_to_type[r] == mesh_lib.AxisTypes.Explicit)
    return SdyArraySharding(sdy_sharding.mesh_shape, dim_shardings,
                            sdy_sharding.logical_device_ids, replicated_axes)
  return sdy_sharding


@util.cache(max_size=128, trace_context_in_key=False)
def get_replicated_hlo_sharding():
  return xc.HloSharding.replicate()


@use_cpp_class(xc.SingleDeviceSharding)
class SingleDeviceSharding(jsharding.Sharding):
  """A :class:`Sharding` that places its data on a single device.

  Args:
    device: A single :py:class:`Device`.

  Examples:

    >>> single_device_sharding = jax.sharding.SingleDeviceSharding(
    ...     jax.devices()[0])
  """

  _device: Device
  _memory_kind: str | None

  @use_cpp_method()
  def __init__(self, device: Device, *, memory_kind: str | None = None):
    self._device = device
    self._memory_kind = memory_kind

  def __reduce__(self):
    return type(self), (self._device,), {'memory_kind': self._memory_kind}

  def __repr__(self):
    mem = '' if self._memory_kind is None else f', memory_kind={self._memory_kind}'
    return f"SingleDeviceSharding(device={self._device!r}{mem})"

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self._device, self.memory_kind))
    return self._hash

  def __eq__(self, other):
    if not isinstance(other, SingleDeviceSharding):
      return False
    if self is other:
      return True
    return (self._device == other._device and
            self.memory_kind == other.memory_kind)

  @property
  def num_devices(self) -> int:
    return len(self.device_set)

  @property
  def device_set(self) -> set[Device]:
    return {self._device}

  @property
  def memory_kind(self) -> str | None:
    return self._memory_kind

  def with_memory_kind(self, kind: str) -> SingleDeviceSharding:
    return SingleDeviceSharding(self._device, memory_kind=kind)

  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:  # type: ignore
    return {self._device: (slice(None),) * len(global_shape)}

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return (self._device,)

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    return get_replicated_hlo_sharding()

  def _to_sdy_sharding(self, num_dimensions: int) -> SdyArraySharding:
    sdy_dim_sharding = [SdyDimSharding(axes=[], is_closed=True)
                        for _ in range(num_dimensions)]
    return SdyArraySharding(None, sdy_dim_sharding)

  @property
  def is_fully_replicated(self) -> bool:
    return True

  @property
  def is_fully_addressable(self) -> bool:
    return True


@util.cache(max_size=4096, trace_context_in_key=False)
def pmap_sharding_devices_indices_map(
    self, global_shape: Shape) -> Mapping[Device, Index]:
  self.shard_shape(global_shape)  # raises a good error message
  indices = sharding_specs.spec_to_indices(global_shape, self.sharding_spec)
  return dict(safe_zip(self.devices.flat, indices))  # type: ignore[arg-type]


@use_cpp_class(xc.PmapSharding)
class PmapSharding(jsharding.Sharding):
  """Describes a sharding used by :func:`jax.pmap`."""
  devices: np.ndarray
  sharding_spec: sharding_specs.ShardingSpec
  _internal_device_list: xc.DeviceList

  @use_cpp_method()
  def __init__(self, devices: Sequence[Device] | np.ndarray,
               sharding_spec: sharding_specs.ShardingSpec):
    self.devices = np.asarray(devices)
    # The sharding spec should be pmap's sharding spec.
    self.sharding_spec = sharding_spec

  def __reduce__(self):
    return (type(self), (self.devices, self.sharding_spec),
            {'memory_kind': self.memory_kind})

  def __eq__(self, other):
    if not isinstance(other, PmapSharding):
      return False
    if self is other:
      return True
    return (self.sharding_spec == other.sharding_spec and
            self.devices.shape == other.devices.shape and
            self._internal_device_list == other._internal_device_list)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self._internal_device_list, self.sharding_spec))
    return self._hash

  def __str__(self):
    device_ids = [d.id for d in self.devices.flat]
    return (f'PmapSharding(sharding_spec={self.sharding_spec}, '
            f'{device_ids=}, '
            f'device_platform={self.devices.flat[0].platform.upper()}, '
            f'device_shape={self.devices.shape})')

  def __repr__(self):
    return (f'PmapSharding(sharding_spec={self.sharding_spec}, '
            f'devices={self.devices})')

  def is_equivalent_to(self: PmapSharding, other: PmapSharding,  # type: ignore
                       ndim: int) -> bool:
    return self == other

  # TODO(yashkatariya): Expose `sharded_dim_size` in the API if required.
  @classmethod
  def default(cls, shape: Shape, sharded_dim: int | None = 0,
              devices: Sequence[xc.Device] | None = None) -> PmapSharding:
    """Creates a :class:`PmapSharding` which matches the default placement
    used by :func:`jax.pmap`.

    Args:
      shape: The shape of the input array.
      sharded_dim: Dimension the input array is sharded on. Defaults to 0.
      devices: Optional sequence of devices to use. If omitted, the implicit
        device order used by pmap is used, which is the order of
        :func:`jax.local_devices`.
    """
    if sharded_dim is None:
      if devices is None:
        raise ValueError("One of sharded_dim or devices must be set.")
      nrep = len(devices)
      return cls(np.array(devices),
          sharding_specs.pmap_sharding_spec(nrep, nrep, shape, None))

    # The dtype doesn't matter here. Its only used for creating the
    # sharding_spec.
    sharding_spec = sharding_specs.create_pmap_sharding_spec(
        tuple(shape), sharded_dim)

    num_ways_sharded = None
    for s in sharding_spec.sharding:
      if isinstance(s, sharding_specs.Unstacked):
        assert num_ways_sharded is None
        num_ways_sharded = s.size
      elif isinstance(s, sharding_specs.Chunked):
        assert num_ways_sharded is None
        if len(s.chunks) == 1:
          num_ways_sharded = s.chunks[0]
        else:
          raise NotImplementedError(
              'Multiple chunks in Chunked dimension not supported.')

    if devices is None:
      pmap_devices: np.ndarray = np.array(
          xla_bridge.local_devices()[:num_ways_sharded])
    else:
      pmap_devices = np.array(devices)
    return cls(pmap_devices, sharding_spec)

  @property
  def num_devices(self) -> int:
    return len(self.device_set)

  @functools.cached_property
  def device_set(self) -> set[Device]:
    return set(self.devices.flat)

  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:
    return pmap_sharding_devices_indices_map(self, global_shape)

  @functools.cached_property
  def _device_assignment(self) -> XLADeviceAssignment:
    return tuple(self.devices.flat)

  @property
  def memory_kind(self) -> str | None:
    try:
      return self._internal_device_list.default_memory_kind
    except:
      return None

  def with_memory_kind(self, kind: str):
    raise NotImplementedError("pmap does not support memories.")

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    raise NotImplementedError("pmap doesn't use OpSharding.")

  def _to_sdy_sharding(self, num_dimensions: int) -> SdyArraySharding:
    raise NotImplementedError("pmap doesn't use SdyArraySharding.")

  @functools.cached_property
  def is_fully_replicated(self) -> bool:
    for s in self.sharding_spec.sharding:
      if isinstance(s, (sharding_specs.Unstacked, sharding_specs.Chunked)):
        return False
    return True

  @functools.cached_property
  def is_fully_addressable(self) -> bool:
    return self._internal_device_list.is_fully_addressable

  def shard_shape(self, global_shape: Shape) -> Shape:
    sharded_dim = None
    sharded_dim_size = None
    for i, s in enumerate(self.sharding_spec.sharding):
      if isinstance(s, sharding_specs.Unstacked):
        sharded_dim = i
        sharded_dim_size = s.size
        sharded_shape = util.tuple_delete(global_shape, sharded_dim)
        break
      elif isinstance(s, sharding_specs.Chunked):
        sharded_dim = i
        assert len(s.chunks) == 1, s.chunks
        sharded_dim_size = s.chunks[0]
        sharded_shape = util.tuple_update(global_shape, sharded_dim, 1)
        break
    if sharded_dim is None:
      return global_shape
    if global_shape[sharded_dim] != sharded_dim_size:
      raise ValueError(
          f'The sharded dimension must be equal to the number of '
          f'devices passed to PmapSharding. Got sharded dimension {sharded_dim} '
          f'with value {global_shape[sharded_dim]} in shape {global_shape} and '
          f'the number of devices={len(self._device_assignment)}')
    return sharded_shape


def _op_sharding_to_pos_sharding(
    op_sharding: xc.OpSharding | xc.HloSharding,
    device_assignment: Sequence[xc.Device],
    memory_kind: str | None = None) -> PositionalSharding:
  if isinstance(op_sharding, xc.OpSharding):
    op_sharding = xc.HloSharding.from_proto(op_sharding)

  if op_sharding.is_replicated():
    return PositionalSharding(
        device_assignment, memory_kind=memory_kind).replicate()

  if len(op_sharding.subgroup_types()) > 1:
    raise NotImplementedError(
        'Unhandled HloSharding type. Please open a bug report!'
    )

  name = device_assignment[0].platform.upper()
  ids = np.array(
      [DeviceIdSet(name, i) for i in op_sharding.tile_assignment_devices()]
  )
  p = PositionalSharding._remake(tuple(device_assignment), ids,
                                 memory_kind=memory_kind)
  p = p.reshape(op_sharding.tile_assignment_dimensions())
  if op_sharding.replicate_on_last_tile_dim():
    p = p.replicate(-1, keepdims=False)
  return p


@util.cache(max_size=4096, trace_context_in_key=False)
def _positional_sharding_to_xla_hlo_sharding(
    self, num_dimensions: int) -> xc.HloSharding:
  if self.shape == (1,) * self.ndim:
    return get_replicated_hlo_sharding()

  pbuf = xc.OpSharding()
  shape = self.shape[self.ndim - num_dimensions:]  # 'rank promotion' of val
  set_size, = {len(device_set) for device_set in self._ids.flat}
  pbuf.type = xc.OpSharding.Type.OTHER
  if set_size > 1:
    pbuf.last_tile_dims = [xc.OpSharding.Type.REPLICATED]
    pbuf.tile_assignment_dimensions = (*shape, set_size)
  else:
    pbuf.tile_assignment_dimensions = shape
  pbuf.tile_assignment_devices = [i for ids in self._ids.flat for i in ids]
  product_of_dims = math.prod(pbuf.tile_assignment_dimensions)
  num_devices = len(pbuf.tile_assignment_devices)
  assert product_of_dims == num_devices, (product_of_dims, num_devices)
  return xc.HloSharding.from_proto(pbuf)


class PositionalSharding(jsharding.Sharding):
  _devices: tuple[xc.Device, ...]
  _memory_kind: str | None
  _ids: np.ndarray  # dtype DeviceIdSet

  def __init__(self, devices: Sequence[xc.Device] | np.ndarray,
               *, memory_kind: str | None = None):
    super().__init__()
    if not isinstance(devices, np.ndarray):
      devices = np.array(devices, dtype='object')
    if not devices.size:
      raise ValueError(f"{self.__class__.__name__}.__init__ requires at least "
                       f"one device, got {devices}")
    self._devices = tuple(devices.flat)
    self._memory_kind = memory_kind
    name = self._devices[0].platform.upper()
    self._ids = np.array([DeviceIdSet(name, i) for i in range(devices.size)],
                         dtype='object').reshape(devices.shape)
    self._internal_device_list = xc.DeviceList(self._devices)
    self._memory_kind = xc.check_and_canonicalize_memory_kind(
        self._memory_kind, self._internal_device_list)

  @property
  def shape(self):
    return self._ids.shape

  @property
  def ndim(self):
    return self._ids.ndim

  def __repr__(self) -> str:
    cls_name = self.__class__.__name__
    ids = self._ids.copy()
    platform_name = self._devices[0].platform.upper()
    for idx, x in np.ndenumerate(ids):
      ids[idx] = DeviceIdSet(platform_name, *(self._devices[i].id for i in x))
    body = np.array2string(ids, prefix=cls_name + '(', suffix=')',
                           max_line_width=100)
    mem = '' if self._memory_kind is None else f', memory_kind={self._memory_kind}'
    return f'{cls_name}({body}{mem}, shape={self.shape})'

  def reshape(self, *shape) -> PositionalSharding:
    return self._remake(self._devices, self._ids.reshape(*shape),
                        memory_kind=self.memory_kind)

  def transpose(self, *axes) -> PositionalSharding:
    return self._remake(self._devices, self._ids.transpose(*axes),
                        memory_kind=self.memory_kind)
  T = property(transpose)

  def replicate(self, axis=None, keepdims=True) -> PositionalSharding:
    new_ids = self._ids.sum(axis=axis, keepdims=keepdims)  # union
    return self._remake(self._devices, new_ids,
                        memory_kind=self.memory_kind)

  def check_compatible_aval(self, aval_shape: Shape) -> None:
    if len(aval_shape) != len(self.shape) and not self.is_fully_replicated:
      raise ValueError(
          f"Sharding {self} is only valid for values of rank "
          f"{len(self.shape)}, but was applied to a value of rank "
          f"{len(aval_shape)}")

  @classmethod
  def _remake(
      cls, devices: tuple[xc.Device, ...], ids: np.ndarray,
      *, memory_kind: str | None = None) -> PositionalSharding:
    sharding = cls(devices, memory_kind=memory_kind)
    sharding._ids = ids
    return sharding

  # Hashable

  def __hash__(self) -> int:
    if not hasattr(self, '_hash'):
      self._hash = hash((self._internal_device_list, self.memory_kind))
    return self._hash

  def __eq__(self, other) -> bool:
    if not isinstance(other, PositionalSharding):
      return False
    if self is other:
      return True
    all_ids_equal = np.array_equal(self._ids,other._ids)
    mem_kind_equal = self.memory_kind == other.memory_kind
    if self._devices is other._devices and mem_kind_equal and all_ids_equal:
      return True
    return (mem_kind_equal and all_ids_equal and
            self._internal_device_list == other._internal_device_list)

  # Sharding interface

  @property
  def num_devices(self) -> int:
    return len(self.device_set)

  @functools.cached_property
  def device_set(self) -> set[xc.Device]:
    return set(self._devices)

  @property
  def memory_kind(self) -> str | None:
    return self._memory_kind

  def with_memory_kind(self, kind: str) -> PositionalSharding:
    return PositionalSharding(self._devices, memory_kind=kind)

  @functools.cached_property
  def is_fully_replicated(self) -> bool:
    return self.shape == (1,) * self.ndim

  # jsharding.Sharding interface

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return self._devices

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    return _positional_sharding_to_xla_hlo_sharding(self, num_dimensions)

  def _to_sdy_sharding(self, num_dimensions: int) -> SdyArraySharding:
    raise NotImplementedError(
        "PositionalSharding can't be converted to an SdyArraySharding.")

  @functools.cached_property
  def is_fully_addressable(self) -> bool:
    return self._internal_device_list.is_fully_addressable


class DeviceIdSet:
  _name: str
  _ids: frozenset[int]
  def __init__(self, name, *ids):
    self._name = name
    self._ids = frozenset(ids)

  def __iter__(self):
    return iter(sorted(self._ids))

  def __add__(self, other) -> DeviceIdSet:
    assert isinstance(other, DeviceIdSet)
    return DeviceIdSet(self._name, *(self._ids | other._ids))

  def __len__(self) -> int:
    return len(self._ids)

  def __repr__(self) -> str:
    ids = ', '.join(safe_map(str, sorted(self._ids)))
    return f'{{{self._name} {ids}}}'

  def __hash__(self) -> int:
    return hash((self._name, self._ids))

  def __eq__(self, other) -> bool:
    return (isinstance(other, DeviceIdSet) and self._name == other._name and
            self._ids == other._ids)


@use_cpp_class(xc.GSPMDSharding)
class GSPMDSharding(jsharding.Sharding):
  _devices: tuple[Device, ...]
  _hlo_sharding: xc.HloSharding
  _memory_kind: str | None
  _device_list: xc.DeviceList | None
  _internal_device_list: xc.DeviceList

  @use_cpp_method()
  def __init__(self, devices: Sequence[Device],
               op_sharding: xc.OpSharding | xc.HloSharding,
               *, memory_kind: str | None = None,
               _device_list: xc.DeviceList | None = None):
    self._devices = tuple(devices)
    if isinstance(op_sharding, xc.OpSharding):
      self._hlo_sharding = xc.HloSharding.from_proto(op_sharding)
    else:
      self._hlo_sharding = op_sharding
    self._memory_kind = memory_kind

  def __reduce__(self):
    return (type(self), (self._devices, self._hlo_sharding.to_proto()),
            {'memory_kind': self._memory_kind})

  @functools.cached_property
  def _hlo_sharding_hash(self):
    if self.is_fully_replicated:
      return hash(get_replicated_hlo_sharding())
    return hash(self._hlo_sharding)

  def __eq__(self, other):
    if not isinstance(other, GSPMDSharding):
      return False
    if self is other:
      return True
    return (are_op_shardings_equal(self._hlo_sharding, other._hlo_sharding)
            and self.memory_kind == other.memory_kind
            and self._internal_device_list == other._internal_device_list)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self._internal_device_list, self._hlo_sharding_hash,
                        self.memory_kind))
    return self._hash

  def __repr__(self):
    mem = '' if self._memory_kind is None else f', memory_kind={self._memory_kind}'
    return f'GSPMDSharding({self._hlo_sharding!r}{mem})'

  def check_compatible_aval(self, aval_shape: Shape) -> None:
    num_ways_dim_sharded, _ = get_num_ways_dim_sharded(self._hlo_sharding)
    if len(aval_shape) < len(num_ways_dim_sharded):
      raise ValueError(
          f"Sharding {self} is only valid for values of rank at least "
          f"{len(num_ways_dim_sharded)}, but was applied to a value of rank "
          f"{len(aval_shape)}")

  @property
  def num_devices(self) -> int:
    return len(self.device_set)

  @functools.cached_property
  def device_set(self) -> set[Device]:
    return set(self._devices)

  @property
  def memory_kind(self) -> str | None:
    return self._memory_kind

  def with_memory_kind(self, kind: str) -> GSPMDSharding:
    return GSPMDSharding(self._devices, self._hlo_sharding, memory_kind=kind)

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return self._devices

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    return self._hlo_sharding

  def _to_sdy_sharding(self, num_dimensions: int) -> SdyArraySharding:
    raise NotImplementedError(
        "GSPMDSharding can't be converted to SdyArraySharding.")

  @functools.cached_property
  def is_fully_replicated(self) -> bool:
    return is_op_sharding_replicated(self._hlo_sharding)

  @functools.cached_property
  def is_fully_addressable(self) -> bool:
    return self._internal_device_list.is_fully_addressable

  @classmethod
  def get_replicated(cls, device_assignment, *, memory_kind: str | None = None):
    return cls(tuple(device_assignment), get_replicated_hlo_sharding(),
               memory_kind=memory_kind)


MeshAxisName = Any


def prepare_axis_resources(axis_resources, arg_name,
                           allow_unconstrained_dims=False):
  # PyTrees don't treat None values as leaves, so we use an is_leaf function.
  entries, treedef = tree_util.tree_flatten(
      axis_resources, is_leaf=lambda x: x is None)
  what = f"{arg_name} leaf specifications"

  new_entries = []
  for entry in entries:
    if isinstance(entry, (UnspecifiedValue, AUTO)) or entry is None:
      new_entries.append(entry)
    elif isinstance(entry, jsharding.Sharding):
      if isinstance(entry, PmapSharding):
        raise ValueError(f'One of {what} got sharding {entry} which is not '
                         'allowed.')
      new_entries.append(entry)
    else:
      parsed_pspec = ParsedPartitionSpec.from_user_input(
          entry, what, allow_unconstrained_dims=allow_unconstrained_dims)
      _check_unique_resources(parsed_pspec, arg_name)
      new_entries.append(parsed_pspec)

  return tree_util.tree_unflatten(treedef, new_entries)


# Axis environments

class AxisEnv(NamedTuple):
  """Represents a pmap mesh (only along the replica axes)."""
  nreps: int
  names: tuple[Any, ...]
  sizes: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class SPMDAxisContext:
  """A hardware axis context for parallel computations that use the GSPMD partitioner.

  This includes the mesh that will later by used to execute this computation,
  as well as a set of mesh axes that are currently lowered in the MANUAL
  sharding mode.
  """
  mesh: mesh_lib.Mesh
  manual_axes: frozenset[MeshAxisName] = frozenset()

  @property
  def axis_env(self):
    # All collectives that touch axis_env should remember to set use_global_device_ids
    # when this context is enabled!
    return self.unsafe_axis_env

  @property
  def unsafe_axis_env(self):
    return AxisEnv(
        nreps=self.mesh.size,
        names=self.mesh.axis_names,
        sizes=tuple(self.mesh.shape.values()))

  def extend_manual(self, axes: frozenset[MeshAxisName]) -> SPMDAxisContext:
    return SPMDAxisContext(self.mesh, self.manual_axes | axes)


@dataclasses.dataclass(frozen=True)
class ReplicaAxisContext:
  """A hardware axis context for parallel computations that are partitioned by JAX.

  Unlike in the SPMDAxisContext, this means that JAX might need to emit calls to
  explicit collectives.
  """
  axis_env: AxisEnv


@dataclasses.dataclass(frozen=True)
class ShardingContext:
  """A hardware axis context for parallel computations that use the sharding
  interface.

  This context also uses the GSPMD partitioner.
  """
  num_devices: int
  device_assignment: tuple[xc.Device, ...] | None = None
  abstract_mesh: mesh_lib.AbstractMesh | None = None

  def __post_init__(self):
    if self.device_assignment is not None:
      assert isinstance(self.device_assignment, tuple)
      assert self.num_devices == len(self.device_assignment)

  # Similar to SPMDContext as ShardingContext also uses the GSPMD partitioner.
  @property
  def axis_env(self):
    return AxisEnv(nreps=1, names=(), sizes=())


# -------------------- XLA OpSharding to PartitionSpec --------------------
# Note that OpSharding is more expressive than PartitionSpecs, so it's not
# always possible to convert them, but the code below should at least
# support handle all cases when this is possible.

def strides_for_sizes(sizes):
  """Returns an array of strides for major-to-minor sizes."""
  return np.cumprod(sizes[::-1])[::-1] // np.asarray(sizes)

def unflatten_array(named_sizes, assignment):
  """Recovers the ordering of axis names based on a device assignment.

  The device assignments that this function can convert into axis orders
  are of the form::

    np.arange(np.prod(named_sizes.values())).transpose(...).flatten()

  for some transposition ``...``. This is satisfied by all OpSharding assignments
  generated from partition specs.

  Arguments:
    named_sizes: A dictionary mapping axis names to their sizes.
    assignment: A permutation of integers between 0 and the product of all
      named sizes.

  Returns:
    A major-to-minor list of axis names that corresponds to the given assignment.
  """
  named_sizes = {name: size for name, size in named_sizes.items() if size != 1}
  sizes = np.fromiter(named_sizes.values(), dtype=np.int64)
  strides = strides_for_sizes(sizes)
  dims = explode_superdims(sizes, unflatten_superdims(assignment))
  dim_to_name = {(size, stride): name for size, stride, name in zip(sizes, strides, named_sizes)}
  return [dim_to_name[d] for d in dims]

def unflatten_superdims(assignment):
  """Unflatten a list of dimension sizes and their strides that generates assignment.

  If this function succeeds for a given ``assignment``, then the following property
  should be satisfied::

    dims_with_strides = unflatten_superdims(assignment)
    base_array = np.arange(map(fst, sorted(dims_with_strides, key=snd, reverse=True)))
    assignment == base_array.transpose(argsort(dims_with_strides, key=snd, reverse=True)).flatten()

  That is, the returned dimensions list all sizes of the base array (with strides
  indicating their initial order). The order of dimensions in the list corresponds
  to the permutation that applied to the base array generates the assignment.
  """
  def check(cond):
    if cond: return
    raise NotImplementedError("Failed to convert OpSharding into a ShardingSpec. "
                              "Please open a bug report!")
  flat_assignment = np.asarray(assignment, dtype=np.int64)
  check(flat_assignment[0] == 0)
  dims = []
  while flat_assignment.size > 1:
    stride = flat_assignment[1]
    for i in range(len(flat_assignment)):
      if flat_assignment[i] != i * stride: break
    else:
      # After this loop i should point to an "element after the sequence", so
      # we have to increment it if the whole array is a strided sequence.
      i += 1
    size = i
    dims.append((size, stride))
    assert size > 1  # Ensure progress
    flat_assignment = flat_assignment[::size]
  return dims

def explode_superdims(sizes, dims):
  """Explode superdims to fit a known shape.

  The unflattening process might mistakenly generate too few too large dimensions.
  For example, ``unflatten_superdims(np.arange(n))`` always returns ``[(n, 1)]``.
  This function takes a list of such contiguous super-dimensions and splits them
  into smaller dimensions such that::

    set(map(fst, explode_superdims(sizes, dims))) == set(sizes)
  """
  strides_to_sizes = {stride: size for size, stride in zip(sizes, strides_for_sizes(sizes))}
  dims = list(reversed(dims))
  final_dims = []
  for size, stride in dims:
    target_size = strides_to_sizes[stride]
    new_dims = []
    while size > target_size:
      assert target_size > 1  # Ensure progress
      assert size % target_size == 0
      new_dims.append((target_size, stride))
      size //= target_size
      stride *= target_size
      target_size = strides_to_sizes[stride]
    assert size == target_size
    new_dims.append((size, stride))
    final_dims += reversed(new_dims)
  return final_dims

def parse_flatten_op_sharding(hlo_sharding: xc.OpSharding | xc.HloSharding,
                              mesh: mesh_lib.Mesh) -> Sequence[ParsedPartitionSpec]:
  if isinstance(hlo_sharding, xc.OpSharding):
    hlo_sharding = xc.HloSharding.from_proto(hlo_sharding)
  if hlo_sharding.tuple_elements():
    out: list[ParsedPartitionSpec] = []
    for s in hlo_sharding.tuple_elements():
      out.extend(parse_flatten_op_sharding(s, mesh))
    return out
  elif hlo_sharding.is_replicated():
    return [ParsedPartitionSpec(PartitionSpec(), ())]
  elif hlo_sharding.is_tiled():
    mesh_shape = mesh.shape
    mesh_axis_order = unflatten_array(
        mesh.shape, hlo_sharding.tile_assignment_devices()
    )
    mesh_axis = iter(mesh_axis_order)
    shape = hlo_sharding.tile_assignment_dimensions()
    partitions = []
    for dim_size in shape:
      dim_partitions = []
      while dim_size > 1:
        axis = next(mesh_axis)
        axis_size = mesh_shape[axis]
        assert dim_size % axis_size == 0
        dim_size //= axis_size
        dim_partitions.append(axis)
      partitions.append(tuple(dim_partitions))
    if len(hlo_sharding.subgroup_types()) > 1:
      raise NotImplementedError(
          'Unhandled HloSharding type. Please open a bug report!'
      )
    if hlo_sharding.replicate_on_last_tile_dim():
      partitions = partitions[:-1]
    while partitions and partitions[-1] == ():
      partitions.pop()
    return [ParsedPartitionSpec(None, partitions)]
  else:
    raise AssertionError("Unhandled OpSharding type. Please open a bug report!")


def _slice_as_tuple(s: slice):
  assert s.step is None
  return (s.start, s.stop)


class NonUniformShardingError(ValueError):
  """Raised when sharding is not uniform across processes."""


def get_process_index_and_count(
    tensor_sharding: jsharding.Sharding, dim: int, ndims: int) -> tuple[int, int]:
  """Get current process index and number of unique processes for given dimension.

  This function facilitates mapping of process-level data to individual
  devices. Each process can use its index to obtain the data corresponding
  to that index. If process level data is sharded on multiple dimensions
  this function can be used to build the cross product of indices in
  each sharded axis. Processes that need to load the same data will have
  the same index. For shardings whose per-process data is not distributed
  on a grid, the number of distinct shards will be such that it is possible to
  build the target shape while maintaining a "cube" shape of local-process data.

  For example, in case of 4 hosts with sharding distributed like so:

  1234
  2143

  For dim 0 (rows): all processes need to access all rows, so we return (0, 1)
  For dim 1 (cols):
     process 1 and 2 returns index 0 out of 2 (need cols 0 and 1),
     process 3 and 4 returns index 1 out of 2 (need cols 2 and 3).

  On the other hand, for a sharding like:

  1212
  3434

  Dim 0 (rows): process 1 and 2 returns (0, 2), process 3 and 4 returns (1, 2)
  Dim 1 (cols): process 1 and 3 returns (0, 2), process 2 and 4 returns (1, 2)

  Note: This function requires sharding to be process uniform in dimension
  `dim`:
   each process has the same number of addressable indices in that
  dimension and all index sets across processes are either disjoint or the same.

  For sharding to be process uniform the addressable shards doesn't need to
  form contiguous subtensor, or even a sparse grid  and  in case of
  interleaved high-dimensional tensor it is possible for sharding to be
  process uniform only in some dimensions but not others.

  For example:
    1111 and 12 and 1212 and 1212
    2222     21     2121     1212

  are all sharding uniform, in both dimensions. However

    1122
    2121
    1121
    1222

  is uniform in dimension 0 (both hosts access all rows), but
  is not uniform in dimension 1 (host 1 accesses columns: 0, 1, and 3),
  while host 2 accesses (0, 1, 2, 3).

  Returns:
    A tuple of (index, num_distinct_shards) for the given dimension.
    It is guaranteed that `index` will cover 0 to `num_distinct_shards - 1`,
    across all processes.

  Raises:
    NonUniformShardingError: if the sharding is not process uniform in dimension
    `dim`.
  """
  # TODO(sandler, yashkatariya): Consider making this function public.

  if (tensor_sharding.is_fully_addressable or
      tensor_sharding.is_fully_replicated):
    return (0, 1)
  # Get device to indices map, we don't care about the concrete
  # global shape here, only to get the distribution of shards across the tensor
  # using (num_devices, num_devices, ...)  This is a universal shape that is
  # compatible with any mesh with num_devices.
  device_map = tensor_sharding.devices_indices_map(
      (tensor_sharding.num_devices,) * ndims)

  # Get the slices for 'dim' for all devices.
  global_slice = {k: v[dim] for k, v in device_map.items()}

  # Contains mapping from process_index to a set of slices for that process.
  process_to_slice = collections.defaultdict(set)
  # Contains global set of slices across all processes.
  all_slices = set()

  # Compute the set of slices for each process and the global set of slices.
  for d, v in global_slice.items():
    key = (v.start, v.stop)
    process_to_slice[d.process_index].add(key)
    all_slices.add(key)

  # Get the set of slices for the current process which we will use to compute
  # the index of the current process.
  current_pid = next(iter(tensor_sharding.addressable_devices)).process_index
  addressable_slices = frozenset(process_to_slice[current_pid])

  # Verify that all processes have the same number of slices.
  slices_per_process = len(addressable_slices)
  if any(len(x) != slices_per_process for x in process_to_slice.values()):
    raise NonUniformShardingError(
        f'{tensor_sharding=} is non-uniform on {dim=} as some processes have '
        'different number of slices.'
    )
  unique_processes = list({frozenset(x) for x in process_to_slice.values()})

  # After removing duplicate processes all unique slices should
  # cover the dimension exactly once. If they don' it means that
  # the sharding is not uniform.
  if sum(len(h) for h in unique_processes) != len(all_slices):
    raise NonUniformShardingError(
        f'{tensor_sharding=} is non-uniform on {dim=}'
    )
  return (unique_processes.index(addressable_slices), len(unique_processes))


def local_to_global_shape(
    sharding: jsharding.Sharding, local_shape: Shape) -> tuple[int | None, ...]:
  """Computes the global shape given the per process if possible.

  The returned shape will have the size of the global tensor in that dimension
  or None, if it is not computable. The latter can happen when sharding
  is not uniform along that dimension, e.g. different hosts require
  different shapes, or if different processes have partial data overlap.

  If at most one dimension is sharded the shape is always computable.
  Generally, global shape is computable for most practical meshes (including
  topology aware such as meshes returned by mesh_utils.create_device_mesh)

  Some examples: Suppose mesh is {'a': 2, 'b': 2, 'c': 2} with 2 devices
  per host, 4 hosts total. For different specs we get:
  - P():
      global_shape = local_shape

  - P(('a', 'b', 'c'), None):
      global_shape =  (4 * local_shape[0], local_shape[1])
      Note: per device shape is (local_shape[0] / 2, local_shape[1])

  - P(('a', 'b'), None)
      global_shape =  (4 * local_shape[0], local_shape[1])
      # NB: the same global shape as above, since sharding along 'c' dimension
      # happens to be within process, and thus doesn't affect the global shape.
      # The underlying difference will be in the per *device* shape, which
      # would be  (local_shape[0], local_shape[1]) in this case.

  - P(None, ('a', 'c'))
      global_shape = (local_shape[0], 2 * local_shape[1])
      # Per device shape is (local_shape[0], local_shape[1] / 2)
  - P(('a', 'c'), 'b'):
      global_shape = (2 * local_shape[0], 2 * local_shape[1])
      # Per device shape is (local_shape[0] / 2, local_shape[1])
  - If devices in the Mesh are randomly permuted: For any partition spec
  which shards more than 1 axis:  e.g. P('a', ('b', 'c')):
      global_shape = (None, None)

  Args:
    local_shape: global shape of the tensor.

  Returns:
    global_shape with Nones in non-uniform dimensions.
  """

  global_shape : list[int | None] = [None] * len(local_shape)
  for i, local_dim in enumerate(local_shape):
    try:
      _, shard_count = get_process_index_and_count(
          sharding, i, ndims=len(local_shape))
      global_shape[i] = local_dim * shard_count
    except NonUniformShardingError:
      global_shape[i] = None
      continue

  return tuple(global_shape)


def num_addressable_indices(
    tensor_sharding: jsharding.Sharding, dim: int, global_shape: Shape) -> int:
  """Returns the number of indices for given dimension this host has access to.

  Each host can have multiple number of devices that are spanning
  possibly discontiguous slices of data. This function computes the
  total number of unique indices for dimension `dim` that any of its
  addressable devices hold.

  In most cases the addressable indices form a sparse grid (and in some
  cases a subcube), and thus each host will hold the same of number of
  indices for each dimension.  However, it is possible to design a mesh that
  addressable shards form a complicated pattern. In that case, the returned
  value is the number of indices that are addressable by at least one device.

  For example, suppose the sharding looks like this: (number indicates
  the host index)

    1221
    1221
    0000

  Then on host 1 and 2, both dim 0 (rows), and  dim=1 (cols) will have size 2,
  while on host 0, dim 0  will have size 1, and dim 1 will have size 4.

  Args:
    tensor_sharding: Sharding of the tensor.
    dim: dimension along which to compute the number of addressable indices.
    global_shape: global shape of the tensor.

  Returns:
    The number of indices for dimension  `dim` that this host holds.
  """
  # TODO(sandler, yashkatariya): Consider making this function public.
  addressables = tensor_sharding.addressable_devices_indices_map(global_shape)
  addressables = cast(Mapping[jsharding.Device, Index], addressables)
  num_unique_slices = len({
      _slice_as_tuple(addressable[dim]) for addressable in addressables.values()
  })
  shard_size = tensor_sharding.shard_shape(global_shape)[dim]
  return shard_size * num_unique_slices


def physical_hlo_sharding(aval, hlo_sharding: xc.HloSharding) -> xc.HloSharding:
  elt_aval = core.physical_element_aval(aval.dtype)
  new_op_sharding = hlo_sharding.to_proto().clone()
  partitions, num_replicas = get_num_ways_dim_sharded(hlo_sharding)
  suffix = [] if num_replicas == 1 else [num_replicas]
  tad = partitions + [1] * elt_aval.ndim + suffix
  new_op_sharding.tile_assignment_dimensions = tad
  return xc.HloSharding.from_proto(new_op_sharding)

def is_single_device_sharding(sharding: jsharding.Sharding) -> bool:
  # Special case PmapSharding here because PmapSharding maps away an axis
  # and needs to be handled separately.test_pjit_single_device_sharding_add
  return sharding.num_devices == 1 and not isinstance(sharding, PmapSharding)

def make_key_array_phys_sharding(aval, sharding):
  if is_single_device_sharding(sharding):
    return sharding
  elif isinstance(sharding, PmapSharding):
    elt_aval = core.physical_element_aval(aval.dtype)
    trailing_sharding = [sharding_specs.NoSharding()] * elt_aval.ndim
    phys_sharding_spec = sharding_specs.ShardingSpec(
        sharding=(*sharding.sharding_spec.sharding, *trailing_sharding),
        mesh_mapping=sharding.sharding_spec.mesh_mapping)
    return PmapSharding(devices=sharding.devices,
                        sharding_spec=phys_sharding_spec)
  elif isinstance(sharding, NamedSharding):
    elt_aval = core.physical_element_aval(aval.dtype)
    trailing_spec = [None] * elt_aval.ndim
    return NamedSharding(
        sharding.mesh,
        PartitionSpec(*sharding.spec, *trailing_spec))
  else:
    hlos = sharding._to_xla_hlo_sharding(aval.ndim)
    return GSPMDSharding(
        sharding._device_assignment, physical_hlo_sharding(aval, hlos))


def physical_sharding(
    aval, sharding: jsharding.Sharding) -> jsharding.Sharding:
  return make_key_array_phys_sharding(aval, sharding)


def get_logical_gspmd_sharding(aval, phys_sharding):
  elt_aval = core.physical_element_aval(aval.dtype)
  phys_hlo_sharding = phys_sharding._to_xla_hlo_sharding(
      aval.ndim + elt_aval.ndim)
  partitions, num_replicas = get_num_ways_dim_sharded(phys_hlo_sharding)
  suffix = [] if num_replicas == 1 else [num_replicas]
  # Create logical sharding by cutting off the replicated trailing dims.
  logical_op_sharding = phys_hlo_sharding.to_proto().clone()
  tad = partitions[:-elt_aval.ndim] + suffix
  logical_op_sharding.tile_assignment_dimensions = tad
  return GSPMDSharding(phys_sharding._device_assignment,
                       xc.HloSharding.from_proto(logical_op_sharding))

def check_replicated_trailing_dims(sharding: jsharding.Sharding, aval):
  if isinstance(sharding, PmapSharding):
    return
  phys_aval = core.physical_aval(aval)
  hlo_s = sharding._to_xla_hlo_sharding(phys_aval.ndim)
  partitions, _ = get_num_ways_dim_sharded(hlo_s)
  num_trailing_dims = phys_aval.ndim - aval.ndim
  if not all(i == 1 for i in partitions[-num_trailing_dims:]):
    raise AssertionError(
        "The trailing dims of extended dtypes should be replicated. Got"
        f" sharding: {sharding}, partitions: {partitions}, "
        f"num_trailing_dims: {num_trailing_dims}")

def logical_sharding(aval, phys_sharding) -> jsharding.Sharding:
  # The trailing dims should always be replicated.
  check_replicated_trailing_dims(phys_sharding, aval)

  if is_single_device_sharding(phys_sharding):
    return phys_sharding
  elif isinstance(phys_sharding, PmapSharding):
    elt_aval = core.physical_element_aval(aval.dtype)
    logical_sharding_spec = sharding_specs.ShardingSpec(
        sharding=phys_sharding.sharding_spec.sharding[:-elt_aval.ndim],
        mesh_mapping=phys_sharding.sharding_spec.mesh_mapping)
    return PmapSharding(devices=phys_sharding.devices,
                        sharding_spec=logical_sharding_spec)
  elif isinstance(phys_sharding, NamedSharding):
    logical_gs = get_logical_gspmd_sharding(aval, phys_sharding)
    assert isinstance(phys_sharding.mesh, mesh_lib.Mesh)
    return _gspmd_to_named_sharding_via_mesh(
        logical_gs, phys_sharding.mesh)
  else:
    return get_logical_gspmd_sharding(aval, phys_sharding)


@util.cache()
def create_mesh_pspec_sharding(
    mesh: mesh_lib.Mesh, pspec: PartitionSpec | None, parsed_pspec=None,
    memory_kind: str | None = None) -> NamedSharding:
  if pspec is None:
    pspec, parsed_pspec = PartitionSpec(), None
  return NamedSharding(mesh, pspec, _parsed_pspec=parsed_pspec,
                       memory_kind=memory_kind)


def _gspmd_to_named_sharding_via_mesh(
    out_s: GSPMDSharding, mesh: mesh_lib.Mesh) -> NamedSharding:
  parsed_pspec = parse_flatten_op_sharding(
      out_s._hlo_sharding, mesh)[0]
  return create_mesh_pspec_sharding(
      mesh, parsed_pspec.get_partition_spec(), parsed_pspec,
      out_s.memory_kind)

def flatten_spec(spec):
  out = []
  for s in spec:
    if s is None:
      continue
    if isinstance(s, tuple):
      out.extend(s)
    else:
      out.append(s)
  return out

def canonicalize_sharding(sharding: NamedSharding | PartitionSpec | None,
                          api_name: str, check_mesh_consistency: bool = True
                          ) -> NamedSharding | None:
  if sharding is None:
    return sharding
  if isinstance(sharding, NamedSharding) and sharding.mesh.empty:
    return None

  cur_mesh = mesh_lib.get_abstract_mesh()
  if isinstance(sharding, PartitionSpec):
    if cur_mesh.empty:
      raise ValueError(
          'Using PartitionSpec when you are not under a mesh context is not'
          ' allowed. Please pass a NamedSharding instance or enter into a mesh'
          f' context via `jax.sharding.use_mesh`. Got {sharding}')
    sharding = NamedSharding(cur_mesh, sharding)
  else:
    # There are cases when you have multiple meshes set. Allow that for full
    # auto mode because of existing use cases.
    # TODO(yashkatariya): Remove this once we disallow different meshes and
    # fix the existing use cases.
    if (sharding.mesh.abstract_mesh._are_all_axes_auto and
        cur_mesh._are_all_axes_auto):
      check_mesh_consistency = False
    if (check_mesh_consistency and not cur_mesh.empty and
        sharding.mesh.abstract_mesh != cur_mesh):
      raise ValueError(
          f'Context mesh {cur_mesh} should match the mesh of sharding'
          f' {sharding.mesh.abstract_mesh} passed to {api_name}.'
          ' This error occurs at source: '
          f' {source_info_util.summarize(source_info_util.current())}')
    if isinstance(sharding.mesh, mesh_lib.Mesh):
      sharding = NamedSharding(sharding.mesh.abstract_mesh, sharding.spec)

  for s in flatten_spec(sharding.spec):
    if sharding.mesh._name_to_type[s] in {
        mesh_lib.AxisTypes.Auto, mesh_lib.AxisTypes.Manual}:
      raise ValueError(
          f'PartitionSpec passed to {api_name} cannot contain axis'
          ' names that are of type Auto or Manual. Got PartitionSpec:'
          f' {sharding.spec} with axis name: {s} of type:'
          f' {sharding.mesh._name_to_type[s]}, This error occurs at source: '
          f' {source_info_util.summarize(source_info_util.current())}')
  return sharding

TypeOfAxis = str | tuple[str, ...] | None

def _normalize(axes: TypeOfAxis = None) -> tuple[str, ...]:
  if axes is None:
    return ()
  return (axes,) if isinstance(axes, str) else axes

def _get_axis_types(
    auto_axes: TypeOfAxis = None, explicit_axes: TypeOfAxis = None,
    manual_axes: TypeOfAxis = None):
  if auto_axes is None and explicit_axes is None and manual_axes is None:
    return None

  auto_axes = _normalize(auto_axes)
  explicit_axes = _normalize(explicit_axes)
  manual_axes = _normalize(manual_axes)

  aua, ea, ma = set(auto_axes), set(explicit_axes), set(manual_axes)
  disjoint = aua.isdisjoint(ea) and aua.isdisjoint(ma) and ea.isdisjoint(ma)
  if not disjoint:
    raise ValueError(
        f'{auto_axes=}, {explicit_axes=} and {manual_axes=} should be'
        ' non-overlapping.')

  out = {}
  if auto_axes:
    out.update({mesh_lib.AxisTypes.Auto: auto_axes})
  if explicit_axes:
    out.update({mesh_lib.AxisTypes.Explicit: explicit_axes})
  if manual_axes:
    out.update({mesh_lib.AxisTypes.Manual: manual_axes})
  return out


def make_mesh(axis_shapes: Sequence[int], axis_names: Sequence[str],
              *, devices: Sequence[xc.Device] | None = None,
              auto_axes: TypeOfAxis = None, explicit_axes: TypeOfAxis = None,
              manual_axes: TypeOfAxis = None) -> mesh_lib.Mesh:
  """Creates an efficient mesh with the shape and axis names specified.

  This function attempts to automatically compute a good mapping from a set of
  logical axes to a physical mesh. For example, on a TPU v3 with 8 devices:

  >>> mesh = jax.make_mesh((8,), ('x'))  # doctest: +SKIP
  >>> [d.id for d in mesh.devices.flat]  # doctest: +SKIP
  [0, 1, 2, 3, 6, 7, 4, 5]

  The above ordering takes into account the physical topology of TPU v3.
  It orders the devices into a ring, which yields efficient all-reduces on a
  TPU v3.

  Now, let's see another example with 16 devices of TPU v3:

  >>> mesh = jax.make_mesh((2, 8), ('x', 'y'))  # doctest: +SKIP
  >>> [d.id for d in mesh.devices.flat]  # doctest: +SKIP
  [0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11, 14, 15, 12, 13]
  >>> mesh = jax.make_mesh((4, 4), ('x', 'y'))  # doctest: +SKIP
  >>> [d.id for d in mesh.devices.flat]  # doctest: +SKIP
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

  As you can see, logical axes (`axis_shapes`) affect the ordering of the
  devices.

  You can use `jax.experimental.mesh_utils.create_device_mesh` if you want to
  use the extra arguments it provides like `contiguous_submeshes` and
  `allow_split_physical_axes`.

  Args:
    axis_shapes: Shape of the mesh. For example, axis_shape=(4, 2)
    axis_names: Names of the mesh axes. For example, axis_names=('x', 'y')
    devices: Optional keyword only argument, that allows you to specify the
      devices you want to create a mesh with.

  Returns:
    A `jax.sharding.Mesh` object.
  """
  if devices is None:
    devices = xla_bridge.devices()
  new_axis_shapes = mesh_utils._canonicalize_axis_sizes(axis_shapes)
  if new_axis_shapes is None:
    raise ValueError(
        '`axis_shapes` passed to `make_mesh` should be a sequence of ints.'
        f' Got {axis_shapes}')
  del axis_shapes

  axis_size = math.prod(new_axis_shapes)
  if axis_size > len(devices):
    raise ValueError(
        f'Number of devices {len(devices)} must be >= the product '
        f'of mesh_shape {new_axis_shapes}')
  elif axis_size < len(devices):
    devices = devices[:axis_size]
  if devices[0].device_kind in (mesh_utils._TPU_V5_LITE, mesh_utils._TPU_V5E):
    allow_split_physical_axes = True
  else:
    allow_split_physical_axes = False
  mesh_devices = mesh_utils.create_device_mesh(
      new_axis_shapes, devices,
      allow_split_physical_axes=allow_split_physical_axes)
  axis_types = _get_axis_types(auto_axes, explicit_axes, manual_axes)
  return mesh_lib.Mesh(mesh_devices, axis_names, axis_types=axis_types)
