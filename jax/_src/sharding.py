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

import abc
import functools
from collections import Counter
import operator as op
from typing import (Sequence, List, Tuple, Optional, Mapping, Dict, Set,
                    FrozenSet, Union, cast)

import jax
from jax._src import core
from jax._src.util import safe_map, safe_zip, use_cpp_class, use_cpp_method
from jax._src.lib import xla_client as xc
from jax._src.lib import xla_extension_version
from jax.interpreters import mlir
from jax._src.interpreters import pxla

import numpy as np

Shape = Tuple[int, ...]
Device = xc.Device
Index = Tuple[slice, ...]
XLADeviceAssignment = Sequence[Device]

@use_cpp_class(xc.Sharding)
class Sharding(metaclass=abc.ABCMeta):
  """Abstract ``Sharding`` interface which describes how a ``jax.Array`` is laid out
  across devices.
  """

  # Abstract methods below that subclasses should implement.

  @abc.abstractproperty
  def device_set(self) -> Set[Device]:
    """A ``set`` of global devices that this ``Sharding`` spans.

    In multi-controller JAX, the set of devices is global, i.e., includes
    non-addressable devices from other processes.
    """
    raise NotImplementedError('Subclasses should implement this method.')

  @abc.abstractmethod
  def devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Optional[Index]]:
    """A global mapping from device to the slice of the global data it contains.

    The devices in this mapping are global devices i.e. includes
    non-addressable devices from other processes.
    """
    raise NotImplementedError('Subclasses should implement this method.')

  @abc.abstractmethod
  def shard_shape(self, global_shape: Shape) -> Shape:
    """Returns the shape of the data on each device.

    The shard shape returned by this function is calculated from the global
    shape (it takes as an input) and the properties of the sharding.
    """
    raise NotImplementedError('Subclasses should implement this method.')

  @abc.abstractmethod
  def is_equivalent_to(self, other: Sharding, ndim: int) -> bool:
    """Returns True if two shardings put the same logical array
    (sharded/unsharded) on the same device(s).

    For example, every XLACompatibleSharding lowers to GSPMDSharding which
    is a general representation. So `jax.sharding.NamedSharding` is equivalent
    to `jax.sharding.PositionalSharding` if both of them lower to the same
    GSPMDSharding.
    """
    raise NotImplementedError('Subclasses should implement this method.')

  #############################################################################
  # Default implementations below that all subclasses will inherit.

  @functools.cached_property
  def addressable_devices(self) -> Set[Device]:
    """A set of devices that are addressable by the current process."""
    return {d for d in self.device_set
            if d.process_index == d.client.process_index()}

  @functools.cached_property
  def is_fully_addressable(self) -> bool:
    """True if the current process can address all of the devices in device_set.
    """
    # The pytype disable is because pytype can't recognize a cached property.
    return len(self.device_set) == len(self.addressable_devices)  # type: ignore

  @functools.lru_cache(maxsize=4096)
  def addressable_devices_indices_map(
      self, global_shape: Shape) -> Mapping[Device, Optional[Index]]:
    """A mapping from addressable device to the slice of global data it contains.

    ``addressable_devices_indices_map`` contains that part of
    ``device_indices_map`` that applies to the addressable devices.
    """
    return {d: ind for d, ind in self.devices_indices_map(global_shape).items()
            if d.process_index == d.client.process_index()}


# Shardings that inherit from XLACompatibleSharding should implement the
# `_device_assignment` property and `_to_xla_op_sharding` method.
@use_cpp_class(xc.XLACompatibleSharding)
class XLACompatibleSharding(Sharding, metaclass=abc.ABCMeta):
  """A `Sharding` that describes shardings expressible to XLA.

  Any ``Sharding`` that is a subclass of ``XLACompatibleSharding`` will work
  with all JAX APIs and transformations that use XLA.
  """

  # Abstract methods below that subclasses should implement.

  @abc.abstractproperty
  def _device_assignment(self) -> XLADeviceAssignment:
    raise NotImplementedError('Subclasses should implement this method.')

  @abc.abstractmethod
  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    raise NotImplementedError('Subclasses should implement this method.')

  #############################################################################
  # Default implementations below that all subclasses will inherit.

  @functools.lru_cache(maxsize=4096)
  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:
    op_sharding = self._to_xla_op_sharding(len(global_shape))
    gspmd_sharding = GSPMDSharding(self._device_assignment, op_sharding)
    return gspmd_sharding.devices_indices_map(global_shape)

  @functools.cached_property
  def _addressable_device_assignment(self) -> XLADeviceAssignment:
    return [d for d in self._device_assignment
            if d.process_index == d.client.process_index()]

  @functools.lru_cache(maxsize=4096)
  def shard_shape(self, global_shape: Shape) -> Shape:
    op_sharding = cast(xc.OpSharding, self._to_xla_op_sharding(len(global_shape)))
    if pxla.is_op_sharding_replicated(op_sharding):
      return global_shape
    partitions, _ = pxla.get_num_ways_dim_sharded(op_sharding)
    assert len(partitions) == len(global_shape), (len(partitions), len(global_shape))
    out = []
    for dim, (s, p) in enumerate(safe_zip(global_shape, partitions)):
      quotient, remainder = divmod(s, p)
      if remainder != 0:
        raise ValueError(
            f"Sharding {self} implies that array axis {dim} is partitioned "
            f"{p} times, but the dimension size is {s} "
            f"(full shape: {global_shape}, "
            f"per-dimension tiling factors: {partitions} should evenly divide "
            "the shape)")
      out.append(quotient)
    return tuple(out)

  def is_equivalent_to(self: XLACompatibleSharding,  # type: ignore
                       other: XLACompatibleSharding, ndim: int) -> bool:
    try:
      return (pxla.are_op_shardings_equal(self._to_xla_op_sharding(ndim),
                                          other._to_xla_op_sharding(ndim)) and
              self._device_assignment == other._device_assignment)
    # NotImplementedError is raised by PmapSharding because it can't lower
    # to OpSharding. So if `other` is a PmapSharding, default to a strict
    # equality check.
    except NotImplementedError:
      return self == other


@functools.lru_cache()
def _check_mesh_resource_axis(mesh, parsed_pspec):
  try:
    [mesh.shape[r] for p in parsed_pspec if p is not None
     for r in p]
  except KeyError as e:
    raise ValueError(f"Resource axis: {e.args[0]} of {parsed_pspec.user_spec} is "
                     "undefined.") from None


def _hashed_index(x) -> int:
  # This works for both `pjit`/`xmap` indices and `pmap` indices (which might
  # have an integer instead of a slice).
  assert all(v.step is None for v in x if isinstance(v, slice))
  return hash(tuple((v.start, v.stop) if isinstance(v, slice) else v for v in x))


@functools.lru_cache(maxsize=4096)
def device_replica_id_map(sharding, global_shape: Shape) -> Mapping[Device, int]:
  try:
    device_indices_map_fn = sharding.devices_indices_map
  except AttributeError:
    raise ValueError(
        f'Cannot calculate replica ids from sharding: {sharding}. Please '
        'create a device to index mapping for your sharding from which replica '
        'ids will be calculated.') from None

  index_to_replica: Dict[int, int] = Counter()
  out = {}
  for device, index in device_indices_map_fn(global_shape).items():
    h_index = _hashed_index(index)
    replica_id = index_to_replica[h_index]
    index_to_replica[h_index] += 1
    out[device] = replica_id
  return out


class _UnconstrainedPartitionSingleton:

  def __str__(self):
    return "UNCONSTRAINED"


# Unconstrained sentinel value for PartitionSpec, representing a dimension for
# which the user wants XLA to assign the best partitioning.
# TODO(yashkatariya): May rename to AUTO.
_UNCONSTRAINED_PARTITION = _UnconstrainedPartitionSingleton()


class PartitionSpec(tuple):
  """Tuple of integer specifying how a value should be partitioned.

  Each integer corresponds to how many ways a dimension is partitioned. We
  create a separate class for this so JAX's pytree utilities can distinguish it
  from a tuple that should be treated as a pytree.
  """

  # A sentinel value representing a dim is unconstrained.
  UNCONSTRAINED = _UNCONSTRAINED_PARTITION

  def __init__(self, *partitions):
    pass

  def __new__(cls, *partitions):
    return tuple.__new__(PartitionSpec, partitions)

  def __repr__(self):
    return "PartitionSpec%s" % tuple.__repr__(self)

  def __reduce__(self):
    return (PartitionSpec, tuple(self))


@use_cpp_class(xc.NamedSharding)
class NamedSharding(XLACompatibleSharding):
  r"""NamedSharding is a way to express ``Sharding``\s using named axes.

  ``Mesh`` and ``PartitionSpec`` can be used to express a ``Sharding`` with a name.

  ``Mesh`` is a NumPy array of JAX devices in a multi-dimensional grid,
  where each axis of the mesh has a name, e.g. 'x' or 'y'. Another name for
  ``Mesh`` is "logical mesh".

  ``PartitionSpec`` is a named tuple, whose elements can be a ``None``,
  a mesh axis or a tuple of mesh axes. Each element describes how an input
  dimension is partitioned across zero or more mesh dimensions. For example,
  PartitionSpec('x', 'y') is a PartitionSpec where the first dimension of data
  is sharded across ``x`` axis of the mesh, and the second dimension is sharded
  across ``y`` axis of the mesh.

  The Distributed arrays and automatic parallelization
  (https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#namedsharding-gives-a-way-to-express-shardings-with-names)
  goes into more details and has diagrams to help explain the concept about
  ``Mesh`` and ``PartitionSpec``.

  Args:
    mesh: A ``jax.sharding.Mesh`` object.
    spec: A ``jax.sharding.PartitionSpec`` object.

  Example:

    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
    >>> spec = P('x', 'y')
    >>> named_sharding = jax.sharding.NamedSharding(mesh, spec)
  """

  @use_cpp_method()
  def __init__(
      self, mesh: pxla.Mesh, spec: PartitionSpec, _parsed_pspec = None):

    self.mesh = mesh
    self.spec = spec
    self._parsed_pspec = _parsed_pspec
    self._preprocess()

  def __reduce__(self):
    return type(self), (self.mesh, self.spec)

  def _preprocess(self):
    # This split exists because you can pass `_parsed_pspec` that has been
    # modified from the original. For example: Adding extra dimension to
    # axis_resources for vmap handlers. In such cases you need to preserve the
    # `sync` attribute of parsed pspecs.
    # PartitionSpec is inferred from the parsed pspec in this case.
    # TODO(yaskatariya): Remove this and replace this with a normalized
    # representation of Parsed Pspec
    if self._parsed_pspec is None:
      from jax.experimental import pjit
      self._parsed_pspec, _, _ = pjit._prepare_axis_resources(
          self.spec, "NamedSharding spec")

    _check_mesh_resource_axis(self.mesh, self._parsed_pspec)

  def __repr__(self):
    return f'NamedSharding(mesh={dict(self.mesh.shape)}, spec={self.spec})'

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self.mesh, self._parsed_pspec))
    return self._hash

  def __eq__(self, other):
    if not isinstance(other, NamedSharding):
      return False
    if id(self) == id(other):
      return True
    if id(self.mesh) == id(other.mesh) and self._parsed_pspec == other._parsed_pspec:
      return True
    return self.mesh == other.mesh and self._parsed_pspec == other._parsed_pspec

  def is_compatible_aval(self, aval_shape: Shape):
    if len(aval_shape) < len(self._parsed_pspec):
      raise ValueError(
          f"Sharding {self} is only valid for values of rank at least "
          f"{len(self._parsed_pspec)}, but was applied to a value of rank "
          f"{len(aval_shape)}")

  @classmethod
  def _from_parsed_pspec(cls, mesh, parsed_pspec):
    return cls(mesh, parsed_pspec.get_partition_spec(), parsed_pspec)

  @functools.cached_property
  def device_set(self) -> Set[Device]:
    return set(self.mesh.devices.flat)

  @functools.cached_property
  def _device_assignment(self) -> XLADeviceAssignment:
    return list(self.mesh.devices.flat)

  @functools.lru_cache(maxsize=4096)
  def _to_xla_op_sharding(
      self,
      num_dimensions: int,
      axis_ctx: Optional[Union[mlir.SPMDAxisContext, mlir.ShardingContext]] = None
  ) -> xc.OpSharding:
    from jax.experimental.pjit import get_array_mapping

    array_mapping = get_array_mapping(self._parsed_pspec)
    # TODO(yashkatariya): Move away from sharding spec in NamedSharding
    # since we don't really need sharding spec.
    sharding_spec = pxla.new_mesh_sharding_specs(
        self.mesh.shape, self.mesh.axis_names)(num_dimensions, array_mapping)
    # Used in `with_sharding_constraint`.
    special_axes = {}
    # Manual axes is only used with xmap.
    if axis_ctx is not None and isinstance(axis_ctx, mlir.SPMDAxisContext):
      axis_names = self.mesh.axis_names
      # Ignore type because mypy doesn't recognize the `hasattr` check above.
      for manual_axis in axis_ctx.manual_axes:  # type: ignore
        special_axes[axis_names.index(manual_axis)] = xc.OpSharding.Type.MANUAL
    return sharding_spec.sharding_proto(special_axes=special_axes)


# TODO(yashkatariya); Remove this after 3 months per the deprecation policy.
MeshPspecSharding = NamedSharding


@functools.lru_cache()
def _get_replicated_op_sharding():
  proto = xc.OpSharding()
  proto.type = xc.OpSharding.Type.REPLICATED
  return proto


@use_cpp_class(xc.SingleDeviceSharding)
class SingleDeviceSharding(XLACompatibleSharding):
  """A subclass of ``XLACompatibleSharding`` that places its data on a single device.

  Args:
    device: A single :py:class:`Device`.

  Example:

    >>> single_device_sharding = jax.sharding.SingleDeviceSharding(
    ...     jax.devices()[0])
  """

  @use_cpp_method()
  def __init__(self, device: Device):
    self._device = device

  def __reduce__(self):
    return type(self), (self._device,)

  def __repr__(self):
    return f"SingleDeviceSharding(device={repr(self._device)})"

  def __hash__(self):
    return hash(self._device)

  def __eq__(self, other):
    if not isinstance(other, SingleDeviceSharding):
      return False
    if id(self) == id(other):
      return True
    return self._device == other._device

  @property
  def device_set(self) -> Set[Device]:
    return {self._device}

  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:  # type: ignore
    return {self._device: (slice(None),) * len(global_shape)}

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return [self._device]

  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    return _get_replicated_op_sharding()


@use_cpp_class(xc.PmapSharding)
class PmapSharding(XLACompatibleSharding):

  @use_cpp_method()
  def __init__(self, devices: np.ndarray, sharding_spec: pxla.ShardingSpec):
    self.devices = devices
    # The sharding spec should be pmap's sharding spec.
    self.sharding_spec = sharding_spec

  def __reduce__(self):
    return type(self), (self.devices, self.sharding_spec)

  def __eq__(self, other):
    if not isinstance(other, PmapSharding):
      return False
    if id(self) == id(other):
      return True
    return (self.sharding_spec == other.sharding_spec and
            np.array_equal(self.devices, other.devices))

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((tuple(self.devices.flat), self.sharding_spec))
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
  def default(cls, shape: Shape, sharded_dim: int = 0) -> PmapSharding:
    """Creates a `PmapSharding` which matches the implicit device order used by
    `pmap`.

    Args:
      shape: The shape of the input array.
      sharded_dim: Dimension the input array is sharded on. Defaults to 0.
    """
    # The dtype doesn't matter here. Its only used for creating the
    # sharding_spec.
    aval = core.ShapedArray(shape, np.int32)
    sharding_spec = pxla._create_pmap_sharding_spec(aval, sharded_dim)

    num_ways_sharded = None
    for s in sharding_spec.sharding:
      if isinstance(s, pxla.Unstacked):
        num_ways_sharded = s.size
    if num_ways_sharded is None:
      raise NotImplementedError(
          '`None` to sharded_dim is not supported. Please file a jax '
          'issue if you need this feature.')

    pmap_devices = jax.local_devices()[:num_ways_sharded]
    return cls(pmap_devices, sharding_spec)

  @functools.cached_property
  def device_set(self) -> Set[Device]:
    return set(self.devices.flat)

  @functools.lru_cache(maxsize=4096)
  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:
    indices = pxla.spec_to_indices(global_shape, self.sharding_spec)
    return dict(safe_zip(self.devices.flat, indices))  # type: ignore[arg-type]

  @functools.cached_property
  def _device_assignment(self) -> XLADeviceAssignment:
    return list(self.devices.flat)

  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    raise NotImplementedError("pmap doesn't use OpSharding.")

  @functools.lru_cache(maxsize=4096)
  def shard_shape(self, global_shape: Shape) -> Shape:
    sharded_dim = None
    sharded_dim_size = None
    for i, s in enumerate(self.sharding_spec.sharding):
      if isinstance(s, pxla.Unstacked):
        sharded_dim = i
        sharded_dim_size = s.size
        break
    if sharded_dim is None:
      return global_shape
    if global_shape[sharded_dim] != sharded_dim_size:
      raise ValueError(
          f'The sharded dimension must be equal to the number of '
          f'devices passed to PmapSharding. Got sharded dimension {sharded_dim} '
          f'with value {global_shape[sharded_dim]} in shape {global_shape} and '
          f'the number of devices={len(self._device_assignment)}')
    return global_shape[:sharded_dim] + global_shape[sharded_dim+1:]


class PositionalSharding(XLACompatibleSharding):
  _devices: List[xc.Device]
  _ids: np.ndarray  # dtype DeviceIdSet

  def __init__(self, devices: Union[Sequence[xc.Device], np.ndarray]):
    if not isinstance(devices, np.ndarray):
      devices = np.array(devices, dtype='object')
    if not devices.size:
      raise ValueError(f"{self.__class__.__name__}.__init__ requires at least "
                       f"one device, got {devices}")
    self._devices = list(devices.flat)
    name = self._devices[0].platform.upper()
    self._ids = np.array([DeviceIdSet(name, i) for i in range(devices.size)],
                         dtype='object').reshape(devices.shape)

  shape = property(op.attrgetter('_ids.shape'))
  ndim = property(op.attrgetter('_ids.ndim'))

  def __repr__(self) -> str:
    cls_name = self.__class__.__name__
    ids = self._ids.copy()
    platform_name = self._devices[0].platform.upper()
    for idx, x in np.ndenumerate(ids):
      ids[idx] = DeviceIdSet(platform_name, *(self._devices[i].id for i in x))
    body = np.array2string(ids, prefix=cls_name + '(', suffix=')',
                           max_line_width=100)
    return f'{cls_name}({body})'

  def reshape(self, *shape):
    return self.remake(self._devices, self._ids.reshape(*shape))

  def transpose(self, *axes):
    return self.remake(self._devices, self._ids.transpose(*axes))
  T = property(transpose)

  def replicate(self, axis=None, keepdims=True):
    new_ids = self._ids.sum(axis=axis, keepdims=keepdims)  # union
    return self.remake(self._devices, new_ids)

  @classmethod
  def remake(
      cls, devices: List[xc.Device], ids: np.ndarray) -> PositionalSharding:
    self = cls.__new__(cls)
    self._devices = devices
    self._ids = ids
    return self

  # Hashable

  def __hash__(self) -> int:
    return id(self._devices)

  def __eq__(self, other) -> bool:
    return (isinstance(other, PositionalSharding) and
            id(self._devices) == id(other._devices) and
            bool(np.all(self._ids == other._ids)))

  # Sharding interface

  @functools.cached_property
  def device_set(self) -> set[xc.Device]:
    return set(self._devices)

  # XLACompatibleSharding interface

  @functools.lru_cache(maxsize=4096)
  def _to_xla_op_sharding(self, num_dimensions: int, axis_ctx=None):
    assert axis_ctx is None

    pbuf = xc.OpSharding()
    if self.shape == (1,) * self.ndim:
      pbuf.type = xc.OpSharding.Type.REPLICATED
      return pbuf

    shape = self.shape[self.ndim - num_dimensions:]  # 'rank promotion' of val
    set_size, = {len(device_set) for device_set in self._ids.flat}
    pbuf.type = xc.OpSharding.Type.OTHER
    if set_size > 1:
      pbuf.last_tile_dims = [xc.OpSharding.Type.REPLICATED]
      pbuf.tile_assignment_dimensions = (*shape, set_size)
    else:
      pbuf.tile_assignment_dimensions = shape
    pbuf.tile_assignment_devices = [i for ids in self._ids.flat for i in ids]
    return pbuf

  @property
  def _device_assignment(self) -> list[xc.Device]:
    return self._devices

class DeviceIdSet:
  _name: str
  _ids: FrozenSet[int]
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


@use_cpp_class(xc.GSPMDSharding if xla_extension_version >= 129 else xc.OpShardingSharding)  # type: ignore
class GSPMDSharding(XLACompatibleSharding):

  @use_cpp_method()
  def __init__(self, devices: Sequence[Device], op_sharding: xc.OpSharding):
    self._devices = tuple(devices)
    self._op_sharding = op_sharding

  def __reduce__(self):
    return type(self), (self._devices, self._op_sharding)

  @functools.cached_property
  def _op_sharding_hash(self):
    return hash(xc.HloSharding.from_proto(self._op_sharding))

  def __eq__(self, other):
    if not isinstance(other, GSPMDSharding):
      return False
    if id(self) == id(other):
      return True
    return (pxla.are_op_shardings_equal(self._op_sharding, other._op_sharding) and
            self._devices == other._devices)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((self._devices, self._op_sharding_hash))
    return self._hash

  def __repr__(self):
    return f'GSPMDSharding({repr(xc.HloSharding.from_proto(self._op_sharding))})'

  def is_compatible_aval(self, aval_shape: Shape):
    num_ways_dim_sharded, _ = pxla.get_num_ways_dim_sharded(self._op_sharding)
    if len(aval_shape) < len(num_ways_dim_sharded):
      raise ValueError(
          f"Sharding {self} is only valid for values of rank at least "
          f"{len(num_ways_dim_sharded)}, but was applied to a value of rank "
          f"{len(aval_shape)}")

  @functools.cached_property
  def device_set(self) -> Set[Device]:
    return set(self._devices)

  @functools.lru_cache(maxsize=4096)
  def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]:
    indices = pxla.op_sharding_to_indices(self._op_sharding, global_shape,
                                          len(self._devices))
    return dict(safe_zip(self._devices, indices))

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    return list(self._devices)

  def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
    return self._op_sharding

  @classmethod
  def get_replicated(cls, device_assignment):
    proto = _get_replicated_op_sharding()
    return cls(device_assignment, proto)


# TODO(yashkatariya); Remove OpShardingSharding after 3 months from Feb 17, 2023
# per the deprecation policy.
OpShardingSharding = GSPMDSharding
