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

from __future__ import annotations

from collections.abc import Sequence
import collections
import dataclasses
import functools
from typing import Any, Union

from jax._src import config
from jax._src.util import use_cpp_class, cache, use_cpp_method
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir.dialects import sdy
from jax._src import mesh as mesh_lib
from jax._src.mesh import AxisType
from jax._src.partition_spec import PartitionSpec
from jax._src import sharding as JSharding
import numpy as np

Shape = tuple[int, ...]
Device = xc.Device
Index = tuple[slice, ...]
XLADeviceAssignment = Sequence[Device]


class AUTO:

  def __init__(self, mesh: mesh_lib.Mesh):
    self.mesh = mesh

  def _to_sdy_sharding(self, ndim: int) -> SdyArray:
    dim_shardings = [SdyDim(axes=[], is_open=True)
                     for _ in range(ndim)]
    return SdyArray(mesh_shape=self.mesh.shape_tuple,
                    dim_shardings=dim_shardings)

class UnspecifiedValue:
  def __repr__(self):
    return "UnspecifiedValue"
UNSPECIFIED = UnspecifiedValue()


MeshAxisName = Any

"""
ArrayMapping specifies how an ndarray should map to mesh axes.

Note that the ordering is crucial for the cases when this mapping is non-injective
(i.e. when multiple mesh axes map to the same positional axis). Then, the
order of entries of the mapping determines a major-to-minor order on mesh axes,
according to which chunks of the value along the repeated dimension will be assigned.

For example, consider a mapping {'x': 1, 'y': 1} and a mesh with shape {'x': 2, 'y': 3}.
The second dimension of the value would get chunked into 6 pieces, and assigned to the
mesh in a way that treats 'y' as the fastest changing (minor) dimension. In this case,
that would mean that a flat list of chunks would get assigned to a flattened list of
mesh devices without any modifications. If the mapping was {'y': 1, 'x': 1}, then the
mesh devices ndarray would have to be transposed before flattening and assignment.
"""
ArrayMapping = collections.OrderedDict[MeshAxisName, int]
ArrayMappingOrAutoOrUnspecified = Union[ArrayMapping, AUTO, UnspecifiedValue]


@use_cpp_class(xc.NamedSharding)
class NamedSharding(JSharding.Sharding):
  r"""A :class:`NamedSharding` expresses sharding using named axes.

  A :class:`NamedSharding` is a pair of a :class:`Mesh` of devices and
  :class:`PartitionSpec` which describes how to shard an array across that
  mesh.

  A :class:`Mesh` is a multidimensional NumPy array of JAX devices,
  where each axis of the mesh has a name, e.g. ``'x'`` or ``'y'``.

  A :class:`PartitionSpec` is a tuple, whose elements can be a ``None``,
  a mesh axis, or a tuple of mesh axes. Each element describes how an input
  dimension is partitioned across zero or more mesh dimensions. For example,
  ``PartitionSpec('x', 'y')`` says that the first dimension of data
  is sharded across ``x`` axis of the mesh, and the second dimension is sharded
  across ``y`` axis of the mesh.

  The Distributed arrays and automatic parallelization
  (https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#namedsharding-gives-a-way-to-express-shardings-with-names)
  tutorial has more details and diagrams that explain how
  :class:`Mesh` and :class:`PartitionSpec` are used.

  Args:
    mesh: A :class:`jax.sharding.Mesh` object.
    spec: A :class:`jax.sharding.PartitionSpec` object.

  Examples:

    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
    >>> spec = P('x', 'y')
    >>> named_sharding = jax.sharding.NamedSharding(mesh, spec)
  """

  mesh: mesh_lib.Mesh | mesh_lib.AbstractMesh
  spec: PartitionSpec
  _memory_kind: str | None
  _logical_device_ids: tuple[int, ...] | None

  @use_cpp_method()
  def __init__(
      self, mesh: mesh_lib.Mesh | mesh_lib.AbstractMesh, spec: PartitionSpec, *,
      memory_kind: str | None = None, _logical_device_ids=None):
    self.mesh = mesh
    self.spec = spec
    self._memory_kind = memory_kind
    self._logical_device_ids = _logical_device_ids
    check_pspec(self.mesh, self.spec)

  def __repr__(self):
    mem = '' if self.memory_kind is None else f', memory_kind={self.memory_kind}'
    ldi = ('' if self._logical_device_ids is None else
           f', logical_device_ids={self._logical_device_ids}')
    mesh_repr = f"{str(self.mesh)}"
    return f'NamedSharding(mesh={mesh_repr}, spec={self.spec}{mem}{ldi})'

  def __reduce__(self):
    return (type(self), (self.mesh, self.spec),
            {'memory_kind': self.memory_kind,
             '_logical_device_ids': self._logical_device_ids})

  @property
  def memory_kind(self) -> str | None:
    return self._memory_kind

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash(
          (self.mesh, self.memory_kind, self.spec, self._logical_device_ids))
    return self._hash

  def __eq__(self, other):
    if not isinstance(other, NamedSharding):
      return False
    if self is other:
      return True
    if (self.spec != other.spec
        or self.memory_kind != other.memory_kind
        or self._logical_device_ids != other._logical_device_ids):
      return False
    return self.mesh is other.mesh or self.mesh == other.mesh

  def check_compatible_aval(self, aval_shape: Shape) -> None:
    if len(aval_shape) < len(self.spec):
      extra_msg = (' For scalars the PartitionSpec should be P()'
                   if len(aval_shape) == 0 else '')
      raise ValueError(
          f"Sharding {self} is only valid for values of rank at least "
          f"{len(self.spec)}, but was applied to a value of rank "
          f"{len(aval_shape)}.{extra_msg}")

  @property
  def num_devices(self) -> int:
    return self.mesh.size

  @property
  def device_set(self) -> set[Device]:
    if isinstance(self.mesh, mesh_lib.AbstractMesh):
      raise ValueError(
          'device_set is not implemented for `jax.sharding.AbstractMesh`.')
    return self.mesh._flat_devices_set

  @property
  def _device_assignment(self) -> XLADeviceAssignment:
    if isinstance(self.mesh, mesh_lib.AbstractMesh):
      raise ValueError('_device_assignment is not implemented for'
                       ' `jax.sharding.AbstractMesh`.')
    return self.mesh._flat_devices_tuple

  @property
  def is_fully_addressable(self) -> bool:
    if isinstance(self.mesh, mesh_lib.AbstractMesh):
      raise ValueError('is_fully_addressable is not implemented for '
                       '`jax.sharding.AbstractMesh`.')
    # Speed up `is_fully_addressable` since there is a high chance that the
    # mesh across multiple NamedSharding objects will be the same.
    if config.enable_empty_arrays.value:
      return self._internal_device_list.is_fully_addressable  # type: ignore
    return not self.mesh.is_multi_process

  @property
  def _is_concrete(self) -> bool:
    if isinstance(self.mesh, mesh_lib.AbstractMesh):
      return False
    return True

  @property
  def addressable_devices(self) -> set[Device]:
    if isinstance(self.mesh, mesh_lib.AbstractMesh):
      raise ValueError('addressable_devices is not implemented for '
                       '`jax.sharding.AbstractMesh`.')
    # Override addressable devices because there is a high chance that the mesh
    # across multiple NamedSharding objects will be the same.
    return self.mesh._local_devices_set

  @functools.cached_property
  def is_fully_replicated(self) -> bool:
    if self.mesh.size == 1:
      return True
    array_mapping = get_array_mapping(self.spec)
    mesh_shape = self.mesh.shape
    num_partitions = 1
    for name in array_mapping:  # type: ignore
      num_partitions *= mesh_shape[name]
    return num_partitions == 1

  def with_memory_kind(self, kind: str) -> NamedSharding:
    return NamedSharding(self.mesh, self.spec, memory_kind=kind)

  def with_spec(self, spec: PartitionSpec | Sequence[Any]) -> NamedSharding:
    if not isinstance(spec, PartitionSpec):
      spec = PartitionSpec(*spec)
    return NamedSharding(self.mesh, spec, memory_kind=self.memory_kind)

  def _to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding:
    return named_sharding_to_xla_hlo_sharding(self, num_dimensions)

  def _to_sdy_sharding(self, num_dimensions: int) -> SdyArray:
    dim_shardings = [SdyDim(axes=[], is_open=False)
                     for _ in range(num_dimensions)]
    for i, dim_spec in enumerate(self.spec):
      if dim_spec is PartitionSpec.UNCONSTRAINED:
        dim_shardings[i].is_open = True
      elif dim_spec is None:
        # Already empty and closed sharding.
        pass
      else:
        dim_spec = dim_spec if isinstance(dim_spec, tuple) else (dim_spec,)
        dim_shardings[i].axes = dim_spec
    return SdyArray(mesh_shape=self.mesh.shape_tuple,
                    dim_shardings=dim_shardings,
                    logical_device_ids=self._logical_device_ids,
                    unreduced_axes=self.spec.unreduced)

NamedSharding.__module__ = 'jax.sharding'

def get_array_mapping(
    axis_resources: PartitionSpec | AUTO | UnspecifiedValue
) -> ArrayMappingOrAutoOrUnspecified:
  if isinstance(axis_resources, (AUTO, UnspecifiedValue)):
    return axis_resources
  d = collections.OrderedDict()
  for i, axes in enumerate(axis_resources):
    if axes is None or axes is PartitionSpec.UNCONSTRAINED:
      continue
    axes = axes if isinstance(axes, tuple) else (axes,)
    for axis in axes:
      d[axis] = i
  return d

@dataclasses.dataclass
class SdyDim:
  axes: Sequence[str]
  is_open: bool
  priority: int | None = None

  def build(self) -> sdy.DimensionShardingAttr:
    return sdy.DimensionShardingAttr.get(
        [sdy.AxisRefAttr.get(axis) for axis in self.axes],
        is_closed=not self.is_open, priority=self.priority)

  def __repr__(self):
    return f'SdyDim({self._custom_repr()})'

  def _custom_repr(self):
    axes_repr = ', '.join(f"'{a}'" for a in self.axes)
    open_repr = ''
    if self.is_open:
      open_repr = ', ?' if self.axes else '?'
    priority_repr = '' if self.priority is None else f'p{self.priority}'
    return f'{{{axes_repr}{open_repr}}}{priority_repr}'

def _get_axes(axes, mesh_shape):
  if not axes:
    return ()
  assert mesh_shape is not None
  # Sort wrt mesh axis names so order is deterministic and doesn't hang in
  # McJAX.
  return tuple(n for n, _ in mesh_shape if n in axes)

@dataclasses.dataclass(kw_only=True)
class SdyArray:
  mesh_shape: tuple[tuple[str, int], ...] | None
  dim_shardings: Sequence[SdyDim]
  logical_device_ids: tuple[int, ...] | None = None
  replicated_axes: tuple[str, ...] = ()
  unreduced_axes: frozenset[str] = frozenset()

  def build(self) -> sdy.TensorShardingAttr:
    if self.mesh_shape is None:
      mesh_attr = sdy.MeshAttr.get([])
    else:
      ldi = ([] if self.logical_device_ids is None else
             list(self.logical_device_ids))
      mesh_attr = sdy.MeshAttr.get(
          [sdy.MeshAxisAttr.get(name, size) for name, size in self.mesh_shape],
          ldi)

    replicated_axes = _get_axes(self.replicated_axes, self.mesh_shape)
    unreduced_axes = _get_axes(self.unreduced_axes, self.mesh_shape)
    return sdy.TensorShardingAttr.get(
        mesh_attr,
        [dim_sharding.build() for dim_sharding in self.dim_shardings],
        replicated_axes=[sdy.AxisRefAttr.get(axis) for axis in replicated_axes],
        unreduced_axes=[sdy.AxisRefAttr.get(axis) for axis in unreduced_axes])

  def __repr__(self):
    dim_sharding_repr = ', '.join(
        d._custom_repr() for d in self.dim_shardings)
    device_id_repr = (f', device_ids={self.logical_device_ids}'
                      if self.logical_device_ids is not None else '')
    rar = (f', replicated_axes={self.replicated_axes}'
           if self.replicated_axes else '')
    return f"SdyArray([{dim_sharding_repr}]{device_id_repr}{rar})"


# TODO(yashkatariya): Upstream this into `_to_sdy_sharding` maybe with an extra
# parameter to it `_to_sdy_sharding(self, ndim, modify_wrt_axis_types=False)`
def modify_sdy_sharding_wrt_axis_types(sdy_sharding: SdyArray, mesh):
  if mesh._any_axis_auto:
    dim_shardings, used_axes = [], []  # type: ignore
    for d in sdy_sharding.dim_shardings:
      # TODO(yashkatariya): Maybe if any mesh axis is auto, mark all axes as open?
      dim_shardings.append(SdyDim(axes=[], is_open=True)
                           if not d.axes and not d.is_open else d)
      used_axes.extend(d.axes)
    remaining_axes = set(mesh.axis_names) - set(used_axes)
    replicated_axes = tuple(r for r in remaining_axes
                            if mesh._name_to_type[r] == mesh_lib.AxisType.Explicit)
    return SdyArray(mesh_shape=sdy_sharding.mesh_shape,
                    dim_shardings=dim_shardings,
                    logical_device_ids=sdy_sharding.logical_device_ids,
                    replicated_axes=replicated_axes)
  return sdy_sharding


@cache(max_size=4096, trace_context_in_key=False)
def named_sharding_to_xla_hlo_sharding(
    self, num_dimensions: int) -> xc.HloSharding:
  mesh_shape = self.mesh.shape
  array_mapping = get_array_mapping(self.spec)
  mesh_axis_pos = {name: i for i, name in enumerate(self.mesh.axis_names)}

  special_axes = {}
  manual_axes = frozenset(self.mesh.manual_axes)
  if manual_axes:
    axis_names = self.mesh.axis_names
    for manual_axis in manual_axes:
      special_axes[axis_names.index(manual_axis)] = xc.OpSharding.Type.MANUAL

  replicated_mesh_axes = []
  for i, (axis_name, axis_val) in enumerate(mesh_shape.items()):
    if axis_name not in array_mapping:  # type: ignore
      replicated_mesh_axes.append((i, axis_val))

  if len(replicated_mesh_axes) == len(mesh_shape) and not special_axes:
    return xc.HloSharding.replicate()

  mesh_permutation = []
  new_mesh_shape = [1] * num_dimensions
  for name, pos in sorted(array_mapping.items(), key=lambda x: x[1]):  # type: ignore
    new_mesh_shape[pos] *= mesh_shape[name]
    mesh_permutation.append(mesh_axis_pos[name])

  last_tile_dims = []
  if replicated_mesh_axes:
    axes_by_type: dict[Any, list[int]] = collections.defaultdict(list)
    size_by_type = collections.defaultdict(lambda: 1)  # type: ignore
    assert {x[0] for x in replicated_mesh_axes}.issuperset(set(special_axes.keys()))
    for i, size in replicated_mesh_axes:
      ty = special_axes.get(i, xc.OpSharding.Type.REPLICATED)
      axes_by_type[ty].append(i)
      size_by_type[ty] *= size
    for ty, axes in sorted(axes_by_type.items(), key=lambda x: x[0].value):
      last_tile_dims.append(ty)
      new_mesh_shape.append(size_by_type[ty])
      mesh_permutation.extend(axes)

  # Explanation of the parameters of `HloSharding.iota_tile`.
  # This is the HloShardingV2 format:
  #   * dims: How many ways each dimension is sharded.
  #       Replicated/Manual dims are added added at the end
  #   * reshape_dims: This is the just the shape of the mesh.
  #   * transpose_perm: This is the order in which mesh axes in PartitionSpec
  #       appear relative to mesh.axis_names order.
  #   * subgroup_types: List of type of OpSharding. Type can be REPLICATED and MANUAL.
  # Let's see an example:
  #   Consider input_shape=(8, 4, 2, 2), mesh={'a': 2, 'b': 2, 'c': 2, 'd': 2}
  #   and partition_spec=P(None, ('d', 'b'), 'c').
  #   Arguments to iota_tile will be:
  #     dims = [1, 4, 2, 1, 2]  # 'a' is replicated hence `2` is at the end.
  #     reshape_dims = [2, 2, 2, 2]
  #     transpose_perm = [3, 1, 2, 0]  # 'a' is replicated hence 0 is at the end
  #     subgroup_types = [xc.OpSharding.Type.REPLICATED]
  dims = new_mesh_shape
  reshape_dims = self.mesh.axis_sizes
  if self._logical_device_ids is None:
    return xc.HloSharding.iota_tile(
        dims=dims, reshape_dims=reshape_dims, transpose_perm=mesh_permutation,
        subgroup_types=last_tile_dims)
  else:
    return xc.HloSharding.subgroup_with_device_ordering(
        np.asarray(self._logical_device_ids)
        .reshape(dims).reshape(reshape_dims).transpose(mesh_permutation)
        .reshape(dims), subgroup_types=last_tile_dims)


def array_mapping_to_axis_resources(array_mapping: ArrayMapping):
  if not array_mapping:
    return PartitionSpec()
  max_index = -1
  reverse_map = collections.defaultdict(list)
  for axis, index in array_mapping.items():
    reverse_map[index].append(axis)
    if index > max_index:
      max_index = index
  partitions = []
  for i in range(max_index + 1):
    axis = reverse_map[i]
    if axis:
      partitions.append(axis[0] if len(axis) == 1 else tuple(axis))
    else:
      partitions.append(None)
  return PartitionSpec(*partitions)


@cache(max_size=128, trace_context_in_key=False)
def check_pspec(mesh, spec, _manual_axes=frozenset()):
  _check_unique_resources(spec, "NamedSharding spec", mesh)
  _check_mesh_resource_axis(mesh, spec)
  _check_mesh_unreduced(mesh, spec)

class DuplicateSpecError(Exception):
  def __init__(self, message, mesh, pspec):
    super().__init__(message)
    self.message = message
    self.mesh = mesh
    self.pspec = pspec

  def __str__(self):
    return f"{self.message}"

def _check_unique_resources(pspec: PartitionSpec, arg_name: str, mesh=None
                            ) -> None:
  resource_counts: dict[MeshAxisName, int] = {}
  duplicate = False
  for d in pspec:
    if d is PartitionSpec.UNCONSTRAINED or d is None:
      continue
    d = d if isinstance(d, tuple) else (d,)
    for resource in d:
      count = resource_counts.get(resource, 0)
      if count > 0:
        duplicate = True
      resource_counts[resource] = count + 1
  if duplicate:
    multiple_uses = [r for r, c in resource_counts.items() if c > 1]
    raise DuplicateSpecError(
        message=(
            f'A single {arg_name} specification can map every mesh axis to at'
            f' most one positional dimension, but {pspec} has duplicate entries'
            f' for {mesh_lib.show_axes(multiple_uses)}'),
        mesh=mesh, pspec=pspec)

def _check_mesh_resource_axis(mesh, pspec):
  for p in pspec:
    if p is PartitionSpec.UNCONSTRAINED or p is None:
      continue
    p = p if isinstance(p, tuple) else (p,)
    for r in p:
      if r not in mesh.axis_names:
        raise ValueError(
            f"Resource axis: {r} of {pspec} "
            f"is not found in mesh: {tuple(mesh.shape.keys())}.")
    if not all(mesh._name_to_type[p[0]] == mesh._name_to_type[r] for r in p):
      raise ValueError(
          'AxisTypes should be the same in a tuple subset of PartitionSpec:'
          f' {pspec}. Got subset {p} with axis'
          f' types: ({", ".join(str(mesh._name_to_type[r]) for r in p)})')
  if (AxisType.Auto not in mesh._axis_types_dict and
      PartitionSpec.UNCONSTRAINED in pspec):
    raise ValueError(
        f'{pspec} cannot contain'
        ' `P.UNCONSTRAINED` when no mesh axis_types are `Auto`. Got mesh'
        f' axis_types: {mesh._axis_types_dict}')

def _check_mesh_unreduced(mesh, pspec):
  for u in pspec.unreduced:
    if u not in mesh.axis_names:
      raise ValueError(
          f'Unreduced axes {u} is not found in {mesh.axis_names=}. '
          f'Got {pspec=}')
    if mesh._name_to_type[u] in (AxisType.Auto, AxisType.Manual):
      raise ValueError(
          'Unreduced axes can only refer to mesh axes that is of type'
          f' `Explicit`. Got unreduced axes: {pspec.unreduced} and'
          f' mesh: {mesh}')
