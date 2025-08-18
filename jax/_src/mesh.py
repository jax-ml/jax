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
"""Definitions of Mesh and AbstractMesh"""

from __future__ import annotations

import collections
from collections.abc import Hashable, Sequence
import contextlib
import dataclasses
import enum
import functools
import math
import threading
from typing import Any, NamedTuple

import numpy as np

from jax._src import config as jax_config
from jax._src import xla_bridge as xb
from jax._src.util import safe_zip, cache, tuple_delete
from jax._src.lib import xla_client as xc

zip, unsafe_zip = safe_zip, zip

MeshAxisName = Any
ResourceAxisName = Hashable


def show_axes(axes):
  return ", ".join(sorted(f"`{a}`" for a in axes))


class ResourceEnv(NamedTuple):
  physical_mesh: Mesh

  def with_mesh(self, mesh: Mesh):
    overlap = set(mesh.axis_names) & (self.resource_axes - set(self.physical_mesh.axis_names))
    if overlap:
      raise ValueError(f"Cannot update the mesh of the current resource "
                       f"environment. The new mesh shadows already defined axes "
                       f"{show_axes(overlap)}")
    return self._replace(physical_mesh=mesh)

  @property
  def physical_resource_axes(self) -> set[ResourceAxisName]:
    return set(self.physical_mesh.axis_names)

  @property
  def resource_axes(self) -> set[ResourceAxisName]:
    return self.physical_resource_axes

  @property
  def shape(self):
    return self.physical_mesh.shape

  @property
  def local_shape(self):
    return self.physical_mesh.local_mesh.shape

  def __repr__(self):
    mesh_repr = ", ".join(
        f"'{k}': {v}" for k, v in self.physical_mesh.shape.items())
    return f"ResourceEnv(mesh=Mesh({mesh_repr}))"


@cache(max_size=128, trace_context_in_key=False)
def _get_local_mesh(global_mesh: Mesh, process_index: int) -> Mesh:
  if global_mesh.empty:
    return global_mesh
  is_local_device = np.vectorize(
      lambda d: d.process_index == process_index, otypes=[bool])(global_mesh.devices)
  subcube_indices = []
  # We take the smallest slice of each dimension that doesn't skip any local device.
  for axis in range(global_mesh.devices.ndim):
    other_axes = tuple_delete(tuple(range(global_mesh.devices.ndim)), axis)
    # NOTE: This re-reduces over many axes multiple times, so we could definitely
    #       optimize it, but I hope it won't be a bottleneck anytime soon.
    local_slices = is_local_device.any(other_axes, keepdims=False)
    nonzero_indices = np.flatnonzero(local_slices)
    start, end = int(np.min(nonzero_indices)), int(np.max(nonzero_indices))
    subcube_indices.append(slice(start, end + 1))
  subcube_indices_tuple = tuple(subcube_indices)
  # We only end up with all conditions being true if the local devices formed a
  # subcube of the full array. This is because we were biased towards taking a
  # "hull" spanned by the devices, and in case the local devices don't form a
  # subcube that hull will contain non-local devices.
  if not is_local_device[subcube_indices_tuple].all():
    raise ValueError(
        "When passing host local inputs to pjit, devices connected to a single"
        " host must form a contiguous subcube of the global device mesh"
    )
  return Mesh(global_mesh.devices[subcube_indices_tuple], global_mesh.axis_names)


class AxisType(enum.Enum):
  Auto = enum.auto()
  Explicit = enum.auto()
  Manual = enum.auto()

  def __repr__(self):
    return self.name

def _normalize_axis_types(axis_names, axis_types, name):
  axis_types = ((AxisType.Auto,) * len(axis_names)
                if axis_types is None else axis_types)
  if not isinstance(axis_types, tuple):
    axis_types = (axis_types,)

  if not all(isinstance(a, AxisType) for a in axis_types):
    raise TypeError(
        f"axis_types passed to {name} must be of type `jax.sharding.AxisType`."
        f" Got {axis_types} of type {tuple(type(a) for a in axis_types)}")
  if len(axis_names) != len(axis_types):
    raise ValueError(
        "Number of axis names should match the number of axis_types. Got"
        f" axis_names={axis_names} and axis_types={axis_types}")
  return axis_types

def all_axis_types_match(axis_types, ty: AxisType) -> bool:
  if not axis_types:
    return False
  return all(t == ty for t in axis_types)

def any_axis_types_match(axis_types, ty: AxisType) -> bool:
  if not axis_types:
    return False
  return any(t == ty for t in axis_types)


class BaseMesh:
  axis_names: tuple[MeshAxisName, ...]
  shape_tuple: tuple[tuple[str, int], ...]
  axis_types: tuple[AxisType, ...]

  @functools.cached_property
  def are_all_axes_manual(self) -> bool:
    return all_axis_types_match(self.axis_types, AxisType.Manual)

  @functools.cached_property
  def are_all_axes_auto(self) -> bool:
    return all_axis_types_match(self.axis_types, AxisType.Auto)

  @functools.cached_property
  def are_all_axes_explicit(self) -> bool:
    return all_axis_types_match(self.axis_types, AxisType.Explicit)

  @functools.cached_property
  def _are_all_axes_auto_or_manual(self) -> bool:
    if not self.axis_types:
      return False
    return all(t == AxisType.Auto or t == AxisType.Manual
               for t in self.axis_types)

  @functools.cached_property
  def _any_axis_manual(self) -> bool:
    return any_axis_types_match(self.axis_types, AxisType.Manual)

  @functools.cached_property
  def _any_axis_auto(self) -> bool:
    return any_axis_types_match(self.axis_types, AxisType.Auto)

  @functools.cached_property
  def _any_axis_explicit(self) -> bool:
    return any_axis_types_match(self.axis_types, AxisType.Explicit)

  @functools.cached_property
  def _any_axis_auto_or_manual(self) -> bool:
    if not self.axis_types:
      return False
    return any(t == AxisType.Auto or t == AxisType.Manual
               for t in self.axis_types)

  @functools.cached_property
  def auto_axes(self):
    return tuple(n for n, t in safe_zip(self.axis_names, self.axis_types)
                 if t == AxisType.Auto)

  @functools.cached_property
  def explicit_axes(self):
    return tuple(n for n, t in safe_zip(self.axis_names, self.axis_types)
                 if t == AxisType.Explicit)

  @functools.cached_property
  def manual_axes(self):
    return tuple(n for n, t in safe_zip(self.axis_names, self.axis_types)
                 if t == AxisType.Manual)

  @functools.cached_property
  def _name_to_type(self):
    return dict(safe_zip(self.axis_names, self.axis_types))


def _unpicke_mesh(devices, axis_names, axis_types):
  return Mesh(devices, axis_names, axis_types)

_mesh_object_dict = {}  # type: ignore


class Mesh(BaseMesh, contextlib.ContextDecorator):
  """Declare the hardware resources available in the scope of this manager.

  See `Distributed arrays and automatic parallelization`_ and
  `Explicit Sharding`_ tutorials.

  Args:
    devices: A NumPy ndarray object containing JAX device objects (as
      obtained e.g. from :py:func:`jax.devices`).
    axis_names: A sequence of resource axis names to be assigned to the
      dimensions of the ``devices`` argument. Its length should match the
      rank of ``devices``.
    axis_types: and optional tuple of :class:`jax.sharding.AxisType` entries corresponding to
      the ``axis_names``. See `Explicit Sharding`_ for more information.

  Examples:

    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P, NamedSharding
    >>> import numpy as np
    ...
    >>> # Declare a 2D mesh with axes `x` and `y`.
    >>> devices = np.array(jax.devices()).reshape(4, 2)
    >>> mesh = Mesh(devices, ('x', 'y'))
    >>> inp = np.arange(16).reshape(8, 2)
    >>> arr = jax.device_put(inp, NamedSharding(mesh, P('x', 'y')))
    >>> out = jax.jit(lambda x: x * 2)(arr)
    >>> assert out.sharding == NamedSharding(mesh, P('x', 'y'))

  .. _Distributed arrays and automatic parallelization: https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
  .. _Explicit Sharding:  https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html
  """

  devices: np.ndarray
  axis_names: tuple[MeshAxisName, ...]

  def __new__(cls, devices: np.ndarray | Sequence[xc.Device],
              axis_names: str | Sequence[MeshAxisName],
              axis_types: tuple[AxisType, ...] | None = None):
    if not isinstance(devices, np.ndarray):
      devices = np.array(devices)
    if isinstance(axis_names, str):
      axis_names = (axis_names,)
    axis_names = tuple(axis_names)
    if any(i is None for i in axis_names):
      raise ValueError(f"Mesh axis names cannot be None. Got: {axis_names}")

    if devices.ndim != len(axis_names):
      raise ValueError(
          "Mesh requires the ndim of its first argument (`devices`) to equal "
          "the length of its second argument (`axis_names`), but got "
          f"devices.ndim == {devices.ndim} and "
          f"len(axis_names) == {len(axis_names)}.")

    axis_types = _normalize_axis_types(axis_names, axis_types, 'Mesh')

    key = (axis_names, devices.shape, tuple(devices.flat), axis_types)
    val = _mesh_object_dict.get(key, None)
    if val is not None:
      return val

    self = super().__new__(cls)
    self.devices = devices.copy()
    self.devices.flags.writeable = False
    self.axis_names = axis_names
    self.axis_types = axis_types
    self._size = math.prod(self.shape.values()) if self.devices.ndim else 0
    _mesh_object_dict[key] = self
    return self

  def __reduce__(self):
    return (_unpicke_mesh, (self.devices, self.axis_names, self.axis_types))

  def __eq__(self, other):
    # This is a performance optimization. Comparing thousands of devices
    # can be expensive.
    if self is other:
      return True
    if not isinstance(other, Mesh):
      return False
    return (self.axis_names == other.axis_names and
            self.devices.shape == other.devices.shape and
            self.axis_types == other.axis_types and
            self._internal_device_list == other._internal_device_list)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash(
          (self.axis_names, self._internal_device_list, self.devices.shape,
           self.axis_types))
    return self._hash

  def __setattr__(self, name, value):
    if hasattr(self, name):
      if getattr(self, name) == value:
        # This can to happen if two threads race, for example if two threads
        # are trying to hash the same Mesh instance.
        return
      raise RuntimeError(
          f"Cannot reassign attributes ({name}) of immutable mesh objects"
      )
    super().__setattr__(name, value)

  def __enter__(self):
    if jax_config.disallow_mesh_context_manager.value:
      raise RuntimeError("Mesh context manager is disabled.")
    new_env = thread_resources.stack[-1].with_mesh(self)
    thread_resources.stack.append(new_env)
    thread_resources.env = new_env
    jax_config.mesh_context_manager.set_local(
        tuple(t.physical_mesh for t in thread_resources.stack
              if not t.physical_mesh.empty))
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    thread_resources.stack.pop()
    thread_resources.env = thread_resources.stack[-1]
    jax_config.mesh_context_manager.set_local(
        tuple(t.physical_mesh for t in thread_resources.stack
              if not t.physical_mesh.empty))
    return False

  def update(self, devices=None, axis_names=None, axis_types=None):
    if devices is None:
      devices = self.devices
    if axis_names is None:
      axis_names = self.axis_names
    if axis_types is None:
      axis_types = self.axis_types
    return Mesh(devices, axis_names, axis_types)

  @functools.cached_property
  def shape(self):
    return collections.OrderedDict(
        (name, size)
        for name, size in safe_zip(self.axis_names, self.devices.shape))

  @functools.cached_property
  def shape_tuple(self):
    return tuple(
        (name, size)
        for name, size in safe_zip(self.axis_names, self.devices.shape))

  @property
  def axis_sizes(self) -> tuple[int, ...]:
    return self.devices.shape

  @property
  def size(self):
    return self._size

  @property
  def empty(self):
    return self.size == 0

  @functools.cached_property
  def is_multi_process(self):
    return self.devices.size != len(self.local_devices)

  @property
  def local_mesh(self):
    return self._local_mesh(xb.process_index())

  def _local_mesh(self, process_index):
    return _get_local_mesh(self, process_index)

  @functools.cached_property
  def device_ids(self):
    assert not self.empty
    return np.vectorize(lambda d: d.id, otypes=[int])(self.devices)

  @functools.cached_property
  def _local_devices_set(self):
    return set(self.local_devices)

  @functools.cached_property
  def _flat_devices_tuple(self):
    return tuple(self.devices.flat)

  @functools.cached_property
  def _internal_device_list(self):
    return xc.DeviceList(self._flat_devices_tuple)

  @functools.cached_property
  def _flat_devices_set(self):
    return set(self.devices.flat)

  def __str__(self):
    mesh_str = ", ".join(f"'{k}': {v}" for k, v in self.shape.items())
    atr = f", axis_types={self.axis_types}"
    return f"Mesh({mesh_str}{atr})"

  @functools.cached_property
  def _repr(self):
    if self.empty:
      return "Mesh(axis_sizes=(), axis_names=())"
    atr = f", axis_types={self.axis_types}"
    return (f"Mesh(axis_sizes={self.device_ids.shape}, "
            f"axis_names={self.axis_names!r}{atr})")

  def __repr__(self):
    return self._repr

  @functools.cached_property
  def local_devices(self):
    return [d for d in self.devices.flat
            if d.process_index == d.client.process_index()]

  @functools.cached_property
  def abstract_mesh(self):
    d = self.devices.flat[0]
    if d is None:
      abstract_device = None
    else:
      num_tpu_cores = getattr(d, 'num_cores', 0) if d.platform == 'tpu' else 0
      abstract_device = AbstractDevice(
          device_kind=d.device_kind, num_tpu_cores=num_tpu_cores)
    return AbstractMesh(
        self.axis_sizes, self.axis_names, axis_types=self.axis_types,
        abstract_device=abstract_device)


EMPTY_ENV = ResourceEnv(Mesh(np.empty((), dtype=object), ()))

class _ThreadResourcesLocalState(threading.local):

  def __init__(self):
    self.stack = [EMPTY_ENV]
    self.env = self.stack[-1]

thread_resources = _ThreadResourcesLocalState()


@dataclasses.dataclass(frozen=True)
class AbstractDevice:
  device_kind: str
  num_tpu_cores: int

  def __repr__(self):
    return (f"AbstractDevice({self._repr()})")

  def _repr(self):
    return f"device_kind={self.device_kind}, num_tpu_cores={self.num_tpu_cores}"


class AbstractMesh(BaseMesh):
  """AbstractMesh contains only axis names and axis sizes.

  It does not contain concrete devices compared to `jax.sharding.Mesh`. You
  should use this as an input to the sharding passed to with_sharding_constraint
  and mesh passed to shard_map to avoid tracing and lowering cache misses when
  your mesh shape and axis names stay the same but the devices change.
  See the description of https://github.com/jax-ml/jax/pull/23022 for more
  details.

  Args:
    axis_sizes: A tuple of integers specifying the size of each resource axis.
    axis_names: A tuple of resource axis names to be assigned to the
      dimensions of the ``devices`` argument. Its length should match the
      rank of ``devices``.
    axis_types: and optional tuple of :class:`jax.sharding.AxisType` entries corresponding to
      the ``axis_names``. See `Explicit Sharding`_ for more information.

  .. _Explicit Sharding:  https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html
  """

  def __init__(self, axis_sizes: tuple[int, ...], axis_names: tuple[str, ...],
               axis_types: AxisType | tuple[AxisType, ...] | None = None,
               *, abstract_device=None):
    self.axis_sizes = axis_sizes
    self.axis_names = axis_names
    self.axis_types = _normalize_axis_types(
        self.axis_names, axis_types, 'AbstractMesh')
    self.abstract_device = abstract_device
    self.size = math.prod(self.axis_sizes) if self.axis_sizes else 0
    self._hash = hash((self.axis_sizes, self.axis_names, self.axis_types,
                       self.abstract_device))

  def __hash__(self):
    return self._hash

  def __eq__(self, other):
    if self is other:
      return True
    if not isinstance(other, AbstractMesh):
      return False
    return (self.axis_sizes == other.axis_sizes and
            self.axis_names == other.axis_names and
            self.axis_types == other.axis_types and
            self.abstract_device == other.abstract_device)

  def __repr__(self):
    mesh_repr = (", ".join(f"'{n}': {v}" for n, v in self.shape_tuple)
                 if self.shape_tuple else "()")
    atr = f", axis_types={self.axis_types}"
    ad = ("" if self.abstract_device is None else
          f", {self.abstract_device._repr()}")
    return f"AbstractMesh({mesh_repr}{atr}{ad})"

  def update(self, axis_sizes=None, axis_names=None, axis_types=None, **kwargs):
    if axis_sizes is None:
      axis_sizes = self.axis_sizes
    if axis_names is None:
      axis_names = self.axis_names
    if axis_types is None:
      axis_types = self.axis_types
    if 'abstract_device' not in kwargs:
      kwargs['abstract_device'] = self.abstract_device
    return AbstractMesh(axis_sizes, axis_names, axis_types, **kwargs)

  @functools.cached_property
  def shape(self):
    return collections.OrderedDict(self.shape_tuple)

  @functools.cached_property
  def shape_tuple(self):
    return tuple(
        (name, size)
        for name, size in safe_zip(self.axis_names, self.axis_sizes))

  @property
  def _internal_device_list(self):
    return None

  @property
  def empty(self):
    return self.size == 0

  @property
  def abstract_mesh(self):
    return self

  def update_axis_types(self, name_to_type: dict[MeshAxisName, AxisType]):
    new_axis_types = tuple(name_to_type[n] if n in name_to_type else a
                           for n, a in zip(self.axis_names, self.axis_types))
    return self.update(axis_types=new_axis_types)

  @property
  def devices(self):
    _raise_value_error("devices")

  @property
  def device_ids(self):
    _raise_value_error("device_ids")

  @property
  def is_multi_process(self):
    _raise_value_error("is_multi_process")

  @property
  def local_devices(self):
    _raise_value_error("local_devices")

  @property
  def local_mesh(self):
    _raise_value_error("local_mesh")

  def __enter__(self):
    _raise_value_error("__enter__")

  def __exit__(self, exc_type, exc_value, traceback):
    _raise_value_error("__exit__")

  @staticmethod
  def _extremely_unsafe_enter_tracing_context(mesh: AbstractMesh):
    prev = jax_config.abstract_mesh_context_manager.swap_local(mesh)
    return prev


# Create this indirection because pytype fails to recognize a property if a
# property raises an exception unconditionally. Remove this once that is fixed.
def _raise_value_error(name):
  raise ValueError(f"AbstractMesh does not implement {name}")

empty_abstract_mesh = AbstractMesh((), ())
empty_concrete_mesh = Mesh(np.empty((), dtype=object), ())

class use_abstract_mesh:
  __slots__ = ['mesh', 'prev']

  def __init__(self, mesh: AbstractMesh):
    if not isinstance(mesh, AbstractMesh):
      raise ValueError(
          "Expected mesh of type `jax.sharding.AbstractMesh`. Got type:"
          f" {type(mesh)}")
    self.mesh = mesh

  def __enter__(self):
    self.prev = jax_config.abstract_mesh_context_manager.swap_local(self.mesh)

  def __exit__(self, exc_type, exc_value, traceback):
    jax_config.abstract_mesh_context_manager.set_local(self.prev)


def get_abstract_mesh() -> AbstractMesh:
  val = jax_config.abstract_mesh_context_manager.value
  return empty_abstract_mesh if val is None else val

def get_concrete_mesh() -> Mesh:
  val = jax_config.device_context.value
  return empty_concrete_mesh if val is None else val
