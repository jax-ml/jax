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
from typing import Any, TYPE_CHECKING

from jax._src.lib import jaxlib_extension_version
from jax._src.lib import _jax
from jax._src.util import use_cpp_class, use_cpp_method, set_module

export = set_module('jax.sharding')

# TODO(phawkins): the union confuses pytype. Just use the Python branch for now
# until the C++ version is the minimum version.
if not TYPE_CHECKING and jaxlib_extension_version >= 352:
  _UNCONSTRAINED_PARTITION = _jax.UNCONSTRAINED_PARTITION
  _canonicalize_partition = _jax.canonicalize_partition
else:
  class UnconstrainedSingleton:

    def __repr__(self):
      return "UNCONSTRAINED"

    def __reduce__(self):
      return (_get_default_unconstrained, ())


  # Unconstrained sentinel value for PartitionSpec, representing a dimension for
  # which the user wants XLA to assign the best partitioning.
  # TODO(yashkatariya): May rename to AUTO.
  _UNCONSTRAINED_PARTITION = UnconstrainedSingleton()

  def _get_default_unconstrained():
    return _UNCONSTRAINED_PARTITION

  def _canonicalize_partition(partition):
    if not partition:
      return None
    if partition is _UNCONSTRAINED_PARTITION:
      return _UNCONSTRAINED_PARTITION
    if isinstance(partition, (tuple, list)):
      if len(partition) == 1:
        return partition[0]
      return tuple(partition)
    return partition

  def _check(partitions, unreduced, reduced):
    for p in partitions:
      p = p if isinstance(p, tuple) else (p,)
      for r in p:
        if r in unreduced:
          raise ValueError(
              "partitions cannot overlap with unreduced axes passed to"
              f" PartitionSpec. Got partitions: {partitions} and unreduced axes:"
              f" {unreduced}")
        if r in reduced:
          raise ValueError(
              "partitions cannot overlap with reduced axes passed to"
              f" PartitionSpec. Got partitions: {partitions} and reduced axes:"
              f" {reduced}")
    if unreduced & reduced:
      raise ValueError(
          "`unreduced` and `reduced` argument to PartitionSpec cannot overlap. "
          f"Got {unreduced=}, {reduced=}")
    if None in unreduced:
      raise ValueError(
          "unreduced cannot contain None. All elements in unreduced should refer"
          " to the mesh axes.")
    if None in reduced:
      raise ValueError(
          "reduced cannot contain None. All elements in reduced should refer"
          " to the mesh axes.")

def unpickle_pspec(partitions, unreduced, reduced):
  return PartitionSpec(*partitions, unreduced=unreduced, reduced=reduced)

def _get_ur_str(unreduced, reduced):
  if unreduced and reduced:
    return f"unreduced={set(unreduced)!r}, reduced={set(reduced)!r}"
  elif unreduced and not reduced:
    return f"unreduced={set(unreduced)!r}"
  elif not unreduced and reduced:
    return f"reduced={set(reduced)!r}"
  assert False  # unreachable

AxisName = Any

class PartitionSpec:
  """Tuple describing how to partition an array across a mesh of devices.

  Each element is either ``None``, a string, or a tuple of strings.
  See the documentation of :class:`jax.sharding.NamedSharding` for more details.

  This class exists so JAX's pytree utilities can distinguish a partition
  specifications from tuples that should be treated as pytrees.
  """
  if jaxlib_extension_version < 352:
    __slots__ = ("_partitions", "unreduced", "reduced")
  __match_args__ = ("_partitions",)

  # A sentinel value representing a dim is unconstrained.
  UNCONSTRAINED = _UNCONSTRAINED_PARTITION

  @use_cpp_method()
  def __init__(self, *partitions, unreduced=frozenset(), reduced=frozenset()):
    self._partitions = tuple(_canonicalize_partition(p) for p in partitions)
    if not isinstance(unreduced, (set, frozenset)):
      raise TypeError(
          "`unreduced` argument of PartitionSpec should be of type"
          f" `frozenset` or `set`. Got type {type(unreduced)}")
    if not isinstance(reduced, (set, frozenset)):
      raise TypeError(
          "`reduced` argument of PartitionSpec should be of type"
          f" `frozenset` or `set`. Got type {type(reduced)}")
    self.unreduced = frozenset(unreduced)
    self.reduced = frozenset(reduced)
    _check(self._partitions, self.unreduced, self.reduced)

  def __repr__(self):
    pr = repr(self._partitions)[1:-1]
    if not self.unreduced and not self.reduced:
      return f"PartitionSpec({pr})"
    ur_str = _get_ur_str(self.unreduced, self.reduced)
    pr = '' if not pr else f"{pr} " if pr.endswith(',') else f"{pr}, "
    return (f"PartitionSpec({pr}{ur_str})")

  def __reduce__(self):
    return (unpickle_pspec, (self._partitions, self.unreduced, self.reduced))

  def __getitem__(self, i):
    return self._partitions[i]

  def __iter__(self):
    return iter(self._partitions)

  def __len__(self):
    return len(self._partitions)

  @use_cpp_method()
  def __eq__(self, other):
    if isinstance(other, PartitionSpec):
      return (self._partitions == other._partitions and
              self.unreduced == other.unreduced and
              self.reduced == other.reduced)
    elif isinstance(other, tuple):
      if self.unreduced:
        raise TypeError(
            f"other {other} cannot be of instance `tuple` when self {self} has"
            " unreduced in `__eq__` of PartitionSpec.")
      if self.reduced:
        raise TypeError(
            f"other {other} cannot be of instance `tuple` when self {self} has"
            " reduced in `__eq__` of PartitionSpec.")
      other_p = tuple(_canonicalize_partition(o) for o in other)
      return self._partitions == other_p
    else:
      return False

  @use_cpp_method()
  def __hash__(self):
    return hash((self._partitions, self.unreduced, self.reduced))

  def __add__(self, other):
    if isinstance(other, PartitionSpec):
      return PartitionSpec(
          *self, *other,
          unreduced={*self.unreduced, *other.unreduced},
          reduced={*self.reduced, *other.reduced})
    elif isinstance(other, tuple):
      if self.unreduced:
        raise TypeError(
            f"other {other} cannot be of instance `tuple` when self {self} has"
            " unreduced in `__add__` of PartitionSpec.")
      if self.reduced:
        raise TypeError(
            f"other {other} cannot be of instance `tuple` when self {self} has"
            " reduced in `__add__` of PartitionSpec.")
      return PartitionSpec(*self, *other)
    else:
      raise NotImplementedError

  def __radd__(self, other):
    if not isinstance(other, tuple):
      raise NotImplementedError
    # other will always be a tuple.
    if self.unreduced:
      raise TypeError(
          f"other {other} cannot be of instance `tuple` when self {self} has"
          " unreduced in `__radd__` of PartitionSpec.")
    if self.reduced:
      raise TypeError(
          f"other {other} cannot be of instance `tuple` when self {self} has"
          " reduced in `__radd__` of PartitionSpec.")
    return PartitionSpec(*other, *self)

  def index(self, value):
    return self._partitions.index(_canonicalize_partition(value))

  def count(self, value):
    return self._partitions.count(_canonicalize_partition(value))

  def with_partitions(self, new_partitions):
    return PartitionSpec(*new_partitions, unreduced=self.unreduced,
                         reduced=self.reduced)

  def with_unreduced(self, new_unreduced):
    return PartitionSpec(*self._partitions, unreduced=new_unreduced,
                         reduced=self.reduced)

  def _normalized_spec_for_aval(self, ndim: int) -> PartitionSpec:
    out = [None if p is _UNCONSTRAINED_PARTITION else p
           for p in self._partitions]
    if len(out) < ndim:
      out.extend([None] * (ndim - len(out)))
    return self.with_partitions(out)

# TODO(phawkins): make this a decorator after the next jaxlib release.
if not TYPE_CHECKING and jaxlib_extension_version >= 352:
  PartitionSpec = use_cpp_class(_jax.PartitionSpec)(PartitionSpec)

# TODO(phawkins): make this a decorator after the next jaxlib release.
if not TYPE_CHECKING:
  PartitionSpec = export(PartitionSpec)
