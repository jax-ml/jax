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
from typing import Any

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

def _check(partitions, unreduced):
  us = set(unreduced)
  for p in partitions:
    p = p if isinstance(p, tuple) else (p,)
    for r in p:
      if r in us:
        raise ValueError(
            "partitions cannot overlap with unreduced axes passed to"
            f" PartitionSpec. Got partitions: {partitions} and unreduced axes:"
            f" {unreduced}")
  if None in unreduced:
    raise ValueError(
        "unreduced cannot contain None. All elements in unreduced should refer"
        " to the mesh axes.")

def unpicke_pspec(partitions, unreduced):
  return PartitionSpec(*partitions, unreduced=unreduced)

AxisName = Any

class PartitionSpec:
  """Tuple describing how to partition an array across a mesh of devices.

  Each element is either ``None``, a string, or a tuple of strings.
  See the documentation of :class:`jax.sharding.NamedSharding` for more details.

  This class exists so JAX's pytree utilities can distinguish a partition
  specifications from tuples that should be treated as pytrees.
  """
  __slots__ = ("_partitions", "_unreduced")
  __match_args__ = ("_partitions",)

  # A sentinel value representing a dim is unconstrained.
  UNCONSTRAINED = _UNCONSTRAINED_PARTITION

  def __init__(self, *partitions,
               unreduced: tuple[AxisName, ...] | AxisName | None = None):
    self._partitions = tuple(_canonicalize_partition(p) for p in partitions)
    self._unreduced = (
        () if unreduced is None else tuple(unreduced)
        if isinstance(unreduced, (list, tuple)) else (unreduced,))
    _check(self._partitions, self._unreduced)

  @property
  def unreduced(self):
    return self._unreduced

  def __repr__(self):
    pr = repr(self._partitions)[1:-1]
    if not self._unreduced:
      return f"PartitionSpec({pr})"
    ur_str = f"unreduced={self._unreduced!r}"
    pr = '' if not pr else f"{pr} " if pr.endswith(',') else f"{pr}, "
    return (f"PartitionSpec({pr}{ur_str})")

  def __reduce__(self):
    return (unpicke_pspec, (self._partitions, self._unreduced))

  def __getitem__(self, i):
    return self._partitions[i]

  def __iter__(self):
    return iter(self._partitions)

  def __len__(self):
    return len(self._partitions)

  def __eq__(self, other):
    if not isinstance(other, (PartitionSpec, tuple)):
      return False
    other_p = tuple(_canonicalize_partition(o) for o in other)
    if isinstance(other, PartitionSpec):
      return (self._partitions == other_p and
              self._unreduced == other._unreduced)
    else:
      if self._unreduced:
        raise TypeError(
            f"other {other} cannot be of instance `tuple` when self {self} has"
            " unreduced in `__eq__` of PartitionSpec.")
      return self._partitions == other_p

  def __hash__(self):
    return hash((self._partitions, self._unreduced))

  def __add__(self, other):
    if not isinstance(other, (tuple, PartitionSpec)):
      raise NotImplementedError
    if isinstance(other, PartitionSpec):
      return PartitionSpec(
          *self, *other,
          unreduced=(*self._unreduced, *other._unreduced))
    else:
      if self._unreduced:
        raise TypeError(
            f"other {other} cannot be of instance `tuple` when self {self} has"
            " unreduced in `__add__` of PartitionSpec.")
      return PartitionSpec(*self, *other)

  def __radd__(self, other):
    if not isinstance(other, tuple):
      raise NotImplementedError
    # other will always be a tuple.
    if self._unreduced:
      raise TypeError(
          f"other {other} cannot be of instance `tuple` when self {self} has"
          " unreduced in `__radd__` of PartitionSpec.")
    return PartitionSpec(*other, *self)

  def index(self, value):
    return self._partitions.index(_canonicalize_partition(value))

  def count(self, value):
    return self._partitions.count(_canonicalize_partition(value))

  def with_partitions(self, new_partitions):
    return PartitionSpec(*new_partitions, unreduced=self._unreduced)

  def with_unreduced(self, new_unreduced):
    return PartitionSpec(*self._partitions, unreduced=new_unreduced)

  def _normalized_spec_for_aval(self, ndim: int) -> PartitionSpec:
    out = [None if p is _UNCONSTRAINED_PARTITION else p
           for p in self._partitions]
    if len(out) < ndim:
      out.extend([None] * (ndim - len(out)))
    return self.with_partitions(out)
