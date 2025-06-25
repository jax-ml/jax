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

from jax._src.lib import _jax
from jax._src.util import use_cpp_class, use_cpp_method

_UNCONSTRAINED_PARTITION = _jax.UNCONSTRAINED_PARTITION
_canonicalize_partition = _jax.canonicalize_partition


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

@use_cpp_class(_jax.PartitionSpec)
class PartitionSpec:
  """Tuple describing how to partition an array across a mesh of devices.

  Each element is either ``None``, a string, or a tuple of strings.
  See the documentation of :class:`jax.sharding.NamedSharding` for more details.

  This class exists so JAX's pytree utilities can distinguish a partition
  specifications from tuples that should be treated as pytrees.
  """
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
    # See the description of https://github.com/jax-ml/jax/pull/29381
    self.reduced = frozenset(reduced)
    # `__init__` is implemented in C++ so this check happens in C++
    # _check(self._partitions, self.unreduced, self.reduced)

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

  def update(self, **kwargs):
    return PartitionSpec(*kwargs.pop("partitions", self._partitions),
                         unreduced=kwargs.pop("unreduced", self.unreduced),
                         reduced=kwargs.pop("reduced", self.reduced))

  def _normalized_spec_for_aval(self, ndim: int) -> PartitionSpec:
    out = [None if p is _UNCONSTRAINED_PARTITION else p
           for p in self._partitions]
    if len(out) < ndim:
      out.extend([None] * (ndim - len(out)))
    return self.update(partitions=out)

PartitionSpec.__module__ = 'jax.sharding'

P = PartitionSpec
