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

from jax._src.util import weak_value_interner, immutable
from jax._src.lib import _jax

AxisName = Any

def _check(partitions, unreduced, reduced):
  if not reduced and not unreduced:
    return
  if None in unreduced:
    raise ValueError(
        "unreduced cannot contain None. All elements in unreduced should refer"
        " to the mesh axes.")
  if None in reduced:
    raise ValueError(
        "reduced cannot contain None. All elements in reduced should refer"
        " to the mesh axes.")
  if unreduced & reduced:
    raise ValueError(
        "`unreduced` and `reduced` argument to PartitionSpec cannot overlap. "
        f"Got unreduced: {unreduced} and reduced: {reduced}")

  for partition in partitions:
    partition = partition if isinstance(partition, tuple) else (partition,)
    for p in partition:
      if p in unreduced:
        raise ValueError(
            "partitions cannot overlap with unreduced axes passed to"
            f" PartitionSpec. Got partitions: {partitions} and unreduced axes:"
            f" {unreduced}")
      if p in reduced:
        raise ValueError(
            "partitions cannot overlap with reduced axes passed to"
            f" PartitionSpec. Got partitions: {partitions} and reduced axes:"
            f" {reduced}")

def _get_ur_str(unreduced, reduced):
  if unreduced and reduced:
    return f"unreduced={set(unreduced)!r}, reduced={set(reduced)!r}"
  elif unreduced and not reduced:
    return f"unreduced={set(unreduced)!r}"
  elif not unreduced and reduced:
    return f"reduced={set(reduced)!r}"
  assert False  # unreachable

_canonicalize_partition = _jax.canonicalize_partition  # type: ignore
_canonicalize_partitions = _jax.canonicalize_partitions  # type: ignore

def _get_default_unconstrained(): return _UNCONSTRAINED_PARTITION

class UnconstrainedSingleton:
  def __repr__(self): return "UNCONSTRAINED"
  def __reduce__(self): return (_get_default_unconstrained, ())

_UNCONSTRAINED_PARTITION = UnconstrainedSingleton()
_jax.set_pspec_unconstrained(_UNCONSTRAINED_PARTITION)  # type: ignore

@immutable
class P:
  """Tuple describing how to partition an array across a mesh of devices.

  Each element is either ``None``, a string, or a tuple of strings.
  See the documentation of :class:`jax.sharding.NamedSharding` for more details.

  This class exists so JAX's pytree utilities can distinguish a partition
  specifications from tuples that should be treated as pytrees.
  """
  __slots__ = ("_partitions", "unreduced", "reduced", "__weakref__")
  _partitions: tuple[AxisName]
  unreduced: frozenset[AxisName]
  reduced: frozenset[AxisName]

  # A sentinel value representing a dim is unconstrained.
  UNCONSTRAINED = _UNCONSTRAINED_PARTITION

  @staticmethod
  @weak_value_interner
  def _create(partitions, unreduced, reduced):
    # We cannot modify the arguments within the interned function, but we are
    # free to throw an exception.
    _check(partitions, unreduced, reduced)
    obj = object.__new__(P)
    object.__setattr__(obj, '_partitions', partitions)
    object.__setattr__(obj, 'unreduced', unreduced)
    object.__setattr__(obj, 'reduced', reduced)
    return obj

  def __new__(cls, *partitions, unreduced=frozenset(), reduced=frozenset()):
    partitions = _canonicalize_partitions(partitions)
    if not isinstance(unreduced, frozenset):
      if not isinstance(unreduced, set):
        raise TypeError(
            f"unreduced argument of PartitionSpec should "
            f"of type `frozenset` or `set`. Got type {unreduced}")
      unreduced = frozenset(unreduced)
    if not isinstance(reduced, frozenset):
      if not isinstance(reduced, set):
        raise TypeError(
            f"reduced argument of PartitionSpec should "
            f"of type `frozenset` or `set`. Got type {reduced}")
      reduced = frozenset(reduced)
    return P._create(partitions, unreduced, reduced)  # type: ignore

  # No __eq__ or __hash__: interned classes use object identity.

  def __init_subclass__(cls, *args, **kwargs):
    raise TypeError("Subclassing `jax.P` is prohibited.")

  @property
  def partitions(self):
    return self._partitions

  def __repr__(self):
    pr = repr(self._partitions)[1:-1]
    if not self.unreduced and not self.reduced:
      return f"P({pr})"
    ur_str = _get_ur_str(self.unreduced, self.reduced)
    pr = '' if not pr else f"{pr} " if pr.endswith(',') else f"{pr}, "
    return (f"P({pr}{ur_str})")

  def __getnewargs_ex__(self):
    return (self._partitions,
            {'unreduced': self.unreduced, 'reduced': self.reduced})

  def __getitem__(self, i):
    return self._partitions[i]

  def __iter__(self):
    return iter(self._partitions)

  def __len__(self):
    return len(self._partitions)

  def __add__(self, other):
    if isinstance(other, P):
      return P(*self, *other, unreduced={*self.unreduced, *other.unreduced},
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
      return P(*self, *other)
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
    return P(*other, *self)

  def index(self, value):
    return self._partitions.index(_canonicalize_partition(value))

  def count(self, value):
    return self._partitions.count(_canonicalize_partition(value))

  def update(self, partitions=None, unreduced=None, reduced=None):
    p = self._partitions if partitions is None else partitions
    ur = self.unreduced if unreduced is None else unreduced
    r = self.reduced if reduced is None else reduced
    return P(*p, unreduced=ur, reduced=r)

  def to_lo(self):
    return [self]

  def to_tangent_spec(self):
    return self

  def to_ct_spec(self):
    return self.update(unreduced=self.reduced, reduced=self.unreduced)

  def _normalized_spec_for_aval(self, ndim: int) -> P:
    out = [None if p is _UNCONSTRAINED_PARTITION else p
          for p in self._partitions]
    if len(out) < ndim:
      out.extend([None] * (ndim - len(out)))
    return self.update(partitions=out)

  def _check_compatible_wrt_shape(self, shape):
    if len(shape) < len(self._partitions):
      extra_msg = (' For scalars the PartitionSpec should be P()'
                  if len(shape) == 0 else '')
      raise ValueError(
          f"PartitionSpec {self} is only valid for values of rank at least "
          f"{len(self._partitions)}, but was applied to a value of rank "
          f"{len(shape)}.{extra_msg}")

P.__module__ = 'jax'
PartitionSpec = P
