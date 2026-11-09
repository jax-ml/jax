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
import enum
from typing import Any

from jax._src.util import weak_value_interner, immutable
from jax._src.lib import _jax

AxisName = Any

def _check(partitions, unreduced, reduced, unreduced_kind):
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
  if unreduced_kind is not None and not isinstance(unreduced_kind, UnreducedKind):
      raise TypeError(
          "Expected unreduced_kind to be of type `jax.sharding.UnreducedKind`"
          f" but got {type(unreduced_kind)}")
  if not unreduced and unreduced_kind is not None:
    raise ValueError(
        "`unreduced_kind` should be `None` when `unreduced` is an empty set."
        f" Got {unreduced_kind=} and {unreduced=}")

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


def _canonicalize_ur(name, val):
  if not isinstance(val, frozenset):
    if not isinstance(val, set):
      raise TypeError(
          f"{name} argument of PartitionSpec should "
          f"of type `frozenset` or `set`. Got type {type(val)}")
    val = frozenset(val)
  return val


class UnreducedKind(enum.Enum):
  sum = enum.auto()
  max = enum.auto()
  min = enum.auto()


@immutable
class P:
  """Tuple describing how to partition an array across a mesh of devices.

  Each element is either ``None``, a string, or a tuple of strings.
  See the documentation of :class:`jax.sharding.NamedSharding` for more details.

  This class exists so JAX's pytree utilities can distinguish a partition
  specifications from tuples that should be treated as pytrees.
  """
  __slots__ = ("_partitions", "unreduced", "reduced", "unreduced_kind",
               "__weakref__")
  _partitions: tuple[AxisName]
  unreduced: frozenset[AxisName]
  reduced: frozenset[AxisName]
  unreduced_kind: UnreducedKind | None

  # A sentinel value representing a dim is unconstrained.
  UNCONSTRAINED = _UNCONSTRAINED_PARTITION

  @staticmethod
  @weak_value_interner
  def _create(partitions, unreduced, reduced, unreduced_kind):
    # We cannot modify the arguments within the interned function, but we are
    # free to throw an exception.
    _check(partitions, unreduced, reduced, unreduced_kind)
    obj = object.__new__(P)
    object.__setattr__(obj, '_partitions', partitions)
    object.__setattr__(obj, 'unreduced', unreduced)
    object.__setattr__(obj, 'reduced', reduced)
    object.__setattr__(obj, 'unreduced_kind', unreduced_kind)
    return obj

  def __new__(cls, *partitions, unreduced=frozenset(), reduced=frozenset(),
              unreduced_kind=None):
    partitions = _canonicalize_partitions(partitions)
    unreduced = _canonicalize_ur('unreduced', unreduced)
    reduced = _canonicalize_ur('reduced', reduced)
    if unreduced and unreduced_kind is None:
      unreduced_kind = UnreducedKind.sum
    return P._create(partitions, unreduced, reduced, unreduced_kind)  # type: ignore

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
    uk = (f", unreduced_kind={self.unreduced_kind.name}" if self.unreduced_kind
          else "")
    return f"P({pr}{ur_str}{uk})"

  def __getnewargs_ex__(self):
    return (self._partitions,
            {'unreduced': self.unreduced, 'reduced': self.reduced,
             'unreduced_kind': self.unreduced_kind})

  def __getitem__(self, i):
    if self.reduced or self.unreduced:
      raise ValueError(
          "Using pspec[...] is dangerous when PartitionSpec has non-empty"
          " unreduced/reduced set. Please use spec.partitions[...]")
    return self._partitions[i]

  def __iter__(self):
    if self.reduced or self.unreduced:
      raise ValueError(
          "Using *pspec is dangerous when PartitionSpec has non-empty"
          " unreduced/reduced set. Please use *spec.partitions")
    return iter(self._partitions)

  def __len__(self):
    return len(self._partitions)

  def __add__(self, other):
    if isinstance(other, P):
      if self.unreduced_kind != other.unreduced_kind:
        raise TypeError(
            "PartitionSpec can't be added if the unreduced_kind differs in self"
            f" and other. Got {self=} and {other=}")
      return P(*self.partitions, *other.partitions,
               unreduced={*self.unreduced, *other.unreduced},
               reduced={*self.reduced, *other.reduced},
               unreduced_kind=self.unreduced_kind)
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

  def update(self, partitions=None, unreduced=None, reduced=None, **kwargs):
    p = self._partitions if partitions is None else partitions
    ur = self.unreduced if unreduced is None else unreduced
    r = self.reduced if reduced is None else reduced
    if 'unreduced_kind' not in kwargs:
      kwargs['unreduced_kind'] = self.unreduced_kind
    return P(*p, unreduced=ur, reduced=r, **kwargs)

  def to_lo(self):
    return [self]

  def to_tangent_spec(self):
    return self

  def to_ct_spec(self):
    assert self.unreduced_kind is None or self.unreduced_kind is UnreducedKind.sum
    kind = UnreducedKind.sum if self.reduced else None
    return self.update(unreduced=self.reduced, reduced=self.unreduced,
                       unreduced_kind=kind)

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
