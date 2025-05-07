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
from typing import TYPE_CHECKING

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


class PartitionSpecImpl:
  """Tuple describing how to partition an array across a mesh of devices.

  Each element is either ``None``, a string, or a tuple of strings.
  See the documentation of :class:`jax.sharding.NamedSharding` for more details.

  This class exists so JAX's pytree utilities can distinguish a partition
  specifications from tuples that should be treated as pytrees.
  """
  __slots__ = ("_partitions",)
  __match_args__ = ("_partitions",)

  # A sentinel value representing a dim is unconstrained.
  UNCONSTRAINED = _UNCONSTRAINED_PARTITION

  def __init__(self, *partitions):
    self._partitions = tuple(_canonicalize_partition(p) for p in partitions)

  def __repr__(self):
    return f"PartitionSpec{self._partitions!r}"

  def __reduce__(self):
    return (PartitionSpec, self._partitions)

  def __getitem__(self, i):
    return self._partitions[i]

  def __iter__(self):
    return iter(self._partitions)

  def __len__(self):
    return len(self._partitions)

  def __eq__(self, other):
    if not isinstance(other, (PartitionSpec, tuple)):
      return False
    other = tuple(_canonicalize_partition(o) for o in other)
    return self._partitions == other

  def __hash__(self):
    return hash(self._partitions)

  def __add__(self, other):
    if not isinstance(other, (tuple, PartitionSpec)):
      return NotImplementedError
    return PartitionSpec(*self, *other)

  def __radd__(self, other):
    if not isinstance(other, (tuple, PartitionSpec)):
      return NotImplementedError
    return PartitionSpec(*other, *self)

  def index(self, value):
    return self._partitions.index(_canonicalize_partition(value))

  def count(self, value):
    return self._partitions.count(_canonicalize_partition(value))

  def _normalized_spec_for_aval(self, ndim: int) -> PartitionSpec:
    out = [None if p is _UNCONSTRAINED_PARTITION else p
           for p in self._partitions]
    if len(out) < ndim:
      out.extend([None] * (ndim - len(out)))
    return PartitionSpec(*out)

if TYPE_CHECKING:
  class PartitionSpec(PartitionSpecImpl, tuple):  # type: ignore
    ...
else:
  PartitionSpec = PartitionSpecImpl
