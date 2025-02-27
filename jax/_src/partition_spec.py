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


class PartitionSpec(tuple):
  """Tuple describing how to partition an array across a mesh of devices.

  Each element is either ``None``, a string, or a tuple of strings.
  See the documentation of :class:`jax.sharding.NamedSharding` for more details.

  This class exists so JAX's pytree utilities can distinguish a partition
  specifications from tuples that should be treated as pytrees.
  """

  # A sentinel value representing a dim is unconstrained.
  UNCONSTRAINED = _UNCONSTRAINED_PARTITION

  def __init__(self, *partitions):
    pass

  def __new__(cls, *partitions):
    partitions = tuple(_canonicalize_partition(p) for p in partitions)
    return tuple.__new__(PartitionSpec, partitions)

  def __repr__(self):
    return f"PartitionSpec{tuple.__repr__(self)}"

  def __reduce__(self):
    return (PartitionSpec, tuple(self))

  def __eq__(self, other):
    if not isinstance(other, tuple):
      return False
    other = tuple(_canonicalize_partition(o) for o in other)
    return super().__eq__(other)

  def __hash__(self):
    return super().__hash__()

  def index(self, value):
    value = _canonicalize_partition(value)
    return super().index(value)

  def _normalized_spec_for_aval(self, ndim: int) -> PartitionSpec:
    out = [None if p is _UNCONSTRAINED_PARTITION else p for p in self]
    if len(out) < ndim:
      out.extend([None] * (ndim - len(out)))
    return PartitionSpec(*out)
