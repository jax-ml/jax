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

class _UnconstrainedPartitionSingleton:

  def __str__(self):
    return "UNCONSTRAINED"


# Unconstrained sentinel value for PartitionSpec, representing a dimension for
# which the user wants XLA to assign the best partitioning.
# TODO(yashkatariya): May rename to AUTO.
_UNCONSTRAINED_PARTITION = _UnconstrainedPartitionSingleton()


class PartitionSpec(tuple):
  """Tuple describing how to partition tensor into mesh .

  Each element is either None, string or a tuple of strings.
  See``NamedSharding`` class for more details.

  We create a separate class for this so JAX's pytree utilities can distinguish
  it from a tuple that should be treated as a pytree.
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
