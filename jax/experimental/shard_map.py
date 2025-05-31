# Copyright 2023 The JAX Authors.
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

from collections.abc import Callable, Hashable
from typing import Any
from jax._src import traceback_util
from jax.sharding import Mesh, AbstractMesh
from jax._src import shard_map as jshmap

Specs = Any
AxisName = Hashable

@traceback_util.api_boundary
def shard_map(
    f: Callable, mesh: Mesh | AbstractMesh, in_specs: Specs, out_specs: Specs,
    check_rep: bool = True, auto: frozenset[AxisName] = frozenset()):
  """Map a function over shards of data.

  Note:
    ``shard_map`` is an experimental API, and still subject to change. For an
    introduction to sharded data, refer to :ref:`sharded-computation`. For a more
    in-depth look at using ``shard_map``, refer to `SPMD multi-device parallelism with shard_map`_.

  Args:
    f: callable to be mapped. Each application of ``f``, or "instance" of ``f``,
      takes as input a shard of the mapped-over arguments and produces a shard
      of the output.
    mesh: a ``jax.sharding.Mesh`` representing the array of devices over which
      to shard the data and on which to execute instances of ``f``. The names of
      the ``Mesh`` can be used in collective communication operations in ``f``.
      This is typically created by a utility function like
      :func:`jax.experimental.mesh_utils.create_device_mesh`.
    in_specs: a pytree with :class:`~jax.sharding.PartitionSpec` instances as leaves,
      with a tree structure that is a tree prefix of the args tuple to be mapped
      over. Similar to :class:`~jax.sharding.NamedSharding`, each ``PartitionSpec``
      represents how the corresponding argument (or subtree of arguments) should
      be sharded along the named axes of ``mesh``. In each ``PartitionSpec``,
      mentioning a ``mesh`` axis name at a position expresses sharding the
      corresponding argument array axis along that positional axis; not
      mentioning an axis name expresses replication. If an argument, or argument
      subtree, has a corresponding spec of None, that argument is not sharded.
    out_specs: a pytree with :class:`~jax.sharding.PartitionSpec` instances as leaves,
      with a tree structure that is a tree prefix of the output of ``f``. Each
      ``PartitionSpec`` represents how the corresponding output shards should be
      concatenated. In each ``PartitionSpec``, mentioning a ``mesh`` axis name at
      a position expresses concatenation of that mesh axis's shards along the
      corresponding positional axis. Not mentioning a ``mesh`` axis name
      expresses a promise that the output values are equal along that mesh axis,
      and that rather than concatenating only a single value should be produced.
    check_rep: If True (default) enable additional validity checks and automatic
      differentiation optimizations. The validity checks concern whether any mesh
      axis names not mentioned in ``out_specs`` are consistent with how the outputs
      of ``f`` are replicated. Must be set False if using a Pallas kernel in ``f``.
    auto: (experimental) an optional set of axis names from ``mesh`` over which we
      do not shard the data or map the function, but rather we allow the
      compiler to control sharding. These names cannot be used in ``in_specs``,
      ``out_specs``, or in communication collectives in ``f``.

  Returns:
    A callable that applies the input function ``f`` across data sharded according to
    the ``mesh`` and ``in_specs``.

  Examples:
    For examples, refer to :ref:`sharded-computation` or `SPMD multi-device parallelism with shard_map`_.

  .. _SPMD multi-device parallelism with shard_map: https://docs.jax.dev/en/latest/notebooks/shard_map.html
  """
  axis_names = frozenset(mesh.axis_names) - auto
  return jshmap._shard_map(
      f, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
      check_vma=check_rep, axis_names=axis_names, _skip_mesh_check=True)
