# Copyright 2025 The JAX Authors.
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

from absl.testing import parameterized
import jax
from jax import lax
from jax import numpy as jnp
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
from jax._src.lib import jaxlib_extension_version
import numpy as np


def randint_sample(shape):
  return jax.random.randint(jax.random.PRNGKey(42), shape, -100, 100)


class AllReduceTest(jt_multiprocess.MultiProcessTest):

  def test_psum_simple(self):
    mesh = jtu.create_mesh((jax.device_count(),), "x")
    spec = jax.P("x")

    @jax.shard_map(mesh=mesh, in_specs=spec, out_specs=spec)
    def f(x):
      return lax.psum(x, "x")

    out = f(jnp.array([1] * jax.device_count()))

    for o in out.addressable_shards:
      self.assertEqual(o.data, np.array([jax.device_count()]))

  @parameterized.parameters(
      (np.int32,), (jnp.float32,), (jnp.float16,), (jnp.bfloat16,)
  )
  def test_psum(self, dtype):
    mesh_shape = (jax.process_count(), jax.local_device_count())
    mesh = jtu.create_mesh(mesh_shape, ("x", "y"))
    spec = jax.P("x", "y")

    @jax.shard_map(mesh=mesh, in_specs=spec, out_specs=spec)
    def f(x):
      return lax.psum(x, ("x", "y"))

    xs = (
        jnp.arange(jax.local_device_count())
        + jax.process_index() * jax.local_device_count()
    )
    xs = jnp.expand_dims(xs, axis=0).astype(dtype)
    sharding = jax.NamedSharding(mesh, spec)
    global_xs = jax.make_array_from_process_local_data(sharding, xs, mesh_shape)
    local_xs = jnp.sum(jnp.arange(jax.device_count())).reshape(1, 1)
    out = f(global_xs)
    for actual in out.addressable_shards:
      jtu.check_close(actual.data, local_xs)

  def test_psum_subset_devices(self):
    mesh_shape = (jax.process_count(), jax.local_device_count())
    mesh = jtu.create_mesh(mesh_shape, ("x", "y"))
    spec = jax.P("x")

    @jax.shard_map(mesh=mesh, in_specs=spec, out_specs=spec)
    def f(x):
      return lax.psum(x, "x")

    xs = (
        jnp.arange(jax.local_device_count())
        + jax.process_index() * jax.local_device_count()
    )
    xs = jnp.expand_dims(xs, axis=0)
    sharding = jax.NamedSharding(mesh, spec)
    global_xs = jax.make_array_from_process_local_data(sharding, xs, mesh_shape)
    local_xs = (
        jnp.arange(jax.device_count())
        .reshape(mesh_shape)
        .sum(axis=0, keepdims=True)
    )
    out = f(global_xs)
    for actual in out.addressable_shards:
      jtu.check_close(actual.data, local_xs)

  def test_psum_multiple_operands(self):
    mesh_shape = (jax.process_count(), jax.local_device_count())
    mesh = jtu.create_mesh(mesh_shape, ("x", "y"))
    spec = jax.P("x", "y")
    sharding = jax.NamedSharding(mesh, spec)
    x = (
        jnp.arange(jax.local_device_count())
        + jax.process_index() * jax.local_device_count()
    )
    x = jnp.expand_dims(x, axis=(0, -1))

    @jax.shard_map(mesh=mesh, in_specs=spec, out_specs=spec)
    def f(x):
      return lax.psum(x, ("x", "y"))

    length = 100
    xs = jnp.tile(x, (1, 1, length))
    global_shape = mesh_shape + (length,)
    global_xs = jax.make_array_from_process_local_data(sharding, xs, global_shape)
    local_xs = jnp.sum(jnp.arange(jax.device_count())) * jnp.ones((1, 1, length))
    out = f(global_xs)
    for actual in out.addressable_shards:
      jtu.check_close(actual.data, local_xs)

    length = 200
    xs = jnp.tile(x, (1, 1, length))
    global_shape = mesh_shape + (length,)
    global_xs = jax.make_array_from_process_local_data(sharding, xs, global_shape)
    local_xs = jnp.sum(jnp.arange(jax.device_count())) * jnp.ones((1, 1, length))
    out = f(global_xs)
    for actual in out.addressable_shards:
      jtu.check_close(actual.data, local_xs)

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_psum_axis_index_groups(self):
    devices = list(range(jax.device_count()))
    axis_index_groups = [devices[0::2], devices[1::2]]
    print(axis_index_groups, jax.devices())
    f = jax.pmap(
        lambda x: lax.psum(x, "i", axis_index_groups=axis_index_groups),
        axis_name="i",
    )
    xs = randint_sample([jax.process_count(), jax.local_device_count(), 100])
    out = f(xs[jax.process_index()])

    xs = xs.reshape([jax.device_count(), 100])
    group0_expected = sum(xs[0::2, :])
    group1_expected = sum(xs[1::2, :])
    for i, shard in enumerate(out.addressable_shards):
      device_id = i + jax.process_index() * jax.local_device_count()
      expected = group0_expected if device_id % 2 == 0 else group1_expected
      np.testing.assert_array_equal(np.squeeze(shard.data), expected)

  def test_psum_scatter_large(self):
    if jaxlib_extension_version < 496:
      self.skipTest("Requires jaxlib_extension_version >= 496")
    n = jax.device_count()
    mesh = jtu.create_mesh((n,), ("x",))
    spec = jax.P("x", None)

    c = 2 * 1024 * 1024  # 8 MiB float32 output shard per rank
    global_shape = (n, n * c)
    sharding = jax.NamedSharding(mesh, spec)

    @jax.jit
    def f(x):
      return jax.shard_map(
          lambda y: lax.psum_scatter(y, "x", scatter_dimension=1, tiled=True),
          mesh=mesh,
          in_specs=spec,
          out_specs=spec,
      )(x)

    def shard_callback(index):
      r = index[0].start or 0
      chunks = [
          np.full((1, c), 100.0 * (k + 1) + r, dtype=np.float32)
          for k in range(n)
      ]
      return np.concatenate(chunks, axis=1)

    rank_sum = n * (n - 1) / 2
    for it in range(10):
      x = jax.make_array_from_callback(global_shape, sharding, shard_callback)
      got = f(x)
      for shard in got.addressable_shards:
        r = shard.index[0].start or 0
        expected = n * 100.0 * (r + 1) + rank_sum
        if not np.all(shard.data == expected):
          wrong = np.flatnonzero(shard.data.ravel() != expected)
          self.fail(
              f"iteration {it}: rank {r}: {wrong.size} of {c} elements wrong,"
              f" got {shard.data.ravel()[wrong[0]]}, want {expected}"
          )


if __name__ == "__main__":
  jt_multiprocess.main()
