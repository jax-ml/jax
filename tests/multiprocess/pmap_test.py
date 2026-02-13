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

"""Multihost tests for pmap."""

import unittest

from absl.testing import parameterized
import jax
from jax import lax
from jax._src import array
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np


def sorted_devices():
  devices = sorted(
      jax.devices(), key=lambda d: (d.process_index(), d.core_on_chip))
  if len(devices) != 8:
    raise unittest.SkipTest("Test assumes that it runs on a TPU donut")
  return devices


class PmapTestMultiHost(jt_multiprocess.MultiProcessTest):

  @jtu.ignore_warning(category=DeprecationWarning)
  def testBasic(self):
    elems_per_host = 4
    devices = jax.local_devices()
    x = [np.arange(i, i + elems_per_host) + jax.process_index() * elems_per_host
         for i in range(len(devices))]
    y = jax.device_put_sharded(x, devices)
    f = jax.pmap(lambda x: lax.psum(x, "i"), axis_name="i")
    out = f(y)

    expected_out = np.array([
        np.arange(i, i + elems_per_host) + p * elems_per_host  # pylint: disable=g-complex-comprehension
        for p in range(jax.process_count()) for i in range(len(devices))
    ])

    self.assertIsInstance(out, array.ArrayImpl)
    if jax.config.jax_pmap_shmap_merge:
      self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
    else:
      self.assertIsInstance(out.sharding, jax.sharding.PmapSharding)
    np.testing.assert_array_equal(
        out, np.array([expected_out.sum(axis=0)] * len(devices)))

  def testLocalPmap(self):
    z = jax.pmap(
        lambda x: lax.axis_index("i"),
        axis_name="i",
        devices=jax.local_devices(),
    )(np.arange(jax.local_device_count()))
    np.testing.assert_array_equal(z, np.arange(jax.local_device_count()))

  @parameterized.named_parameters(
      ("sharded_dim_0", 0),
      ("sharded_dim_1", 1),
  )
  @jtu.ignore_warning(category=DeprecationWarning)
  def test_default_pmap_sharding(self, sharded_dim):    n = jax.local_device_count()
    shape = (n, 1) if sharded_dim == 0 else (1, n)

    ps = jax.sharding.PmapSharding.default(shape, sharded_dim)
    inp = jnp.arange(np.prod(shape)).reshape(shape)
    compiled = jax.pmap(lambda x: x, in_axes=sharded_dim).lower(inp).compile()
    pmap_in_sharding, = compiled._executable.unsafe_call.in_handler.in_shardings

    self.assertEqual(ps._device_assignment, pmap_in_sharding._device_assignment)
    self.assertEqual(ps.sharding_spec, pmap_in_sharding.sharding_spec)

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_global_axis_size_initial_style(self):
    xs = jnp.ones(jax.local_device_count())
    pmapped_f = jax.pmap(lambda x: jax.lax.all_gather(x, "i"), axis_name="i")
    jaxpr = jax.make_jaxpr(pmapped_f)(xs)
    jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, xs)  # does not crash

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_array_device_size_mismatch_with_mesh(self):
    """Test pmap when input array's device count differs from pmap mesh."""
    local_devices = jax.local_devices()
    n = len(local_devices)

    local_mesh = jax.sharding.Mesh(np.array(local_devices), ("x",))
    local_sharding = jax.sharding.NamedSharding(
        local_mesh, jax.sharding.PartitionSpec("x")
    )

    local_data = jnp.arange(n, dtype=jnp.float32)
    local_arr = jax.device_put(local_data, local_sharding)

    f = jax.pmap(lambda x: x + 1, devices=local_devices)
    out = f(local_arr)
    expected = np.arange(n, dtype=np.float32) + 1
    np.testing.assert_array_equal(out, expected)

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_pmap_with_scalars(self):
    """Test pmap with scalar inputs."""
    n = jax.local_device_count()
    scalars = [1.0] * n
    f = jax.pmap(lambda x: x + 1)
    out = f(np.array(scalars))
    np.testing.assert_array_equal(out, np.array([2.0] * n))

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_pmap_with_numpy_arrays(self):
    """Test pmap with numpy array inputs."""
    n = jax.local_device_count()
    np_input = np.arange(n * 4, dtype=np.float32).reshape((n, 4))
    f = jax.pmap(lambda x: x * 2)
    out = f(np_input)
    np.testing.assert_array_equal(out, np_input * 2)

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_pmap_with_prng_keys(self):
    """Test pmap with PRNGKey inputs."""
    n = jax.local_device_count()
    keys = jax.random.split(jax.random.key(0), n)
    f = jax.pmap(lambda k: jax.random.normal(k, shape=(2,)))
    out = f(keys)
    self.assertEqual(out.shape, (n, 2))
    for i in range(n):
      for j in range(i + 1, n):
        self.assertFalse(np.allclose(out.addressable_data(i), out.addressable_data(j)))

  @jtu.ignore_warning(category=DeprecationWarning)
  def test_pmap_with_float0(self):
    """Test pmap with float0 dtype arrays (used in autodiff for integer args)."""
    n = jax.local_device_count()
    float0_arr = np.zeros((n, 3), dtype=jax.dtypes.float0)

    f = jax.pmap(lambda x: x)
    out = f(float0_arr)
    self.assertEqual(out.shape, (n, 3))
    self.assertEqual(out.dtype, np.dtype(bool))

  @jtu.ignore_warning(category=UserWarning,
                      message=".*Using jit-of-pmap can lead to inefficient data movement")
  def test_replicated_output_sharding_multi_process(self):
    if not jax.config.jax_pmap_shmap_merge:
      self.skipTest("Only applies to pmap shmap merge")

    f = jax.pmap(lambda x: x, axis_name="i", out_axes=None)
    x = jnp.arange(jax.local_device_count())
    out = f(x)

    self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
    self.assertEqual(out.sharding.spec, jax.sharding.PartitionSpec())
    self.assertEqual(out.sharding.mesh.size, jax.local_device_count())


if __name__ == "__main__":
  jt_multiprocess.main()
