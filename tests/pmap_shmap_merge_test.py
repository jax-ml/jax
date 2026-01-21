# Copyright 2026 The JAX Authors.
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

import math
import unittest
import warnings

from absl.testing import absltest
import jax
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import stages
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)

# Suppress the deprecation warning from @config.pmap_shmap_merge(True) decorator
# which is triggered at class definition time.
warnings.filterwarnings(
    'ignore',
    message='Setting `jax_pmap_shmap_merge` is deprecated',
    category=DeprecationWarning,
)


class PmapShmapMergeTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 2:
      raise unittest.SkipTest('test requires at least two devices')

  @config.pmap_shmap_merge(True)
  def test_store_exception(self):
    def f(x):
      return x

    inp = jnp.ones((jax.device_count(), 1), dtype=jnp.float32)
    jax.pmap(f, axis_name='i')(inp)
    inp = jnp.ones((jax.device_count(), 1), dtype=jnp.int32)
    jax.pmap(f, axis_name='i')(inp)

  @config.pmap_shmap_merge(True)
  def test_prng_key(self):
    keys = jax.random.split(jax.random.key(0), jax.device_count())
    out = jax.pmap(lambda x: x)(keys)
    self.assertEqual(type(out), type(keys))
    out = jax.pmap(lambda x, y: y, in_axes=(0, None))(keys, jax.random.key(0))
    self.assertEqual(type(out), type(keys))
    out = jax.pmap(lambda x, y: y, in_axes=(0, None), out_axes=None)(
        keys, jax.random.key(0)
    )
    self.assertEqual(type(out), type(keys))

  @config.pmap_shmap_merge(True)
  def test_lower_with_flattened_args(self):
    shape = (jax.device_count(), 3)

    inputs = np.reshape(np.arange(math.prod(shape)), shape)
    # The shard_map implementation of pmap takes pytree args, but the inner
    # jitted_f must take flattened args.
    _ = jax.pmap(lambda x: x[0]).lower((inputs, ())).compile()  # doesn't crash

  @config.pmap_shmap_merge(True)
  def test_float0_dtype_input(self):
    inputs = np.array([b''] * jax.device_count(), dtype=dtypes.float0)
    _ = jax.pmap(lambda x: x)(inputs)  # doesn't crash

  @config.pmap_shmap_merge(True)
  def test_float0_dtype_output(self):
    inputs = np.ones(jax.device_count())
    _ = jax.pmap(lambda x: jnp.array(b'', dtype=dtypes.float0))(
        inputs
    )  # doesn't crash

  @config.pmap_shmap_merge(True)
  def test_lowered_args_info(self):
    shmap_lowered = jax.pmap(lambda x: x).lower(
        (jnp.ones((1,), jnp.float32), ())
    )
    aval = core.ShapedArray((1,), jnp.float32)
    expected_args_info = (
        (
            (
                stages.ArgInfo(aval, donated=False),
                (),
            ),
        ),
        {},
    )
    self.assertEqual(
        shmap_lowered.args_info, expected_args_info
    )  # doesn't crash

  @config.pmap_shmap_merge(True)
  def test_wrapped(self):
    f = lambda x: x
    g = jax.pmap(f)
    self.assertTrue(hasattr(g, '__wrapped__'))
    self.assertEqual(g.__wrapped__, f)

  @config.pmap_shmap_merge(True)
  def test_numpy_input_sharding(self):
    # Test that pmap correctly handles numpy array inputs by providing
    # explicit in_shardings to the underlying jit(shard_map).
    # Without explicit in_shardings, jit would default to UnspecifiedValue
    # for numpy inputs, causing failures.
    np_input = np.arange(jax.device_count(), dtype=np.float32)
    result = jax.pmap(lambda x: x * 2)(np_input)
    expected = np_input * 2
    self.assertAllClose(result, expected)


if __name__ == '__main__':
  absltest.main()
