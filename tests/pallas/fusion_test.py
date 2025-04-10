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

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax.experimental.pallas import fuser
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


class FusionTest(jtu.JaxTestCase):

  def test_basic_fusion(self):

    @jax.jit
    @fuser.fuse
    @fuser.fusable
    def f(x_fn, y_fn):
      x = x_fn()
      if y_fn is None:
        y_fn = lambda x: x
      return y_fn(x)

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=jnp.float32)
    np.testing.assert_array_equal(f(x), x)

  def test_separate_output_fusions_trivial(self):

    @fuser.fusable(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      if z_fns is None:
        z_fns = lambda x: x, lambda x: x
      z_fn1, z_fn2 = z_fns
      return z_fn1(x), z_fn2(y)

    @jax.jit
    @fuser.fuse
    def g(x, y):
      x, y = f(x, y)
      return x, y * 2

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=jnp.float32)
    y = jax.random.normal(jax.random.key(1), (1, 128), dtype=jnp.float32)
    x_out, y_out = g(x, y)
    np.testing.assert_array_equal(x_out, x)
    np.testing.assert_array_equal(y_out, y * 2)

  def test_separate_output_fusions_should_error_if_not_disjoint(self):

    @fuser.fusable(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      if z_fns is None:
        z_fns = lambda x: x, lambda x: x
      z_fn1, z_fn2 = z_fns
      return z_fn1(x), z_fn2(y)

    @jax.jit
    @fuser.fuse
    def g(x, y):
      x_res, y_res = f(x, y)
      return x_res + y_res

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=jnp.float32)
    y = jax.random.normal(jax.random.key(1), (128, 128), dtype=jnp.float32)

    with self.assertRaisesRegex(
        ValueError,
        "Outputs must be disjoint in order to use separate output fusions",
    ):
      g(x, y)

  def test_separate_output_fusions_allows_permute(self):

    @fuser.fusable(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      if z_fns is None:
        z_fns = lambda x: x, lambda x: x
      z_fn1, z_fn2 = z_fns
      return z_fn1(x), z_fn2(y)

    @jax.jit
    @fuser.fuse
    def g(x, y):
      x_res, y_res = f(x, y)
      return y_res * 2, x_res

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=jnp.float32)
    y = jax.random.normal(jax.random.key(1), (1, 128), dtype=jnp.float32)
    y_out, x_out = g(x, y)
    np.testing.assert_array_equal(x_out, x)
    np.testing.assert_array_equal(y_out, y * 2)

  def test_separate_output_fusions_with_nesting(self):

    @fuser.fusable(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      if z_fns is None:
        z_fns = lambda x: x, lambda x: x
      z_fn1, z_fn2 = z_fns
      return z_fn1(x), z_fn2(y)

    @jax.jit
    @fuser.fuse
    def g(x, y):
      x_res, y_res = f(x, y)
      return (x_res * 2, x_res + x_res), y_res

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=jnp.float32)
    y = jax.random.normal(jax.random.key(1), (1, 128), dtype=jnp.float32)
    (x1_out, x2_out), y_out = g(x, y)
    np.testing.assert_array_equal(x1_out, x * 2)
    np.testing.assert_array_equal(x2_out, x + x)
    np.testing.assert_array_equal(y_out, y)

  def test_separate_output_fusions_with_nesting_and_permutation(self):

    @fuser.fusable(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      if z_fns is None:
        z_fns = lambda x: x, lambda x: x
      z_fn1, z_fn2 = z_fns
      return z_fn1(x), z_fn2(y)

    @jax.jit
    @fuser.fuse
    def g(x, y):
      x_res, y_res = f(x, y)
      return y_res, (x_res * 2, x_res + x_res)

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=jnp.float32)
    y = jax.random.normal(jax.random.key(1), (1, 128), dtype=jnp.float32)
    y_out, (x1_out, x2_out) = g(x, y)
    np.testing.assert_array_equal(x1_out, x * 2)
    np.testing.assert_array_equal(x2_out, x + x)
    np.testing.assert_array_equal(y_out, y)

  def test_separate_output_fusions_with_deep_output_mask(self):

    @fuser.fusable(output_fusion_prefix=(True, (True, True)))
    def f(x_fn, y_fn, z_fn, o_fns):
      x = x_fn()
      y = y_fn()
      z = z_fn()
      if o_fns is None:
        o_fns = lambda x: x, (lambda x: x, lambda x: x)
      o_fn1, (o_fn2, o_fn3) = o_fns
      return o_fn1(x), (o_fn2(y), o_fn3(z))

    @jax.jit
    @fuser.fuse
    def g(x, y, z):
      x_res, (y_res, z_res) = f(x, y, z)
      return (x_res * 2, (y_res, z_res + z_res))

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=jnp.float32)
    y = jax.random.normal(jax.random.key(1), (1, 128), dtype=jnp.float32)
    z = jax.random.normal(jax.random.key(1), (128, 1), dtype=jnp.float32)
    x_out, (y_out, z_out) = g(x, y, z)
    np.testing.assert_array_equal(x_out, x * 2)
    np.testing.assert_array_equal(y_out, y)
    np.testing.assert_array_equal(z_out, z + z)

  def test_separate_output_fusions_with_reused_value(self):
    @fuser.fusable(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      if z_fns is None:
        z_fns = lambda x: x, lambda x: x
      z_fn1, z_fn2 = z_fns
      return z_fn1(x), z_fn2(y)

    @jax.jit
    @fuser.fuse
    def g(x, y, a):
      x_res, y_res = f(x, y)
      return y_res + a, (x_res * 2, x_res + x_res + a)

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=jnp.float32)
    y = jax.random.normal(jax.random.key(1), (1, 128), dtype=jnp.float32)
    a = jax.random.normal(jax.random.key(1), (1, 128), dtype=jnp.float32)
    y_out, (x1_out, x2_out) = g(x, y, a)
    np.testing.assert_array_equal(x1_out, x * 2)
    np.testing.assert_array_equal(x2_out, x + x + a)
    np.testing.assert_array_equal(y_out, y + a)

  def test_empty_fusion(self):
    @fuser.fusable
    def f(x_fn, y_fn):
      x = x_fn()
      if y_fn is None:
        y_fn = lambda x: x
      return y_fn(x)

    @jax.jit
    @fuser.fuse
    def g(x, a):
      _ = f(x)
      return a

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=jnp.float32)
    a = jax.random.normal(jax.random.key(1), (128, 128), dtype=jnp.float32)
    y_out = g(x, a)
    np.testing.assert_array_equal(y_out, a)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
