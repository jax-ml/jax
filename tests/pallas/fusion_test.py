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
import dataclasses

from absl.testing import absltest
import jax
from jax import lax
from jax._src import core as jax_core
from jax._src import hijax
from jax._src import test_util as jtu
from jax.experimental.pallas import fuser
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


class FusionTest(jtu.JaxTestCase):

  def test_basic_fusion(self):

    @jax.jit
    @fuser.fuse
    @fuser.fusible
    def f(x_fn, y_fn):
      x = x_fn()
      if y_fn is None:
        y_fn = lambda x: x
      return y_fn(x)

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=jnp.float32)
    np.testing.assert_array_equal(f(x), x)

  def test_separate_output_fusions_trivial(self):

    @fuser.fusible(output_fusion_prefix=(True, True))
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

  def test_custom_fusion(self):
    const = jnp.array(1.0, dtype=jnp.float32)
    const2 = jnp.array(2.0, dtype=jnp.float32)
    const3 = jnp.array(3.0, dtype=jnp.float32)

    @fuser.custom_fusion
    def c(x, y):
      return x + y + const

    c.def_pull_block_spec(lambda bss: (bss[0], bss[0]))
    c.def_push_block_spec(lambda bss: (bss[0],))
    c.def_eval_rule(lambda _, x, y: (c(x, y),))
    c.def_pallas_impl(lambda x, y: x + y + const2 + const3)

    @fuser.fusible(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      if z_fns is None:
        z_fns = lambda x: x, lambda x: x
      z_fn1, z_fn2 = z_fns
      return z_fn1(x), z_fn2(y)

    def g(x, y, z):
      x, y = f(x, c(y, z))
      return c(x, z), y * 2

    x = jax.random.normal(jax.random.key(0), (4, 4), dtype=jnp.float32)
    y = jax.random.normal(jax.random.key(1), (1, 4), dtype=jnp.float32)
    z = jax.random.normal(jax.random.key(2), (1, 4), dtype=jnp.float32)
    x_out, y_out = g(x, y, z)
    np.testing.assert_array_equal(x_out, (x + z + 1.0))
    np.testing.assert_array_equal(y_out, (y + z + 1.0) * 2)

    g_fused = jax.jit(fuser.fuse(g))
    x_out, y_out = g_fused(x, y, z)
    np.testing.assert_allclose(x_out, (x + z + 1.0))
    np.testing.assert_allclose(y_out, (y + z + 1.0) * 2)

  def test_separate_output_fusions_should_error_if_not_disjoint(self):

    @fuser.fusible(output_fusion_prefix=(True, True))
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

    @fuser.fusible(output_fusion_prefix=(True, True))
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

    @fuser.fusible(output_fusion_prefix=(True, True))
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

    @fuser.fusible(output_fusion_prefix=(True, True))
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

    @fuser.fusible(output_fusion_prefix=(True, (True, True)))
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

    @fuser.fusible(output_fusion_prefix=(True, True))
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

    @fuser.fusible
    def f(x_fn, y_fn):
      x = x_fn()
      if y_fn is None:
        y_fn = lambda x: x
      return y_fn(x)

    @jax.jit
    @fuser.fuse
    def g(x, a):
      _ = lax.dce_sink(f(x))
      return a

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=jnp.float32)
    a = jax.random.normal(jax.random.key(1), (128, 128), dtype=jnp.float32)
    y_out = g(x, a)
    np.testing.assert_array_equal(y_out, a)


@dataclasses.dataclass(frozen=True)
class ArrayTuple:
  x0: jax.Array
  x1: jax.Array


@dataclasses.dataclass(frozen=True)
class ArrayTupleTy(hijax.HiType):
  x0: jax_core.ShapedArray
  x1: jax_core.ShapedArray

  def lo_ty(self) -> list[jax_core.ShapedArray]:
    return [self.x0, self.x1]

  def lower_val(self, hi_val: ArrayTuple) -> list[jax.Array]:
    return [hi_val.x0, hi_val.x1]

  def raise_val(self, x0, x1) -> ArrayTuple:
    return ArrayTuple(x0, x1)


hijax.register_hitype(
    ArrayTuple, lambda t: ArrayTupleTy(jax.typeof(t.x0), jax.typeof(t.x1))
)


class FusionHijaxTest(jtu.JaxTestCase):

  def test_basic_fusion(self):

    @jax.jit
    @fuser.fuse
    @fuser.fusible
    def f(x_fn, y_fn):
      x = x_fn()
      if y_fn is None:
        y_fn = lambda x: x
      return y_fn(x)

    xt = ArrayTuple(x0=jnp.ones((8, 8)), x1=jnp.zeros(4))
    ot = f(xt)
    np.testing.assert_array_equal(ot.x0, xt.x0)
    np.testing.assert_array_equal(ot.x1, xt.x1)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
