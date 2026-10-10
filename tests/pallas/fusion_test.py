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
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import core as jax_core
from jax._src import hijax
from jax._src import test_util as jtu
from jax._src.pallas.fuser import fusible_dtype
from jax.experimental import rebindable as rebindable_lib
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

  def test_nested_fuse(self):
    @fuser.fusible
    def f(x_fn, y_fn):
      x = x_fn()
      if y_fn is None:
        y_fn = lambda x: x
      return y_fn(x)

    @fuser.fuse
    def inner(x):
      return f(x) + 1.0

    @jax.jit
    @fuser.fuse
    def outer(x):
      return inner(x) * 2.0

    x = jnp.ones((4, 4), dtype=jnp.float32)
    np.testing.assert_allclose(outer(x), (x + 1.0) * 2.0)
    np.testing.assert_allclose(inner(x), x + 1.0)

    @jax.jit
    @fuser.fuse
    @fuser.fuse
    def double_fused(x):
      return f(x) + 3.0

    np.testing.assert_allclose(double_fused(x), x + 3.0)

    # Pass fuse wrapper as an argument
    @functools.partial(fuser.fuse, static_argnums=0)
    def outer_with_fn_arg(fn, x):
      return fn(x) * 2.0

    np.testing.assert_allclose(
        jax.jit(outer_with_fn_arg, static_argnums=0)(inner, x), (x + 1.0) * 2.0
    )

  def test_nested_fuse_cache(self):
    @fuser.fusible
    def f(x_fn, y_fn):
      x = x_fn()
      if y_fn is None:
        y_fn = lambda x: x
      return y_fn(x)

    @fuser.fuse
    def inner(x):
      return f(x) + 1.0

    @fusible_dtype.physicalize
    def cached_inner(x):
      return inner(x)

    @jax.jit
    @fuser.fuse
    def outer_cached(x):
      return cached_inner(x) * 2.0

    x = jnp.ones((4, 4), dtype=jnp.float32)
    np.testing.assert_allclose(cached_inner(x), x + 1.0)
    np.testing.assert_allclose(outer_cached(x), (x + 1.0) * 2.0)

  def test_separate_output_fusions_trivial(self):

    @fuser.fusible(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      z_fn1, z_fn2 = z_fns
      if z_fn1 is None:
        z_fn1 = lambda x: x
      if z_fn2 is None:
        z_fn2 = lambda x: x
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

  def test_output_fusions_are_actually_trivial(self):

    @fuser.fusible(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      z_fn1, z_fn2 = z_fns
      self.assertIsNone(z_fn1)
      self.assertIsNone(z_fn2)
      if z_fn1 is None:
        z_fn1 = lambda x: x
      if z_fn2 is None:
        z_fn2 = lambda x: x
      return z_fn1(x), z_fn2(y)

    @jax.jit
    @fuser.fuse
    def g(x, y):
      x, y = f(x, y)
      return x, y

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=jnp.float32)
    y = jax.random.normal(jax.random.key(1), (1, 128), dtype=jnp.float32)
    x_out, y_out = g(x, y)
    np.testing.assert_array_equal(x_out, x)
    np.testing.assert_array_equal(y_out, y)

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
      z_fn1, z_fn2 = z_fns
      if z_fn1 is None:
        z_fn1 = lambda x: x
      if z_fn2 is None:
        z_fn2 = lambda x: x
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

  def test_custom_fusion_errors(self):
    x = jnp.ones((4, 4), dtype=jnp.float32)

    @fuser.custom_fusion
    def missing_eval(x):
      return x + 1.0

    with self.assertRaisesRegex(ValueError, "missing an evaluation rule"):
      missing_eval(x)

    missing_eval.def_eval_rule(lambda _, x: (missing_eval(x),))
    with self.assertRaisesRegex(ValueError, "missing a pull_block_spec rule"):
      missing_eval(x)

    missing_eval.def_pull_block_spec(lambda bss: (bss[0],))
    missing_eval.def_pallas_impl(lambda x: jnp.ones((2, 2), dtype=jnp.float32))
    with self.assertRaisesRegex(
        ValueError, "mismatched output abstract values"
    ):
      jax.jit(missing_eval)(x)

  def test_separate_output_fusions_should_error_if_not_disjoint(self):

    @fuser.fusible(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      z_fn1, z_fn2 = z_fns
      if z_fn1 is None:
        z_fn1 = lambda x: x
      if z_fn2 is None:
        z_fn2 = lambda x: x
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
      z_fn1, z_fn2 = z_fns
      if z_fn1 is None:
        z_fn1 = lambda x: x
      if z_fn2 is None:
        z_fn2 = lambda x: x
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
      z_fn1, z_fn2 = z_fns
      if z_fn1 is None:
        z_fn1 = lambda x: x
      if z_fn2 is None:
        z_fn2 = lambda x: x
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
      z_fn1, z_fn2 = z_fns
      if z_fn1 is None:
        z_fn1 = lambda x: x
      if z_fn2 is None:
        z_fn2 = lambda x: x
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
      o_fn1, (o_fn2, o_fn3) = o_fns
      if o_fn1 is None:
        o_fn1 = lambda x: x
      if o_fn2 is None:
        o_fn2 = lambda x: x
      if o_fn3 is None:
        o_fn3 = lambda x: x
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
      z_fn1, z_fn2 = z_fns
      if z_fn1 is None:
        z_fn1 = lambda x: x
      if z_fn2 is None:
        z_fn2 = lambda x: x
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

  def test_vjp_support(self):
    @fuser.fusible
    def f(x_fn, y_fn, z_fn):
      x = x_fn()
      y = y_fn()
      z = x * y
      if z_fn is None:
        z_fn = lambda x: x
      return z_fn(z)

    x, y = jnp.array(2.0), jnp.array(3.0)
    val, vjp_fun = jax.vjp(f, x, y)
    np.testing.assert_allclose(val, 6.0)

    grads = vjp_fun(jnp.array(1.0))
    np.testing.assert_allclose(grads, (3.0, 2.0))

  def test_vmap_support(self):
    @fuser.fusible
    def f(x_fn, y_fn, out_fn):
      x = x_fn()
      y = y_fn()
      if out_fn is None:
        out_fn = lambda x: x
      return out_fn(x * y)

    x = jnp.array([2.0])
    y = jnp.array(4.0)

    val = jax.vmap(f, in_axes=(0, None))(x, y)
    np.testing.assert_allclose(val, jnp.array([8.0]))

  def test_effect_support(self):
    ref = jax.new_ref(jnp.array(0.0))

    @fuser.fusible
    def f(x_fn, y_fn, out_fn):
      x = x_fn()
      y = y_fn()
      ref[...] = x + y
      if out_fn is None:
        out_fn = lambda x: x
      return out_fn(x * y)

    x, y = jnp.array(2.0), jnp.array(3.0)
    closed_jaxpr = jax.make_jaxpr(f)(x, y)
    self.assertLen(closed_jaxpr.effects, 1)

    jax.core.eval_jaxpr(closed_jaxpr, closed_jaxpr.consts, x, y)

    np.testing.assert_allclose(ref[...], 5.0)

  def test_physicalize_fusible(self):

    @fuser.fusible
    def f(x_fn, out_fn):
      x = x_fn()
      if out_fn is None:
        out_fn = lambda x: x
      return out_fn(x + 1.0)

    x = jnp.array(1.0)
    y = fusible_dtype.physicalize(f)(x)
    np.testing.assert_allclose(y, 2.0)

  @parameterized.named_parameters(
      {"testcase_name": f"cvjp3_{cvjp3}_remat3_{remat3}", "cvjp3": cvjp3, "remat3": remat3}
      for cvjp3 in (False, True)
      for remat3 in (False, True)
  )
  def test_fusible_physicalize_custom_vjp_and_remat(self, cvjp3, remat3):
    with config.custom_vjp3(cvjp3), config.remat3(remat3):
      @jax.custom_vjp
      def custom_fn(x):
        return x * 2.0
      def custom_fn_fwd(x):
        return custom_fn(x), None
      def custom_fn_bwd(res, g):
        return (g * 2.0,)
      custom_fn.defvjp(custom_fn_fwd, custom_fn_bwd)

      def f(x):
        return jax.checkpoint(custom_fn)(x) + 1.0

      x = jnp.array(3.0)
      y = fusible_dtype.physicalize(f)(x)
      np.testing.assert_allclose(y, 7.0)

  @parameterized.named_parameters(
      {"testcase_name": f"cvjp3_{cvjp3}", "cvjp3": cvjp3}
      for cvjp3 in (False, True)
  )
  def test_fusible_physicalize_custom_vjp_grad(self, cvjp3):
    with config.custom_vjp3(cvjp3):
      @functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
      def custom_fn(x, scale):
        return x * scale
      def custom_fn_fwd(x, scale):
        return custom_fn(x, scale), None
      def custom_fn_bwd(scale, res, g):
        return (g * scale,)
      custom_fn.defvjp(custom_fn_fwd, custom_fn_bwd)

      def f(x):
        return custom_fn(x, 2.0) + 1.0

      x = jnp.array(3.0)
      gy = jax.grad(fusible_dtype.physicalize(f))(x)
      np.testing.assert_allclose(gy, 2.0)

  @parameterized.parameters([False, True])
  def test_fusible_physicalize_custom_vjp_grad_with_consts(self, cvjp3):
    with config.custom_vjp3(cvjp3):
      scale = jnp.array(2.0)  # Closed over, so a const of the call_jaxpr.

      @jax.custom_vjp
      def custom_fn(x):
        return x * scale
      def custom_fn_fwd(x):
        return custom_fn(x), None
      def custom_fn_bwd(res, g):
        del res
        return (g * scale,)
      custom_fn.defvjp(custom_fn_fwd, custom_fn_bwd)

      def f(x):
        return custom_fn(x) + 1.0

      x = jnp.array(3.0)
      gy = jax.grad(fusible_dtype.physicalize(f))(x)
      np.testing.assert_allclose(gy, 2.0)

  def test_fusible_outside_fuse(self):
    @fuser.fusible
    def f(x_fn, out_fn):
      if out_fn is None:
        out_fn = lambda x: x
      return out_fn(x_fn() + 1.0)

    mesh = jax.sharding.Mesh(jax.devices()[:1], ('x',))
    P = jax.sharding.PartitionSpec

    @jax.custom_vjp
    def g(x):
      return f(x)

    def g_fwd(x):
      y = f(x)
      return y, x

    def g_bwd(_, grad):
      return (grad,)

    g.defvjp(g_fwd, g_bwd)

    def body(x):
      return g(x)

    @jax.jit
    def run(x):
      return jax.shard_map(body, mesh=mesh, in_specs=P(), out_specs=P())(x)

    x = jnp.ones((128, 128))
    y = run(x)
    np.testing.assert_allclose(y, jnp.full((128, 128), 2.0))

  def test_fusible_jit(self):
    @fuser.fusible
    def quantize(x_fn, out_fn):
      x = x_fn()
      return x * 2.0

    x = jnp.ones((128, 128))
    result = jax.jit(quantize)(x)
    np.testing.assert_allclose(result, x * 2.0)

  def test_fusible_jit_grad(self):
    @fuser.fusible
    def quantize(x_fn, out_fn):
      x = x_fn()
      return jnp.sum(x * 2.0)

    x = jnp.ones((128, 128))
    result = jax.jit(jax.grad(quantize))(x)
    np.testing.assert_allclose(result, jnp.full_like(x, 2.0))

  def test_fusible_ref_arg_grad(self):
    # A Ref passed as a fusible operand: its gradient is accumulated into the
    # Ref's gradient ref and reaches the array it was created from.
    @fuser.fusible
    def scale(x_fn, y_ref_fn, out_fn):
      del out_fn
      return x_fn() * y_ref_fn()[...]

    def loss(x, y):
      return jnp.sum(scale(x, jax.new_ref(y)))

    x = jnp.arange(16.0).reshape(4, 4)
    y = jnp.full((4, 4), 3.0)
    dx, dy = jax.jit(jax.grad(loss, argnums=(0, 1)))(x, y)
    np.testing.assert_allclose(dx, y)
    np.testing.assert_allclose(dy, x)

  def test_fusible_shard_map_jit(self):
    @fuser.fusible
    def quantize(x_fn, out_fn):
      x = x_fn()
      return x * 2.0

    mesh = jax.make_mesh((jax.device_count(),), ('data',))

    @jax.shard_map(
        in_specs=(jax.P('data'),), out_specs=jax.P('data'), mesh=mesh
    )
    def f(x):
      return quantize(x)

    sharding = jax.sharding.NamedSharding(mesh, jax.P('data'))
    x = jax.device_put(jnp.ones(jax.device_count() * 128), sharding)
    result = jax.jit(f)(x)
    np.testing.assert_allclose(result, x * 2.0)

  def test_fusible_shard_map_jit_grad(self):

    @fuser.fusible
    def quantize(x_fn, out_fn):
      x = x_fn()
      return jax.lax.psum(jnp.sum(x * 2.0), axis_name='data')

    mesh = jax.make_mesh((jax.device_count(),), ('data',))

    @jax.shard_map(
        in_specs=(jax.P('data'),), out_specs=jax.P(), mesh=mesh, check_vma=False
    )
    def f(x):
      return quantize(x)

    sharding = jax.sharding.NamedSharding(mesh, jax.P('data'))
    x = jax.device_put(jnp.ones(jax.device_count() * 128), sharding)
    result = jax.jit(jax.grad(f))(x)
    np.testing.assert_allclose(result, jnp.full_like(x, 2.0))

  def test_fusible_shard_map_mlir_compilation(self):
    @fuser.fusible
    def quantize(x_fn, out_fn):
      x = x_fn()
      return jnp.sum(x * 2.0)

    mesh = jax.make_mesh((jax.device_count(),), ('data',))

    @jax.shard_map(
        in_specs=(jax.P('data'),), out_specs=jax.P(), mesh=mesh, check_vma=False
    )
    def f(x):
      return quantize(x)

    sharding = jax.sharding.NamedSharding(mesh, jax.P('data'))
    x = jax.device_put(jnp.ones(jax.device_count() * 128), sharding)

    lowered = jax.jit(jax.grad(f)).lower(x)
    self.assertIsNotNone(lowered)

  def test_fusible_custom_vjp_shard_map_grad(self):
    @fuser.fusible
    def quantize(x_fn, out_fn):
      x = x_fn()
      if out_fn is None:
        out_fn = lambda x: x
      return out_fn(x * 2.0)

    @fuser.fuse
    def backward_fusion(grad):
      return quantize(grad) * 2.0

    @functools.partial(jax.custom_vjp, nondiff_argnums=())
    def wrapped_fun(x):
      return x * 2.0

    def wrapped_fun_fwd(x):
      return wrapped_fun(x), None

    def wrapped_fun_bwd(res, grad):
      del res
      return (backward_fusion(grad),)

    wrapped_fun.defvjp(wrapped_fun_fwd, wrapped_fun_bwd)

    mesh = jax.make_mesh((jax.device_count(),), ('data',))

    @jax.shard_map(
        in_specs=(jax.P('data'),),
        out_specs=jax.P('data'),
        mesh=mesh,
        check_vma=False,
    )
    def f(x):
      return wrapped_fun(x)

    sharding = jax.sharding.NamedSharding(mesh, jax.P('data'))
    x = jax.device_put(jnp.ones(jax.device_count() * 128), sharding)

    lowered = jax.jit(jax.grad(lambda x: jnp.sum(jax.checkpoint(f)(x)))).lower(
        x
    )
    self.assertIsNotNone(lowered)

  def test_fusible_remat_lower(self):
    @fuser.fusible
    def quantize(x_fn, out_fn):
      x = x_fn()
      return x * 2.0

    @jax.checkpoint
    def f(x):
      return quantize(x)

    x = jnp.ones(128)
    lowered = jax.jit(f).lower(x)
    self.assertIsNotNone(lowered)

  def test_fusible_shard_map_checkpoint_grad(self):
    @fuser.fusible
    def quantize(x_fn, out_fn):
      del out_fn
      x = x_fn()
      return x * 2.0

    mesh = jax.make_mesh((jax.device_count(),), ('data',))

    @jax.checkpoint
    @functools.partial(
        jax.shard_map,
        in_specs=(jax.P('data'),),
        out_specs=jax.P('data'),
        mesh=mesh,
        check_vma=False,
    )
    def f(x):
      return quantize(x)

    sharding = jax.sharding.NamedSharding(mesh, jax.P('data'))
    x = jax.device_put(jnp.ones(jax.device_count() * 128), sharding)

    def loss(x):
      return jnp.sum(f(x))

    lowered = jax.jit(jax.grad(loss)).lower(x)
    self.assertIsNotNone(lowered)

  def test_fusible_custom_gradient_shard_map_grad(self):
    @fuser.fusible
    def quantize(x_fn, out_fn):
      del out_fn
      x = x_fn()
      return x * 2.0

    @fuser.fuse
    def backward_fusion(grad):
      return quantize(grad) * 2.0

    @jax.custom_gradient
    def wrapped_fun(x):
      return x * 2.0, lambda grad: (backward_fusion(grad),)

    mesh = jax.make_mesh((jax.device_count(),), ('data',))

    @jax.shard_map(
        in_specs=(jax.P('data'),),
        out_specs=jax.P('data'),
        mesh=mesh,
        check_vma=False,
    )
    def f(x):
      return wrapped_fun(x)

    sharding = jax.sharding.NamedSharding(mesh, jax.P('data'))
    x = jax.device_put(jnp.ones(jax.device_count() * 128), sharding)

    lowered = jax.jit(jax.grad(lambda x: jnp.sum(jax.checkpoint(f)(x)))).lower(
        x
    )
    self.assertIsNotNone(lowered)

  def test_fusible_custom_gradient_shard_map_grad_no_fuse(self):
    @fuser.fusible
    def quantize(x_fn, out_fn):
      del out_fn
      x = x_fn()
      return x * 2.0

    def backward_fusion(grad):
      return quantize(grad) * 2.0

    @jax.custom_gradient
    def wrapped_fun(x):
      return x * 2.0, lambda grad: (backward_fusion(grad),)

    mesh = jax.make_mesh((jax.device_count(),), ('data',))

    @jax.shard_map(
        in_specs=(jax.P('data'),),
        out_specs=jax.P('data'),
        mesh=mesh,
        check_vma=False,
    )
    def f(x):
      return jax.checkpoint(wrapped_fun)(x)

    sharding = jax.sharding.NamedSharding(mesh, jax.P('data'))
    x = jax.device_put(jnp.ones(jax.device_count() * 128), sharding)

    lowered = jax.jit(jax.grad(lambda x: jnp.sum(f(x)))).lower(x)
    self.assertIsNotNone(lowered)

  def test_fusible_custom_gradient_forward_shard_map_grad(self):
    @fuser.fusible
    def quantize(x_fn, out_fn):
      del out_fn
      x = x_fn()
      return x * 2.0

    @jax.custom_gradient
    def wrapped_fun(x):
      res = quantize(x)
      return res, lambda grad: (grad * 2.0,)

    mesh = jax.make_mesh((jax.device_count(),), ('data',))

    @jax.shard_map(
        in_specs=(jax.P('data'),),
        out_specs=jax.P('data'),
        mesh=mesh,
        check_vma=False,
    )
    def f(x):
      return wrapped_fun(x)

    sharding = jax.sharding.NamedSharding(mesh, jax.P('data'))
    x = jax.device_put(jnp.ones(jax.device_count() * 128), sharding)

    lowered = jax.jit(jax.grad(lambda x: jnp.sum(jax.checkpoint(f)(x)))).lower(
        x
    )
    self.assertIsNotNone(lowered)

  def test_fusible_nested_custom_derivatives(self):
    @fuser.fusible
    def quantize(x_fn, out_fn):
      del out_fn
      x = x_fn()
      return x * 2.0

    @jax.custom_gradient
    def inner_fun(x):
      res = quantize(x)
      return res, lambda grad: (grad * 2.0,)

    @jax.custom_vjp
    def outer_fun(x):
      return inner_fun(x)

    def outer_fun_fwd(x):
      return inner_fun(x), None

    def outer_fun_bwd(res, grad):
      del res
      return (grad,)

    outer_fun.defvjp(outer_fun_fwd, outer_fun_bwd)

    mesh = jax.make_mesh((jax.device_count(),), ('data',))

    @jax.shard_map(
        in_specs=(jax.P('data'),),
        out_specs=jax.P('data'),
        mesh=mesh,
        check_vma=False,
    )
    def f(x):
      return outer_fun(x)

    sharding = jax.sharding.NamedSharding(mesh, jax.P('data'))
    x = jax.device_put(jnp.ones(jax.device_count() * 128), sharding)

    lowered = jax.jit(f).lower(x)
    self.assertIsNotNone(lowered)

  @parameterized.parameters(
      {"static_argnums": 1},
      {"static_argnames": "mode"},
  )
  def test_fuse_static_args(self, **kwargs):
    @fuser.fusible
    def f(x_fn, y_fn):
      x = x_fn()
      if y_fn is None:
        y_fn = lambda x: x
      return y_fn(x)

    @fuser.fuse(**kwargs)
    def g(x, mode="identity"):
      if mode == "double":
        return f(x * 2)
      elif mode == "negate":
        return f(-x)
      else:
        return f(x)

    x = jax.random.normal(jax.random.key(0), (4, 4), dtype=jnp.float32)
    np.testing.assert_array_equal(g(x, mode="double"), x * 2)
    np.testing.assert_array_equal(g(x, "negate"), -x)
    np.testing.assert_array_equal(g(x), x)

  def test_fusible_jvp_symbolic_zeros_jacfwd(self):
    @fuser.fusible
    def add_scaled(x_fn, y_fn, out_fn):
      x, y = x_fn(), y_fn()
      if out_fn is None:
        out_fn = lambda v: v
      return out_fn(x * 2.0 + y)

    x = jnp.ones((4, 4))
    y = jnp.ones((4, 4))

    # Partial forward-mode differentiation: differentiating w.r.t. x leaves y
    # with a symbolic zero tangent during JVP.
    jac_x = jax.jacfwd(add_scaled, argnums=0)(x, y)
    expected_jac = jnp.zeros((4, 4, 4, 4), dtype=x.dtype)
    for i in range(4):
      for j in range(4):
        expected_jac = expected_jac.at[i, j, i, j].set(2.0)
    np.testing.assert_allclose(jac_x, expected_jac)

  def test_fusible_jvp_symbolic_zeros_scan(self):
    @fuser.fusible
    def add_scaled(x_fn, y_fn, out_fn):
      x, y = x_fn(), y_fn()
      if out_fn is None:
        out_fn = lambda v: v
      return out_fn(x * 2.0 + y)

    x = jnp.ones((4, 4))
    y = jnp.ones((4, 4))

    # Forward-mode differentiation through jax.lax.scan:
    # scan pushes tangents through the body using JVP, passing ad.Zero for
    # constants like y.
    def scan_body(carry, _):
      return add_scaled(carry, y), None

    def scan_loss(init_x):
      final_carry, _ = jax.lax.scan(scan_body, init_x, None, length=2)
      return jnp.sum(final_carry)

    jac_x = jax.jacfwd(scan_loss)(x)
    np.testing.assert_allclose(jac_x, jnp.full_like(x, 4.0))

  def test_fusible_vjp_unused_output(self):
    @fuser.fusible
    def multi_out(x_fn, out_fn):
      x = x_fn()
      if out_fn is None:
        out_fn = lambda a, b: (a, b)
      return out_fn(x * 2.0, x * 3.0)

    def loss(x):
      y1, _ = multi_out(x)
      return jnp.sum(y1)

    x = jnp.ones((4, 4))
    grad_x = jax.grad(loss)(x)
    np.testing.assert_allclose(grad_x, jnp.full_like(x, 2.0))

  def test_fusible_closed_over_constants(self):
    c = jnp.array(3.0)

    @fuser.fuse
    def f(x):
      @fuser.fusible
      def inner(x_fn, out_fn):
        x = x_fn()
        if out_fn is None:
          out_fn = lambda v: v
        return out_fn(x * c)

      return inner(x)

    x = jnp.ones((4, 4), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_allclose(y, x * 3.0)


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

  def test_rebindable_inside_fusible(self):
    traces = []

    @functools.partial(rebindable_lib.rebindable, hyperparams=("bm", "bn"))
    def matmul_kernel(x_fn, y_fn, out_fn, *, bm, bn):
      traces.append((bm, bn, out_fn is not None))
      acc = jnp.dot(x_fn(), y_fn())
      return acc if out_fn is None else out_fn(acc)

    @fuser.fusible
    def matmul(x_fn, y_fn, out_fn):
      return matmul_kernel(x_fn, y_fn, out_fn, bm=16, bn=32)

    x = jnp.ones((32, 64), dtype=jnp.float32)
    y = jnp.ones((64, 32), dtype=jnp.float32)
    residual = jnp.full((32, 32), 2.0, dtype=jnp.float32)

    def step(a, b, res):
      @fuser.fuse
      def _fused(a_in, b_in):
        out = matmul(a_in * 2.0, b_in)
        return out + res, jax.nn.relu(out)
      return _fused(a, b)

    traced = jax.jit(step).trace(x, y, residual)
    [site] = rebindable_lib.extract_rebindables(traced)
    self.assertEqual(site.rebindable.hyperparams, {"bm": 16, "bn": 32})
    # The fused kernel's inputs are the leaves of its fusions: a, b and the
    # residual closed over by the epilogue.
    self.assertLen(site.rebindable.in_avals_flat, 3)

    traces.clear()
    rebound = rebindable_lib.rebind(traced, lambda t: dict(bm=32, bn=64))
    self.assertEqual(traces, [])  # rebinding is lazy
    out1, out2 = rebound.lower().compile()(x, y, residual)
    self.assertEqual(traces, [(32, 64, True)])  # one trace, with fusions
    expected = jnp.dot(x * 2.0, y)
    np.testing.assert_allclose(out1, expected + residual)
    np.testing.assert_allclose(out2, jax.nn.relu(expected))

  def test_rebindable_captures_fused_kernel_not_identity_kernel(self):
    """The tuned site must be the kernel with prologue/epilogue fused in.

    `fusible` first traces its body with trivial (identity) fusions; `fuse`
    then re-runs it with the real fusions. Only the latter may survive as the
    rebindable site, otherwise a tuner would benchmark a bare matmul.
    """
    traces = []

    @functools.partial(rebindable_lib.rebindable, hyperparams="bm")
    def matmul_kernel(x_fn, y_fn, out_fn, *, bm):
      traces.append((bm, out_fn is not None))
      acc = jnp.dot(x_fn(), y_fn())
      return acc if out_fn is None else out_fn(acc)

    matmul = fuser.fusible(
        lambda x_fn, y_fn, out_fn: matmul_kernel(x_fn, y_fn, out_fn, bm=16))

    @fuser.fuse
    def fused(a, b, res):
      out = matmul(jnp.tanh(a), b)
      return jnp.exp(out) + res, jnp.sin(out)

    x = jnp.full((32, 64), 0.01, jnp.float32)
    y = jnp.full((64, 32), 0.02, jnp.float32)
    res = jnp.full((32, 32), 3.0, jnp.float32)
    traced = jax.jit(fused).trace(x, y, res)

    # The identity-fusion trace happened, but it is not what got staged.
    self.assertIn((16, False), traces)
    [site] = rebindable_lib.extract_rebindables(traced)
    self.assertLen(site.rebindable.out_avals_flat, 2)  # both epilogue outputs
    self.assertLen(site.rebindable.in_avals_flat, 3)  # a, b and the residual

    def prims(jaxpr, into_rebindables):
      names = set()
      for eqn in jaxpr.eqns:
        prim = eqn.params.get("_prim")
        if isinstance(prim, rebindable_lib.Rebindable):
          if into_rebindables:
            names |= prims(prim.jaxpr, into_rebindables)
          continue
        names.add(eqn.primitive.name)
        for sub in jax_core.jaxprs_in_params(eqn.params):
          names |= prims(sub, into_rebindables)
      return names

    fused_ops = {"tanh", "dot_general", "exp", "add", "sin"}
    self.assertLessEqual(fused_ops, prims(site.rebindable.jaxpr, True))
    self.assertFalse(fused_ops & prims(traced.jaxpr, False),
                     "prologue/epilogue ops leaked outside the rebindable site")

    traces.clear()
    rebound = rebindable_lib.rebind(traced, lambda s: dict(bm=32))
    out1, out2 = rebound.lower().compile()(x, y, res)
    self.assertEqual(traces, [(32, True)])  # retraced with the real fusions
    acc = jnp.dot(jnp.tanh(x), y)
    np.testing.assert_allclose(out1, jnp.exp(acc) + res, rtol=1e-6)
    np.testing.assert_allclose(out2, jnp.sin(acc), rtol=1e-6)

  def test_rebindable_in_fused_kernel_under_shard_map(self):
    # The fused prologue/epilogue carry varying-axis types inside shard_map;
    # re-tracing the rebindable (physicalization, rebinding) must keep them.
    @functools.partial(rebindable_lib.rebindable, hyperparams="k")
    def kernel(x_fn, out_fn, *, k):
      y = x_fn() * k
      return y if out_fn is None else out_fn(y)

    scale = fuser.fusible(lambda x_fn, out_fn: kernel(x_fn, out_fn, k=2.0))
    mesh = jax.make_mesh((1,), ("data",))
    f = jax.jit(jax.shard_map(
        fuser.fuse(lambda x: jnp.sin(scale(x * 3.0))), mesh=mesh,
        in_specs=jax.P("data"), out_specs=jax.P("data")))
    x = jax.device_put(jnp.arange(8.0), jax.NamedSharding(mesh, jax.P("data")))
    traced = f.trace(x)
    [site] = rebindable_lib.extract_rebindables(traced)
    self.assertEqual(site.abstract_mesh.manual_axes, ("data",))
    rebound = rebindable_lib.rebind(traced, lambda s: dict(k=5.0))
    np.testing.assert_allclose(rebound.lower().compile()(x),
                               jnp.sin(jnp.arange(8.0) * 15.0), rtol=1e-6)

  def test_rebindable_inside_fusible_without_fuse(self):
    @functools.partial(rebindable_lib.rebindable, hyperparams="bm")
    def kernel(x_fn, out_fn, *, bm):
      del bm
      out = x_fn() * 3.0
      return out if out_fn is None else out_fn(out)

    f = fuser.fusible(lambda x_fn, out_fn: kernel(x_fn, out_fn, bm=8))
    x = jnp.ones((8,), dtype=jnp.float32)
    traced = jax.jit(f).trace(x)
    rebound = rebindable_lib.rebind(traced, lambda t: dict(bm=16))
    [site] = rebindable_lib.extract_rebindables(rebound)
    self.assertEqual(site.rebindable.hyperparams, {"bm": 16})
    np.testing.assert_allclose(rebound.lower().compile()(x), x * 3.0)

  def test_rebindable_body_is_physicalized(self):
    @dataclasses.dataclass(frozen=True)
    class PairDType(fusible_dtype.FusionDType):

      def __str__(self):
        return "pair"

      def abstract_unpack(self, x):
        return (x.update(dtype=jnp.float32), x.update(dtype=jnp.float32))

      def abstract_pack(self, x, y):
        return x.update(dtype=self)

      def pull_block_spec_one_step(self, aval_out, block_spec):
        return block_spec, block_spec

      def unpack_push_block_spec(self, aval_in, block_spec):
        return block_spec, block_spec

      def unpack_pull_block_spec(self, aval_in, block_spec1, block_spec2):
        return (block_spec1,)

    @functools.partial(rebindable_lib.rebindable, hyperparams="bm")
    def add_kernel(p_fn, out_fn, *, bm):
      del bm
      x, y = fusible_dtype.unpack(p_fn())
      return x + y if out_fn is None else out_fn(x + y)

    add = fuser.fusible(lambda p_fn, out_fn: add_kernel(p_fn, out_fn, bm=8))

    @jax.jit
    @fuser.fuse
    def f(x, y):
      return add(fusible_dtype.pack(x, y, dtype=PairDType())) * 2.0

    x, y = jnp.ones(8, jnp.float32), jnp.full(8, 2.0, jnp.float32)
    traced = f.trace(x, y)
    self.assertLen(rebindable_lib.extract_rebindables(traced), 1)
    rebound = rebindable_lib.rebind(traced, lambda s: dict(bm=16))
    np.testing.assert_allclose(rebound.lower().compile()(x, y), (x + y) * 2.0)


@jtu.with_config(jax_custom_vjp3=True)
class FusibleCustomVJP3Test(jtu.JaxTestCase):

  def test_fusible_custom_vjp3_grad(self):
    @jax.custom_vjp
    def scale(x):
      return x * 2.0

    def scale_fwd(x):
      return scale(x), x

    def scale_bwd(x, g):
      # Intentionally different from fwd to verify custom vjp rule is preserved
      # during expand().
      return (g * 3.0,)

    scale.defvjp(scale_fwd, scale_bwd)

    @fuser.fusible
    def f(x_fn, out_fn):
      x = x_fn()
      y = scale(x)
      if out_fn is None:
        out_fn = lambda v: v
      return out_fn(y)

    x = jnp.ones((4, 4), dtype=jnp.float32)
    loss = lambda v: jnp.sum(f(v))
    grad_x = jax.grad(loss)(x)
    np.testing.assert_allclose(grad_x, jnp.full_like(x, 3.0))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
