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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import core as jax_core
from jax._src import hijax
from jax._src import test_util as jtu
from jax._src.pallas.fuser import fusible_dtype
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
import jax.numpy as jnp
from jax.sharding import AxisType, PartitionSpec as P
import numpy as np

jax.config.parse_flags_with_absl()
jtu.request_cpu_devices(4)


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
    with mock.patch.object(
        fusible_dtype, 'has_fusion_dtypes', return_value=True
    ):
      y = fusible_dtype.physicalize(f)(x)
    np.testing.assert_allclose(y, 2.0)

  @parameterized.named_parameters(
      {"testcase_name": f"cvjp3_{cvjp3}_remat3_{remat3}", "cvjp3": cvjp3, "remat3": remat3}
      for cvjp3 in (False, True)
      for remat3 in (False, True)
  )
  def test_fusible_physicalize_custom_vjp_and_remat(self, cvjp3, remat3):
    with (
        config.custom_vjp3(cvjp3),
        config.remat3(remat3),
        mock.patch.object(
            fusible_dtype, 'has_fusion_dtypes', return_value=True
        ),
    ):
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
    with (
        config.custom_vjp3(cvjp3),
        mock.patch.object(
            fusible_dtype, 'has_fusion_dtypes', return_value=True
        ),
    ):
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
    with (
        config.custom_vjp3(cvjp3),
        mock.patch.object(
            fusible_dtype, 'has_fusion_dtypes', return_value=True
        ),
    ):
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

  def test_fusion_dtype_op_registry_isolation(self):
    class DTypeA(fusible_dtype.FusionDType):
      abstract_unpack = None
      abstract_pack = None
      pull_block_spec_one_step = None
      unpack_push_block_spec = None
      unpack_pull_block_spec = None

    class DTypeB(fusible_dtype.FusionDType):
      abstract_unpack = None
      abstract_pack = None
      pull_block_spec_one_step = None
      unpack_push_block_spec = None
      unpack_pull_block_spec = None

    @DTypeA.register_op(lax.neg_p)
    def neg_rule(ctx, x):
      del ctx
      return x

    self.assertIsNotNone(neg_rule)
    self.assertIs(DTypeA.get_op_rule(lax.neg_p), neg_rule)
    self.assertIsNone(DTypeB.get_op_rule(lax.neg_p))
    self.assertIsNone(fusible_dtype.FusionDType.get_op_rule(lax.neg_p))

  def test_pack_unpack_dtype_abstract_eval_errors(self):
    x = jnp.ones((4, 4), dtype=jnp.float32)
    with self.assertRaisesRegex(
        ValueError, "Attempted to pack non-fusion dtype:.*float32"
    ):
      jax.jit(lambda x: fusible_dtype.pack(x, dtype=jnp.float32))(x)

    with self.assertRaisesRegex(
        ValueError, "Attempted to unpack non-fusion dtype: float32"
    ):
      jax.jit(fusible_dtype.unpack)(x)

  @parameterized.parameters(0, 1, 2)
  def test_fusible_vmap_batch_axis(self, in_axis):
    @fuser.fusible
    def scale(x_fn, out_fn):
      x = x_fn()
      if out_fn is None:
        out_fn = lambda v: v
      return out_fn(x * 2.0)

    shape = [1, 4, 8]
    shape.insert(in_axis, 1)
    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    y = jax.vmap(scale, in_axes=in_axis)(x)
    expected = jnp.squeeze(x, axis=in_axis)[None] * 2.0
    np.testing.assert_allclose(y, expected)

  def test_fuse_and_physicalize_fusion_dtype_fastpath(self):
    @dataclasses.dataclass(frozen=True)
    class PairDType(fusible_dtype.FusionDType):
      pull_block_spec_one_step = None
      unpack_push_block_spec = None
      unpack_pull_block_spec = None
      abstract_pack = lambda self, x, y: x.update(dtype=self)
      abstract_unpack = lambda self, x: (
          x.update(dtype=jnp.float32),
          x.update(dtype=jnp.float32),
      )

    @fuser.fusible
    def kernel(x_fn, out_fn):
      x = x_fn()
      if isinstance(x.dtype, PairDType):
        a, b = fusible_dtype.unpack(x)
        x = a + b
      return (out_fn or (lambda v: v))(x + 1.0)

    x = jnp.ones((4, 4), dtype=jnp.float32)
    with mock.patch.object(
        fusible_dtype,
        "physicalize_closed_jaxpr",
        wraps=fusible_dtype.physicalize_closed_jaxpr,
    ) as mock_phys:
      y = jax.jit(fuser.fuse(lambda x: kernel(x * 2.0) * 3.0))(x)
      np.testing.assert_allclose(y, (x * 2.0 + 1.0) * 3.0)
      mock_phys.assert_not_called()

      fused_pair = jax.jit(
          fuser.fuse(
              lambda x, y: kernel(fusible_dtype.pack(x, y, dtype=PairDType()))
          )
      )
      np.testing.assert_allclose(fused_pair(x, x), 2.0 * x + 1.0)
      mock_phys.assert_called_once()

    no_fusion_fn = lambda x: lax.scan(
        lambda c, _: (c + 1.0, None), x, None, length=2
    )[0]
    no_fusion_jaxpr = jax.make_jaxpr(no_fusion_fn)(x)
    self.assertIs(
        fusible_dtype.physicalize_closed_jaxpr(no_fusion_jaxpr), no_fusion_jaxpr
    )

    mixed_jaxpr = jax.make_jaxpr(
        lambda x, y: no_fusion_fn(
            sum(
                fusible_dtype.unpack(
                    fusible_dtype.pack(x, y, dtype=PairDType())
                )
            )
        )
    )(x, x)
    phys_jaxpr = fusible_dtype.physicalize_closed_jaxpr(mixed_jaxpr)
    self.assertFalse(fusible_dtype.has_fusion_dtypes(phys_jaxpr))
    self.assertIs(
        phys_jaxpr.eqns[-1].params["jaxpr"],
        mixed_jaxpr.eqns[-1].params["jaxpr"],
    )

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


class ShardMapFusionTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("Requires at least 4 devices")
    self.mesh = jtu.create_mesh(
        (2, 2), ("x", "y"), axis_types=(AxisType.Explicit,) * 2
    )

  @staticmethod
  @fuser.fusible
  def _id_fusible(x_fn, out_fn):
    x = x_fn()
    return x if out_fn is None else out_fn(x)

  def test_shard_map_all_four_regions(self):
    @fuser.fusible
    def add_fusible(x_fn, y_fn, out_fn):
      res = x_fn() + y_fn()
      return res if out_fn is None else out_fn(res)

    def fn(x, y, z, bias):
      iota = lax.broadcasted_iota(jnp.int32, x.shape, 0).astype(jnp.float32)
      x_pre = jnp.sin(x) * 2.0 + iota
      y_pre = (
          lax.squeeze(lax.slice(y, (1, 0, 0), (2, 8, 8)), (0,)) + bias[None, :]
      )

      @jax.shard_map(
          mesh=self.mesh,
          in_specs=(P("x", "y"), P("x", "y"), P("x", "y")),
          out_specs=P("x", "y"),
      )
      def shmapped(x_s, y_s, z_s):
        return (add_fusible(x_s + 1.0, jnp.cos(y_s)) - z_s) * 0.5

      return jnp.tanh(shmapped(x_pre, y_pre, z)) + x * 0.25 + bias[None, :]

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(64, dtype=jnp.float32).reshape(8, 8) / 64.0, P("x", "y")
      )
      y = jax.device_put(
          jnp.arange(192, dtype=jnp.float32).reshape(3, 8, 8) / 192.0,
          P(None, "x", "y"),
      )
      z = jax.device_put(jnp.ones((8, 8), dtype=jnp.float32) * 0.1, P("x", "y"))
      bias = jax.device_put(jnp.arange(8, dtype=jnp.float32), P("y"))
      np.testing.assert_allclose(
          jax.jit(fuser.fuse(fn))(x, y, z, bias),
          jax.jit(fn)(x, y, z, bias),
          rtol=1e-5,
          atol=1e-5,
      )

  def test_shard_map_with_tiled_pull_and_push_block_spec(self):
    @fuser.fusible
    def tiled_matmul(x_fn, y_fn, out_fn):
      has_out_fn = out_fn is not None
      if out_fn is None:
        out_fn = lambda x: (x,)
      m, k = x_fn.out_type.shape
      _, n = y_fn.out_type.shape
      bm, bk, bn = m // 2, k, n // 2
      grid = (2, 2)
      x_bs = pl.BlockSpec((bm, bk), lambda i, j: (i, 0))
      y_bs = pl.BlockSpec((bk, bn), lambda i, j: (0, j))
      z_bs = pl.BlockSpec((bm, bn), lambda i, j: (i, j))
      mm_out_type = jax.ShapeDtypeStruct((m, n), x_fn.out_type.dtype)
      x_fn2, x_vals, x_sp = fuser.get_fusion_values(x_fn)
      y_fn2, y_vals, y_sp = fuser.get_fusion_values(y_fn)
      out_fn2, out_vals, out_sp = fuser.get_fusion_values(out_fn, mm_out_type)
      sp_handler = fuser.make_scalar_prefetch_handler(0)
      x_kfn, (x_val_bs,), _ = fuser.pull_block_spec(
          x_fn2, x_bs, scalar_prefetch_handler=sp_handler, grid_len=len(grid)
      )(x_vals)
      y_kfn, (y_val_bs,), _ = fuser.pull_block_spec(
          y_fn2, y_bs, scalar_prefetch_handler=sp_handler, grid_len=len(grid)
      )(y_vals)
      out_kfn, out_val_bs, _, out_type, out_bs = fuser.push_pull_block_spec(
          out_fn2,
          z_bs,
          scalar_prefetch_handler=sp_handler,
          grid_len=len(grid),
      )(out_vals, mm_out_type)

      def kernel(x_val_refs, y_val_refs, out_val_refs, o_ref):
        pids = (pl.program_id(0), pl.program_id(1))
        x_t = x_kfn(pids, x_sp, jax.tree.map(lambda r: r[...], x_val_refs))
        y_t = y_kfn(pids, y_sp, jax.tree.map(lambda r: r[...], y_val_refs))
        (o_ref[...],) = out_kfn(
            pids,
            out_sp,
            jax.tree.map(lambda r: r[...], out_val_refs),
            x_t @ y_t,
        )

      res = pl.pallas_call(
          kernel,
          out_shape=out_type,
          grid=grid,
          in_specs=(x_val_bs, y_val_bs, out_val_bs),
          out_specs=out_bs,
          interpret=True,
      )(x_vals, y_vals, out_vals)
      return res if has_out_fn else res[0]

    def fn(x, y, z):
      iota = lax.broadcasted_iota(jnp.int32, x.shape, 0).astype(jnp.float32)
      x_pre = x * 2.0 + iota
      y_pre = lax.squeeze(lax.slice(y, (1, 0, 0), (2, 8, 16)), (0,))

      @jax.shard_map(
          mesh=self.mesh,
          in_specs=(P("x", None), P(None, "y"), P("x", "y")),
          out_specs=P("x", "y"),
          check_vma=False,
      )
      def shmapped(x_s, y_s, z_s):
        return tiled_matmul(x_s + 1.0, y_s) + z_s

      return shmapped(x_pre, y_pre, z) * 0.5 - z

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(128, dtype=jnp.float32).reshape(16, 8) / 128.0,
          P("x", None),
      )
      y = jax.device_put(
          jnp.arange(384, dtype=jnp.float32).reshape(3, 8, 16) / 384.0,
          P(None, None, "y"),
      )
      z = jax.device_put(
          jnp.ones((16, 16), dtype=jnp.float32) * 0.25, P("x", "y")
      )
      fused_fn = fuser.fuse(fn)
      fused_jaxpr = jax.make_jaxpr(fused_fn)(x, y, z).jaxpr
      self.assertLen(fused_jaxpr.eqns, 1)
      self.assertEqual(fused_jaxpr.eqns[0].primitive.name, "shard_map")
      inner_prims = [
          e.primitive.name for e in fused_jaxpr.eqns[0].params["jaxpr"].eqns
      ]
      self.assertEqual(inner_prims, ["pallas_call"])
      np.testing.assert_allclose(
          jax.jit(fused_fn)(x, y, z),
          jax.jit(fn)(x, y, z),
          rtol=1e-5,
          atol=1e-5,
      )

  def test_shard_map_replicated_outer_input_in_sharded_op(self):
    def fn(x, rep_scale, rep_shift):
      @jax.shard_map(
          mesh=self.mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
      )
      def shmapped(x_s):
        return self._id_fusible(x_s + 1.0)

      return shmapped(x * rep_scale) + rep_shift, rep_scale

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(64, dtype=jnp.float32).reshape(8, 8), P("x", "y")
      )
      rep_scale = jax.device_put(
          jnp.full((8, 8), 2.0, dtype=jnp.float32), P(None, None)
      )
      rep_shift = jax.device_put(
          jnp.full((8, 8), 3.0, dtype=jnp.float32), P(None, None)
      )
      exp_out, exp_bypass = jax.jit(fn)(x, rep_scale, rep_shift)
      act_out, act_bypass = jax.jit(fuser.fuse(fn))(x, rep_scale, rep_shift)
      np.testing.assert_allclose(act_out, exp_out)
      np.testing.assert_allclose(act_bypass, exp_bypass)

  def test_shard_map_separate_output_fusions_and_permutation(self):
    @fuser.fusible(output_fusion_prefix=(True, True))
    def multi_fusible(x_fn, y_fn, out_fns):
      fn1, fn2 = out_fns
      return (x_fn() + 1.0 if fn1 is None else fn1(x_fn() + 1.0)), (
          y_fn() + 2.0 if fn2 is None else fn2(y_fn() + 2.0)
      )

    def fn(x, y):
      @jax.shard_map(
          mesh=self.mesh,
          in_specs=(P("x", None), P(None, "y")),
          out_specs=(P("x", None), P(None, "y")),
      )
      def shmapped(x_s, y_s):
        a, b = multi_fusible(x_s, y_s)
        return a * 2.0, b * 5.0

      a_out, b_out = shmapped(x * 3.0, y * 4.0)
      return b_out - 1.0, (a_out + 10.0, a_out * 0.5)

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(64, dtype=jnp.float32).reshape(8, 8), P("x", None)
      )
      y = jax.device_put(
          jnp.arange(64, dtype=jnp.float32).reshape(8, 8), P(None, "y")
      )
      b_exp, (a1_exp, a2_exp) = jax.jit(fn)(x, y)
      b_act, (a1_act, a2_act) = jax.jit(fuser.fuse(fn))(x, y)
      np.testing.assert_allclose(b_act, b_exp)
      np.testing.assert_allclose(a1_act, a1_exp)
      np.testing.assert_allclose(a2_act, a2_exp)

  def test_shard_map_reshape_and_transpose_outer_ops(self):
    def fn(x):
      x_r = jnp.reshape(
          jnp.swapaxes(x, 1, 2), (32, 8), out_sharding=P("x", "y")
      )

      @jax.shard_map(
          mesh=self.mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
      )
      def shmapped(x_s):
        return self._id_fusible(x_s + 1.0) * 2.0

      return jnp.swapaxes(shmapped(x_r), 0, 1)

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(256, dtype=jnp.float32).reshape(4, 8, 8), P("x", "y", None)
      )
      np.testing.assert_allclose(jax.jit(fuser.fuse(fn))(x), jax.jit(fn)(x))

  def test_shard_map_compound_axes_and_independent_outer_output(self):
    def fn(x):
      x_pre = x * 2.0 + 1.0

      @jax.shard_map(
          mesh=self.mesh,
          in_specs=P(("x", "y"), None),
          out_specs=P(("x", "y"), None),
      )
      def shmapped(x_s):
        return self._id_fusible(x_s + 3.0)

      return shmapped(x_pre) * 0.5, x_pre + 5.0

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(64, dtype=jnp.float32).reshape(8, 8), P(("x", "y"), None)
      )
      exp_shmap, exp_indep = jax.jit(fn)(x)
      act_shmap, act_indep = jax.jit(fuser.fuse(fn))(x)
      np.testing.assert_allclose(act_shmap, exp_shmap)
      np.testing.assert_allclose(act_indep, exp_indep)

  def test_shard_map_multiple_fusibles_raises(self):
    @jax.jit
    @fuser.fuse
    def fn_two_in_shmap(x):
      @jax.shard_map(
          mesh=self.mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
      )
      def shmapped(x_s):
        return self._id_fusible(self._id_fusible(x_s))

      return shmapped(x)

    with jax.set_mesh(self.mesh):
      x = jax.device_put(jnp.ones((8, 8), dtype=jnp.float32), P("x", "y"))
      with self.assertRaisesRegex(ValueError, "Multiple fusible equations"):
        fn_two_in_shmap(x)

  def test_shard_map_dynamic_slice_scalar_prefetch_and_literals(self):
    def fn(x, idx):
      zero = jnp.int32(0)
      x_squeezed = lax.squeeze(
          lax.dynamic_slice(x, (idx, zero, zero), (1, 8, 8)), (0,)
      )

      @jax.shard_map(
          mesh=self.mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
      )
      def shmapped(x_s):
        return self._id_fusible(x_s * 2.0) + 1.0

      return shmapped(x_squeezed), jnp.int32(42)

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(192, dtype=jnp.float32).reshape(3, 8, 8), P(None, "x", "y")
      )
      idx = jax.device_put(jnp.int32(2), P())
      exp_out, exp_lit = jax.jit(fn)(x, idx)
      act_out, act_lit = jax.jit(fuser.fuse(fn))(x, idx)
      np.testing.assert_allclose(act_out, exp_out)
      np.testing.assert_array_equal(act_lit, exp_lit)

  def test_shard_map_with_fusion_dtype_physicalization(self):
    class PairFusionDType(fusible_dtype.FusionDType):

      def __str__(self):
        return "pair_fusion_dtype"

      def abstract_unpack(self, x):
        return (x.update(dtype=jnp.float32), x.update(dtype=jnp.float32))

      def abstract_pack(self, x, y):
        del y
        return x.update(dtype=self)

      def pull_block_spec_one_step(self, aval_out, block_spec):
        del aval_out
        return block_spec, block_spec

      def unpack_push_block_spec(self, aval_in, block_spec):
        del aval_in
        return block_spec, block_spec

      def unpack_pull_block_spec(self, aval_in, block_spec1, block_spec2):
        del aval_in, block_spec2
        return (block_spec1,)

      def pack_eval_rule(self, ctx, x, y):
        del ctx
        return fusible_dtype.pack(x, y, dtype=self)

    @fuser.fusible
    def inner_fusible(x_fn, out_fn):
      has_out_fn = out_fn is not None
      if out_fn is None:
        out_fn = lambda x: (x,)
      bs = pl.BlockSpec(x_fn.out_type.shape, lambda i: (i, 0))
      out_type = jax.ShapeDtypeStruct(x_fn.out_type.shape, jnp.float32)
      x_fn2, x_vals, x_sp = fuser.get_fusion_values(x_fn)
      out_fn2, out_vals, out_sp = fuser.get_fusion_values(out_fn, out_type)
      sp_handler = fuser.make_scalar_prefetch_handler(0)
      x_kfn, (x_val_bs,), _ = fuser.pull_block_spec(
          x_fn2, bs, scalar_prefetch_handler=sp_handler, grid_len=1
      )(x_vals)
      out_kfn, out_val_bs, _, z_out_type, out_bs = fuser.push_pull_block_spec(
          out_fn2, bs, scalar_prefetch_handler=sp_handler, grid_len=1
      )(out_vals, out_type)

      def kernel(x_val_refs, out_val_refs, o_ref):
        pids = (pl.program_id(0),)
        u, v = fusible_dtype.unpack(
            x_kfn(pids, x_sp, jax.tree.map(lambda r: r[...], x_val_refs))
        )
        (o_ref[...],) = out_kfn(
            pids, out_sp, jax.tree.map(lambda r: r[...], out_val_refs), u + v
        )

      res = pl.pallas_call(
          kernel,
          out_shape=z_out_type,
          grid=(1,),
          in_specs=(x_val_bs, out_val_bs),
          out_specs=out_bs,
          interpret=True,
      )(x_vals, out_vals)
      return res if has_out_fn else res[0]

    @jax.jit
    @fuser.fuse(resolve_fusion_dtypes=True)
    def fn(x, y):
      packed = fusible_dtype.pack(x * 2.0, y + 1.0, dtype=PairFusionDType())

      @jax.shard_map(
          mesh=self.mesh,
          in_specs=P("x", "y"),
          out_specs=P("x", "y"),
          check_vma=False,
      )
      def shmapped(p_s):
        return inner_fusible(p_s)

      return shmapped(packed) * 3.0

    @jax.jit
    @fusible_dtype.physicalize
    def direct_shmap(x, y):
      p = fusible_dtype.pack(x, y, dtype=PairFusionDType())
      p_out = jax.shard_map(
          lambda p_s: p_s,
          mesh=self.mesh,
          in_specs=P("x", "y"),
          out_specs=P("x", "y"),
      )(p)
      u, v = fusible_dtype.unpack(p_out)
      return u - v

    with jax.set_mesh(self.mesh):
      x = jax.device_put(
          jnp.arange(64, dtype=jnp.float32).reshape(8, 8), P("x", "y")
      )
      y = jax.device_put(jnp.ones((8, 8), dtype=jnp.float32), P("x", "y"))
      fused_jaxpr = jax.make_jaxpr(fn)(x, y).jaxpr.eqns[0].params["jaxpr"]
      self.assertLen(fused_jaxpr.eqns, 1)
      self.assertEqual(fused_jaxpr.eqns[0].primitive.name, "shard_map")
      inner_prims = [
          e.primitive.name for e in fused_jaxpr.eqns[0].params["jaxpr"].eqns
      ]
      self.assertEqual(inner_prims, ["pallas_call"])
      np.testing.assert_allclose(fn(x, y), (x * 2.0 + (y + 1.0)) * 3.0)
      np.testing.assert_allclose(direct_shmap(x, y), x - y)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
