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
import contextlib
import unittest

from absl.testing import absltest, parameterized

import jax
from jax import tree_util

from jax import numpy as jnp
from jax.config import config
from jax.experimental.jax2tf import jax_export
try:
  from jax.experimental.jax2tf import jax2tf  # TODO: temporary
except ImportError:
  jax2tf = None  # type: ignore

from jax._src import core
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb

import numpy as np


config.parse_flags_with_absl()


class JaxExportTest(jtu.JaxTestCase):

  def test_basic_export_only(self):
    def my_fun(x):
      return jnp.sin(x)
    exp = jax_export.export(my_fun)(jax.ShapeDtypeStruct((4,), dtype=np.float32))
    self.assertEqual("my_fun", exp.fun_name)
    self.assertEqual(jax_export.default_lowering_platform(), exp.lowering_platform)
    self.assertEqual(tree_util.tree_flatten(((1,), {}))[1], exp.in_tree)
    self.assertEqual((core.ShapedArray((4,), dtype=np.float32),), exp.in_avals)
    self.assertEqual((core.ShapedArray((4,), dtype=np.float32),), exp.out_avals)

  def test_pytree_export_only(self):
    a = np.arange(4, dtype=np.float32)
    b = np.arange(6, dtype=np.float32)
    def f(a_b_pair, *, a, b):
      return (dict(res=a_b_pair, a=a, b=b), jnp.sin(a), jnp.cos(b))

    exp = jax_export.export(f, lowering_platform="cpu")((a, b), a=a, b=b)
    a_aval = core.ShapedArray(a.shape, a.dtype)
    b_aval = core.ShapedArray(b.shape, b.dtype)
    self.assertEqual(exp.lowering_platform, "cpu")
    args = ((a, b),)
    kwargs = dict(a=a, b=b)
    self.assertEqual(exp.in_tree, tree_util.tree_flatten((args, kwargs))[1])
    self.assertEqual(exp.in_avals, (a_aval, b_aval, a_aval, b_aval))
    self.assertEqual(exp.out_tree, tree_util.tree_flatten(f(*args, **kwargs))[1])
    self.assertEqual(exp.out_avals, (a_aval, b_aval, a_aval, b_aval, a_aval, b_aval))

  def test_poly_export_only(self):
    a = np.arange(12, dtype=np.float32).reshape((3, 4))
    def f(a, b):  # a: f32[2w,h]  b: f32[w,h]
      return jnp.concatenate([a, b], axis=0)

    exp = jax_export.export(f)(
        jax_export.poly_spec(a.shape, a.dtype, "(2*w, h)"),
        jax_export.poly_spec(a.shape, a.dtype, "(w, h)"))
    self.assertEqual("(2*w, h)", str(exp.in_avals[0].shape))
    self.assertEqual("(w, h)", str(exp.in_avals[1].shape))
    self.assertEqual("(3*w, h)", str(exp.out_avals[0].shape))

  def test_poly_pytree_export_only(self):
    a = np.arange(12, dtype=np.float32).reshape((3, 4))
    def f(a0, a1, *, ak):
      return jnp.concatenate([a0, a1, ak], axis=0)

    a_poly_spec = jax_export.poly_spec(a.shape, a.dtype, "(w, h)")
    exp = jax_export.export(f)(a_poly_spec, a_poly_spec, ak=a_poly_spec)
    self.assertEqual("(w, h)", str(exp.in_avals[0].shape))
    self.assertEqual("(3*w, h)", str(exp.out_avals[0].shape))

  def test_basic(self):
    f = jnp.sin
    x = np.arange(4, dtype=np.float32)
    exp_f = jax_export.export(f)(x)

    f1 = jax_export.call_exported(exp_f)
    self.assertAllClose(f(x), f1(x))

  def test_call_exported_lambda(self):
    # When we export a lambda, the exported.fun_name is not a valid MLIR function name
    f = lambda x: jnp.sin(x)
    x = np.arange(4, dtype=np.float32)
    exp_f = jax_export.export(f)(x)
    f1 = jax_export.call_exported(exp_f)
    self.assertAllClose(f(x), f1(x))

  def test_call_twice_exported(self):
    def f(x): return jnp.sin(x)
    x = np.arange(4, dtype=np.float32)

    @jax.jit
    def f1(x):
      exp_f = jax_export.export(f)(x)
      return jax_export.call_exported(exp_f)(x) + jax_export.call_exported(exp_f)(x)

    self.assertAllClose(2. * f(x), f1(x))

  def test_unused_args(self):
    f = lambda x, y: jnp.sin(x)
    x = np.arange(4, dtype=np.float32)
    y = np.arange(6, dtype=np.float32)
    exp_f = jax_export.export(f)(x, y)

    f1 = jax_export.call_exported(exp_f)
    self.assertAllClose(f(x, y), f1(x, y))

  def test_pytree(self):
    a = np.arange(4, dtype=np.float32)
    b = np.arange(6, dtype=np.float32)
    def f(a_b_pair, a, b):
      return (dict(res=a_b_pair, a=a, b=b), jnp.sin(a), jnp.cos(b))

    exp_f = jax_export.export(f)((a, b), a=a, b=b)
    f1 = jax_export.call_exported(exp_f)
    self.assertAllClose(f((a, b), a=a, b=b),
                        f1((a, b), a=a, b=b))

  def test_error_wrong_intree(self):
    def f(a_b_pair, *, c):
      return jnp.sin(a_b_pair[0]) + jnp.cos(a_b_pair[1]) + c
    a = b = c = np.arange(4, dtype=np.float32)
    exp_f = jax_export.export(f)((a, b), c=c)

    with self.assertRaisesRegex(
        ValueError,
        "The invocation args and kwargs must have the same pytree structure"):
      jax_export.call_exported(exp_f)(a, b, c=(a, b))

  def test_error_wrong_avals(self):
    def f(a, *, b):  # a: f32[4] and b: f32[4]
      return jnp.sin(a) + jnp.cos(b)
    f32_4 = np.arange(4, dtype=np.float32)
    exp_f = jax_export.export(f)(f32_4, b=f32_4)

    with self.assertRaisesRegex(ValueError,
        r"Shape mismatch for args\[0\].shape\[0\]"):
      jax_export.call_exported(exp_f)(np.arange(6, dtype=np.float32), b=f32_4)

    with self.assertRaisesRegex(ValueError,
        r"Shape mismatch for kwargs\['b'\].shape\[0\]"):
      jax_export.call_exported(exp_f)(f32_4, b=np.arange(6, dtype=np.float32))

    with self.assertRaisesRegex(ValueError,
        r"Rank mismatch for args\[0\]"):
      jax_export.call_exported(exp_f)(f32_4.reshape((1, 4)), b=f32_4)

    with self.assertRaisesRegex(ValueError,
        r"Dtype mismatch for args\[0\]"):
      jax_export.call_exported(exp_f)(f32_4.astype(np.float16), b=f32_4)

  @parameterized.named_parameters(
      dict(testcase_name=p, platform=p)
      for p in ("cpu", "cuda", "rocm", "tpu"))
  def test_error_wrong_platform(self, platform):
    a = np.arange(4, dtype=np.float32)

    exp_f = jax_export.export(jnp.sin, lowering_platform=platform)(a)
    if xb.canonicalize_platform(jtu.device_under_test()) == platform:
      raise unittest.SkipTest("")

    with self.assertRaisesRegex(
        ValueError, "The exported function .* was lowered for platform"):
      jax_export.call_exported(exp_f)(a)

  def test_grad(self):
    f = lambda x: jnp.sum(jnp.sin(x))
    x = np.arange(4, dtype=np.float32)
    exp_f = jax_export.export(f)(x)

    f1 = jax_export.call_exported(exp_f)
    self.assertAllClose(jax.grad(f)(x), jax.grad(f1)(x))

  def test_pytree_vjp(self):
    def f(a_b_pair, *, a, b):
      return (dict(res=a_b_pair, a=2. * a, b=3. * b),
              jnp.sin(4. * a))

    a = np.arange(4, dtype=np.float32)
    b = np.arange(6, dtype=np.float32)
    exp_f = jax_export.export(f)((a, b), a=a, b=b)

    out_ct = f((a, b), a=a, b=b)  # The output has the right structure as the cotangent
    def f1_jax(a, b):  # For VJP, make a function without kwargs
      res = f((a, b), a=a, b=b)
      return res
    def f1_exp(a, b):  # For VJP, make a function without kwargs
      res = jax_export.call_exported(exp_f)((a, b), a=a, b=b)
      return res
    jax_vjp = jax.vjp(f1_jax, a, b)[1](out_ct)
    exp_vjp = jax.vjp(f1_exp, a, b)[1](out_ct)
    self.assertAllClose(jax_vjp, exp_vjp)

  def test_roundtrip(self):
    def f1(x):
      return jnp.sin(x)
    a = np.arange(4, dtype=np.float32)
    exp_f1 = jax_export.export(f1)(a)
    def f2(x):
      res1 = jax_export.call_exported(exp_f1)(x)
      res2 = jax_export.call_exported(exp_f1)(res1)
      return jnp.cos(res2)
    exp_f2 = jax_export.export(f2)(a)

    self.assertAllClose(jnp.cos(jnp.sin(jnp.sin(a))),
                        jax_export.call_exported(exp_f2)(a))

  # An inner function is exported with polymorphic shapes inner_poly_spec, and
  # is called from an outer function, which is exported with outer_poly_spec.
  @parameterized.named_parameters(
      dict(testcase_name=f"inner={d['inner_poly_spec']}_outer={d['outer_poly_spec']}",  # type: ignore
           **d)  # type: ignore
      for d in (
          dict(inner_poly_spec="3,a,a+b", outer_poly_spec="3,4,12"),
          dict(inner_poly_spec="3,a,a+b", outer_poly_spec="3,4,c"),
          dict(inner_poly_spec="3,a,a+b", outer_poly_spec="3,c,c",
               expect_error=(
                   r"Dimension variable 'b' must have integer value >= 1. "
                   r"Found 0 when solving a \+ b == args\[0\].shape\[2\]")),
          dict(inner_poly_spec="3,a,a+b", outer_poly_spec="c,4,12",
               expect_error=r"Shape mismatch for args\[0\].shape\[0\] \(expected constant\)"),
          dict(inner_poly_spec="3,a,a+b", outer_poly_spec="3,c+4,12"),  # TODO: This should be an error, c = 0
          dict(inner_poly_spec="3,4,3*a", outer_poly_spec="3,4,12"),
          dict(inner_poly_spec="3,4,5*a", outer_poly_spec="3,4,12",
               expect_error=(
                   r"Dimension variable 'a' must have integer value >= 1. "
                   r"Non-zero remainder 2 for factor 5 when solving 5\*a == args\[0\].shape\[2\]")),
          # dict(inner_poly_spec="3,4,5*a", outer_poly_spec="3,4,c"),  # TODO: there should be an error 5*a != c == 12
          # dict(inner_poly_spec="3,a,a", outer_poly_spec="3,a,a"),  # TODO: this should be a dynamic error
          dict(inner_poly_spec="3,a", inner_x_shape=(3, 4), outer_poly_spec="3,a,a",
               expect_error=r"Rank mismatch for args\[0\]"),
          dict(inner_poly_spec="3,a,a+b", inner_x_dtype=np.int32, outer_poly_spec="3,c,d",
               expect_error=r"Dtype mismatch for args\[0\]"),
      ))
  def test_poly(self, inner_poly_spec="3,a,a+b", inner_x_shape=(3, 4, 6),
                inner_x_dtype=np.float32,
                outer_poly_spec="3,c+4,12",  outer_x_shape=(3, 4, 12),
                expect_error=None):
    # Polymorphic export called with static or polymorphic shapes
    def inner(x):  # x: inner_poly_spec
      return jnp.reshape(x, (-1, x.shape[1]))

    inner_x = np.arange(np.prod(inner_x_shape),
                        dtype=inner_x_dtype).reshape(inner_x_shape)  # inner_x : f32[3,4,6]
    inner_exp = jax_export.export(inner)(
        jax_export.poly_spec(inner_x.shape, inner_x.dtype, inner_poly_spec))

    outer_x = np.arange(np.prod(outer_x_shape),
                        dtype=np.float32).reshape(outer_x_shape)  # outer_x : f32[3,4,12]
    def outer(x):  # x: outer_poly_spec
      # Use an addition to test that the shapes are refined properly for the
      # result of the call_exported.
      return jax_export.call_exported(inner_exp)(x) + inner(x)

    with contextlib.ExitStack() as stack:
      if expect_error is not None:
        stack.push(self.assertRaisesRegex(ValueError, expect_error))

      # Call it after exporting again, with polymorphic shapes
      outer_exp = jax_export.export(outer)(
          jax_export.poly_spec(outer_x.shape, outer_x.dtype, outer_poly_spec))
      # TODO: for now, we use XlaCallModule to run modules with polymorphic shapes
      # until we create the python bindings to invoke shape refinement.
      if jax2tf is not None:
        res2 = jax2tf._run_exported_as_tf([outer_x], outer_exp)[0].numpy()
        # res2 = jax_export.call_exported(exp2)(x2)
        self.assertAllClose(2. * inner(outer_x), res2)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
