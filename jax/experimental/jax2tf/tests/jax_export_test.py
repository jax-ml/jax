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
import re
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
    def f(a):
      return jnp.concatenate([a, a], axis=0)

    exp = jax_export.export(f)(
        jax_export.poly_spec(a.shape, a.dtype, "(w, h)"))
    self.assertEqual("(w, h)", str(exp.in_avals[0].shape))
    self.assertEqual("(2*w, h)", str(exp.out_avals[0].shape))

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
        r"Shape mismatch for args\[0\] in dimension 0"):
      jax_export.call_exported(exp_f)(np.arange(6, dtype=np.float32), b=f32_4)

    with self.assertRaisesRegex(ValueError,
        r"Shape mismatch for kwargs\['b'\] in dimension 0"):
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
  # is called from an outer function, that is exported with outer_poly_spec.
  @parameterized.named_parameters(
      dict(testcase_name=f"inner={inner_poly_spec}_outer={outer_poly_spec}",
           inner_poly_spec=inner_poly_spec, outer_poly_spec=outer_poly_spec,
           expect_error=expect_error)
      for inner_poly_spec, outer_poly_spec, expect_error in (
          ("3,a,a+b", "3,4,12", None),
          ("3,a,a+b", "3,4,c", None),
          ("3,a,a+b", "3,c,c", r"Dimension variable.*b.*must have.* >= 1. Found value 0"),
          ("3,a,a+b", "c,4,12", r"Shape mismatch for args\[0\] in dimension 0"),
          ("3,a,a+b", "3,c+4,12", None),  # TODO: This should be an error, c = 0
          ("3,4,3*a", "3,4,12", None),
          ("3,4,5*a", "3,4,12", r"Dimension variable 'a' must have integer value >= 1. Found value 2.4"),
          # ("3,a,a", "3,a,a", None),  # TODO: wrong error. It should be shape mismatch
          # ("3,4,5*a", "3,4,c", None),  # TODO: wrong error. It should be "not divisible by 5"
      ))
  def test_poly(self, inner_poly_spec="3,a,a+b",
                outer_poly_spec="3,4,12", expect_error=None):
    # Polymorphic export called with static or polymorphic shapes
    def inner(x):  # x: export_poly_spec
      return jnp.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))

    x1 = np.arange(3 * 4 * 6, dtype=np.float32).reshape((3, 4, 6))  # x1 : f32[3,4,6]
    exp1 = jax_export.export(inner)(jax_export.poly_spec(x1.shape, x1.dtype, inner_poly_spec))

    x2 = np.concatenate([x1, x1], axis=2)  # x2: f32[3,4,12]
    def outer(x):  # x: call_poly_spec
      # Use an addition to test that the shapes are refined properly for the
      # result of the call_exported.
      return jax_export.call_exported(exp1)(x) + inner(x)

    with contextlib.ExitStack() as stack:
      if expect_error is not None:
        stack.push(self.assertRaisesRegex(ValueError, expect_error))

      # Call it after exporting again, with polymorphic shapes
      exp2 = jax_export.export(outer)(
          jax_export.poly_spec(x2.shape, x2.dtype, outer_poly_spec))
      # TODO: for now, we use XlaCallModule to run modules with polymorphic shapes
      # until we create the python bindings to invoke shape refinement.
      if jax2tf is not None:
        res2 = jax2tf._run_exported_as_tf([x2], exp2)[0].numpy()
        # res2 = jax_export.call_exported(exp2)(x2)
        self.assertAllClose(2. * inner(x2), res2)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
