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
import logging
import math
import re
from typing import List, Optional, Sequence
import unittest

from absl.testing import absltest
import jax
from jax import numpy as jnp
from jax import tree_util
from jax.config import config
from jax.experimental.jax2tf import jax_export
from jax.lib import xla_client as xc

from jax._src import core
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir

from jax._src.lib.mlir.dialects import hlo
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib import xla_extension

import numpy as np

config.parse_flags_with_absl()


if xc.mlir_api_version >= 50:
  def _temp_call_exported(exp: jax_export.Exported, *args: jax.Array,
                          skip_shape_check: bool = False):
    assert all(core.is_constant_shape(a.shape) for a in args)
    return jax_export.call_exported(exp)(*args)
else:
  def _temp_call_exported(exp: jax_export.Exported, *args: jax.Array,
                          skip_shape_check: bool = False):
    """Temporary runner for an Exported.

    Normally we would use jax_export.call_exported, but if the exported has
    shape polymorphism and we are using jaxlib before 0.4.12 we use
    Once we upgrade the jaxlib we can replace all uses of this function with
    jax_export.call_exported.
    """
    assert all(core.is_constant_shape(a.shape) for a in args)
    if not exp.module_uses_dim_vars:
      return jax_export.call_exported(exp)(*args)
    else:
      # We only get here in external tests, because internal ones use newest jaxlib
      from jax.experimental.jax2tf import jax2tf  # TODO: temporary
      # call_exported does the shape checking, we must do it manually for
      # XlaCallModule.
      if not skip_shape_check:
        shape_check = exp.shape_check_module()
        if shape_check is not None:
          err_msg = _run_shape_check_module(exp, shape_check[0], shape_check[1], args)
          if err_msg is not None:
            raise ValueError(err_msg)

      numpy_results = map(lambda res_tf: res_tf.numpy(),
                          jax2tf._run_exported_as_tf(args, exp))
      return exp.out_tree.unflatten(numpy_results)

def _run_shape_check_module(primal_exported: jax_export.Exported,
                            shape_check_module_serialized: bytes,
                            shape_check_messages: Sequence[str],
                            args: Sequence[jax.Array]
                            ) -> Optional[str]:
  """Helper to run a shape checking module.

  We only need to do this in tests, because otherwise this will be done
  implicitly when we call XlaCallModule.

  Returns: the error message, or None if no error.
  """
  args = tuple(args)
  # We cannot just make an Exported and run it, because call_exported will
  # do the shape checks statically. So, we wrap the shape check module
  # with static shape arguments

  static_in_avals = tuple(core.get_aval(a) for a in args)
  context = mlir.make_ir_context()
  with context, ir.Location.unknown(context):
    wrapped_module = ir.Module.parse(
        xla_extension.mlir.deserialize_portable_artifact(
            shape_check_module_serialized))
    symbol_table = ir.SymbolTable(wrapped_module.operation)
    orig_main = symbol_table["main"]
    orig_main.attributes["sym_visibility"] = ir.StringAttr.get("private")
    symbol_table.set_symbol_name(orig_main, "_wrapped_jax_export_main")
    orig_main_name = ir.StringAttr(symbol_table.insert(orig_main)).value
    # Use static shapes
    new_main_input_types = [mlir.aval_to_ir_type(a) for a in static_in_avals]
    orig_output_types = orig_main.type.results
    new_main_ftype = ir.FunctionType.get(
      new_main_input_types, orig_output_types
    )
    new_main_op = func_dialect.FuncOp(
      "main",
      new_main_ftype,
      ip=ir.InsertionPoint.at_block_begin(wrapped_module.body),
    )
    new_main_op.attributes["sym_visibility"] = ir.StringAttr.get("public")
    symbol_table.insert(new_main_op)
    entry_block = new_main_op.add_entry_block()
    with ir.InsertionPoint(entry_block):
      orig_main_args: List[ir.Value] = []
      for new_arg, orig_arg_type in zip(
        new_main_op.arguments, orig_main.type.inputs
      ):
        orig_main_args.append(hlo.ConvertOp(orig_arg_type, new_arg).result)
      call = func_dialect.CallOp(
        orig_output_types,
        ir.FlatSymbolRefAttr.get(orig_main_name),
        orig_main_args,
      )
      func_dialect.ReturnOp(call.results)
  symbol_table.set_symbol_name(new_main_op, "main")

  wrapped_module_serialized, version = jax_export._serialize_module(wrapped_module)
  # Make an Exported and then run it
  out_avals = (core.ShapedArray((), dtype=np.int32),) * 3
  exp = jax_export.Exported(
    fun_name=f"shape_check_{primal_exported.fun_name}",
    in_tree=tree_util.tree_flatten((args, {}))[1],
    in_avals=static_in_avals,
    out_tree=tree_util.tree_flatten(out_avals)[1],
    out_avals=out_avals,
    in_shardings=None,
    out_shardings=None,
    lowering_platform=primal_exported.lowering_platform,
    disabled_checks=(),
    mlir_module_serialized=wrapped_module_serialized,
    xla_call_module_version=version,
    module_kept_var_idx=tuple(sorted(range(len(static_in_avals)))),
    module_uses_dim_vars=True,
    _get_vjp=lambda _: None)  # type: ignore

  code, op1, op2 = _temp_call_exported(exp, *args,
                                       skip_shape_check=True)
  if code == -1:
    return None
  else:
    return shape_check_messages[code].replace("%1", str(int(op1))).replace(
      "%2", str(int(op2)))


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

  @jtu.parameterized_filterable(
    testcase_name=lambda kw: kw["platform"],
    kwargs=[dict(platform=p)
            for p in ("cpu", "cuda", "rocm", "tpu")])
  def test_error_wrong_platform(self, platform):
    a = np.arange(4, dtype=np.float32)

    exp_f = jax_export.export(jnp.sin, lowering_platform=platform)(a)
    if xb.canonicalize_platform(jtu.device_under_test()) == platform:
      raise unittest.SkipTest("")

    with self.assertRaisesRegex(
        ValueError, "The exported function .* was lowered for platform"):
      jax_export.call_exported(exp_f)(a)

    # Now try with the platform check disabled
    exp_f_no_platform_check = jax_export.export(
      jnp.sin, lowering_platform=platform,
      disabled_checks=[jax_export.DisabledSafetyCheck.platform()])(a)
    res = jax_export.call_exported(exp_f_no_platform_check)(a)
    self.assertAllClose(res, jnp.sin(a))

  def test_error_disallowed_custom_call(self):
    if jtu.device_under_test() != "cpu":
      self.skipTest("Test intended for CPU only")
    # For now triangular_solve on CPU uses the unsupported "blas_strsm" target
    a = np.arange(16, dtype=np.float32).reshape((4, 4))
    b = np.arange(4, dtype=np.float32).reshape((4, 1))
    with self.assertRaisesRegex(ValueError,
        "Cannot serialize code with custom calls whose targets .*"):
      jax_export.export(
        lambda a, b: jax.lax.linalg.triangular_solve(a, b, left_side=True),
      )(a, b)

    # Now try again with the safety check disabled
    exp = jax_export.export(
        lambda a, b: jax.lax.linalg.triangular_solve(a, b, left_side=True),
        disabled_checks=(jax_export.DisabledSafetyCheck.custom_call("blas_strsm"),)
      )(a, b)
    self.assertIn("blas_strsm", exp.mlir_module)

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

  # A function is exported with f32[poly_spec] and is called with different arg
  # shapes. We use jax_export.call_exported and we also run the shape check
  # module.
  @jtu.parameterized_filterable(
    testcase_name=lambda kw:f"poly_spec={kw['poly_spec']}_arg_shape={kw['arg_shape']}",  # type: ignore
    #one_containing="",
    kwargs=[
      dict(poly_spec="3,4,12", arg_shape=(3, 4, 12)),
      dict(poly_spec="3,4,12", arg_shape=(3, 4, 13),
           # The shape check module does not test constant dimensions
           expect_error_run=re.escape(
               r"Shape mismatch for args[0].shape[2] (expected same constant)")),
      dict(poly_spec="3,4,6*a", arg_shape=(3, 4, 12)),
      dict(poly_spec="3,a,a+8", arg_shape=(3, 4, 12)),
      dict(poly_spec="3,4,a+1", arg_shape=(3, 4, 1),
           expect_error=re.escape(
              r"Dimension variable 'a' must have integer "
              r"value >= 1. Found 0 when solving "
              r"a + 1 == args[0].shape[2].")),
      dict(poly_spec="3,4,6*a", arg_shape=(3, 4, 13),
           expect_error=re.escape(
               r"Dimension variable 'a' must have integer value >= 1. "
               r"Non-zero remainder 1 for factor 6 when solving "
               r"6*a == args[0].shape[2]")),
      dict(poly_spec="3,a,a+8", arg_shape=(3, 4, 13),
           expect_error=re.escape(
              r"Found inconsistency 13 != 12 when solving "
              r"a + 8 == args[0].shape[2]")),
  ])
  def test_poly_shape_checks(
      self, poly_spec="3,a,a+8",
      arg_shape=(3, 4, 12), arg_dtype=np.float32,
      expect_error=None,  # If given, applies for expect_error_run and expect_error_shape_check
      expect_error_run=None,  # Error from running the exported module
      expect_error_shape_check=None):  # Error from running the shape check module
    if expect_error is not None:
      self.assertIsNone(expect_error_run, None)
      self.assertIsNone(expect_error_shape_check, None)
      expect_error_run = expect_error_shape_check = expect_error
    def f(x):  # x: f32[poly_spec]
      return jnp.reshape(x, (-1, x.shape[1]))

    exp_f = jax_export.export(f)(
        jax_export.poly_spec((3, 4, 12), np.float32, poly_spec))
    self.assertEqual(exp_f.module_uses_dim_vars, poly_spec != "3,4,12")
    arg = np.arange(np.prod(arg_shape),
                    dtype=arg_dtype).reshape(arg_shape)  # arg : f32[3,4,12]

    with contextlib.ExitStack() as stack:
      if expect_error_run is not None:
        stack.push(self.assertRaisesRegex(Exception, expect_error_run))

      res = _temp_call_exported(exp_f, arg)

    if not expect_error_run:
      self.assertAllClose(res, f(arg))

    # Test the shape_check_module
    shape_check = exp_f.shape_check_module()
    # We have shape_check only if the exported has polymorphic inputs shapes.
    if all(core.is_constant_shape(a.shape) for a in exp_f.in_avals):
      self.assertIsNone(shape_check)
      self.assertIsNone(expect_error_shape_check)
    else:
      self.assertIsNotNone(shape_check)
      shape_check_module, shape_check_messages = shape_check
      err_msg = _run_shape_check_module(exp_f,
          shape_check_module, shape_check_messages, (arg,))

      if expect_error_shape_check is None:
        self.assertIsNone(err_msg)
      else:
        self.assertRegex(err_msg, expect_error_shape_check)


  # An inner function is exported with polymorphic shapes inner_poly_spec, and
  # is called from an outer function, which is exported with outer_poly_spec.
  @jtu.parameterized_filterable(
    testcase_name=lambda kw:f"inner={kw['inner_poly_spec']}_outer={kw['outer_poly_spec']}",  # type: ignore
    #one_containing="",
    # By default arg_shape = (3, 4, 12) for both the outer function and the inner
    # The inner function is exported for f32.
    kwargs=[
      # Both inner and outer are static shapes
      dict(inner_poly_spec="3,4,12", outer_poly_spec="3,4,12"),
      # Inner has poly shapes but outer has static shapes
      dict(inner_poly_spec="3,a,a+b", outer_poly_spec="3,4,12"),
      dict(inner_poly_spec="3,4,3*a", outer_poly_spec="3,4,12"),
      dict(inner_poly_spec="3,a,a", outer_poly_spec="3,4,12",
           expect_error_outer_exp=re.escape(
             r"Found inconsistency 12 != 4 when solving a == args[0].shape[2]")),
      dict(inner_poly_spec="3,4,5*a", outer_poly_spec="3,4,12",
           expect_error_outer_exp=re.escape(
             r"Dimension variable 'a' must have integer value >= 1. "
             r"Non-zero remainder 2 for factor 5 when solving 5*a == args[0].shape[2]")),
      dict(inner_poly_spec="3,4,12+a", outer_poly_spec="3,4,12",
           expect_error_outer_exp=re.escape(
             r"Dimension variable 'a' must have integer value >= 1. "
             r"Found 0 when solving a + 12 == args[0].shape[2]")),
      # Both inner and outer have poly shapes.
      dict(inner_poly_spec="3,a,b", outer_poly_spec="3,4,c"),
      dict(inner_poly_spec="3,4,3*a", outer_poly_spec="3,4,6*c"),
      dict(inner_poly_spec="3,a,a+8", outer_poly_spec="3,c+2,c+10"),
      dict(inner_poly_spec="3,a,a+b", outer_poly_spec="3,4,c",
           expect_error_outer_exp=re.escape(
             r"Dimension variable 'b' must have integer value >= 1. "
             r"Found c + -4 when solving a + b == args[0].shape[2]")),
      dict(inner_poly_spec="3,a,a", outer_poly_spec="3,4,c",
           expect_error_outer_exp=re.escape(
             r"Found inconsistency c != 4 when solving a == args[0].shape[2]"
           )),
      dict(inner_poly_spec="3,a,a", arg_shape=(3, 4),
           outer_poly_spec="3,c",
           expect_error_outer_exp=r"Rank mismatch for args\[0\]"),
      dict(inner_poly_spec="3,a,a+b", arg_dtype=np.int32,
           outer_poly_spec="3,c,d",
           expect_error_outer_exp=r"Dtype mismatch for args\[0\]"),

      dict(inner_poly_spec="3,4,5*a", outer_poly_spec="3,4,c",
           expect_error_outer_exp=re.escape(
             r"Dimension variable 'a' must have integer value >= 1. "
             r"Non-zero remainder mod(c, 5) for factor 5 when solving 5*a == args[0].shape[2]"
           )),
      dict(inner_poly_spec="3,a,a+b", outer_poly_spec="3,c,c",
           expect_error_outer_exp=re.escape(
               r"Dimension variable 'b' must have integer value >= 1. "
               r"Found 0 when solving a + b == args[0].shape[2]")),
      dict(inner_poly_spec="3,a,a+b", outer_poly_spec="c,4,12",
           expect_error_outer_exp=re.escape(
             r"Shape mismatch for args[0].shape[0] (expected same constant)")),
      dict(inner_poly_spec="3,4,5*a", outer_poly_spec="3,4,25*c",
           expect_error_run=re.escape(
             r"Dimension variable 'c' must have integer value >= 1. "
             r"Non-zero remainder 12 for factor 25 when solving 25*c == args[0].shape[2]")),
      dict(inner_poly_spec="3,a,b", outer_poly_spec="3,c+4,12",
           expect_error_run=re.escape(
               r"Dimension variable 'c' must have integer value >= 1. "
               r"Found 0 when solving c + 4 == args[0].shape[1]")),
      dict(inner_poly_spec="3,a,a", outer_poly_spec="3,a,a",
           expect_error_run=re.escape(
               r"Found inconsistency 12 != 4 when solving "
               r"a == args[0].shape[2]")),
  ])
  def test_poly_shape_checks_nested(
      self, inner_poly_spec="3,4,5*a",
      arg_shape=(3, 4, 12), arg_dtype=np.float32,
      outer_poly_spec="3,4,25*c",
      expect_error_outer_exp=None,
      expect_error_run=None):
    # Polymorphic export called with static or polymorphic shapes
    def inner(x):  # x: inner_poly_spec
      return jnp.reshape(x, (-1, x.shape[1]))

    arg = np.arange(np.prod(arg_shape),
                    dtype=arg_dtype).reshape(arg_shape)  # x : f32[3,4,12]
    inner_exp = jax_export.export(inner)(
        jax_export.poly_spec((3, 4, 12), np.float32, inner_poly_spec))

    self.assertEqual(inner_exp.module_uses_dim_vars,
                     (inner_poly_spec != "3,4,12"))
    def outer(x):  # x: outer_poly_spec
      # Use an addition to test that the shapes are refined properly for the
      # result of the call_exported.
      return jax_export.call_exported(inner_exp)(x) + inner(x)

    with contextlib.ExitStack() as stack:
      if expect_error_outer_exp is not None:
        stack.push(self.assertRaisesRegex(ValueError, expect_error_outer_exp))

      # Call it after exporting again, with polymorphic shapes
      outer_exp = jax_export.export(outer)(
          jax_export.poly_spec(arg.shape, arg.dtype, outer_poly_spec))

    if expect_error_outer_exp is not None:
      return

    self.assertEqual(outer_exp.module_uses_dim_vars,
                     (inner_poly_spec != "3,4,12" or outer_poly_spec != "3,4,12"))
    shape_check = outer_exp.shape_check_module()
    if all(core.is_constant_shape(a.shape) for a in outer_exp.in_avals):
      self.assertIsNone(shape_check)
    else:
      self.assertIsNotNone(shape_check)

    with contextlib.ExitStack() as stack:
      if expect_error_run is not None:
        stack.push(self.assertRaisesRegex(Exception, expect_error_run))

      res = _temp_call_exported(outer_exp, arg)

    if expect_error_run is not None:
      return
    self.assertAllClose(2. * inner(arg), res)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
