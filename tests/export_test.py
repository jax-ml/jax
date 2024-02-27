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
from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import math
import re
import unittest

from absl.testing import absltest
import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import export
from jax.experimental.export import _export
from jax.experimental import pjit
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import effects
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
from jax._src.interpreters import mlir

from jax._src.lib.mlir.dialects import hlo

import numpy as np

config.parse_flags_with_absl()

prev_xla_flags = None
def setUpModule():
  global prev_xla_flags
  # This will control the CPU devices. On TPU we always have 2 devices
  prev_xla_flags = jtu.set_host_platform_device_count(2)

# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  prev_xla_flags()

### Setup for testing lowering with effects
@dataclasses.dataclass(frozen=True)
class ForTestingOrderedEffect1(effects.Effect):
  pass

@dataclasses.dataclass(frozen=True)
class ForTestingOrderedEffect2(effects.Effect):
  pass

@dataclasses.dataclass(frozen=True)
class ForTestingUnorderedEffect1(effects.Effect):
  pass


class ForTestingOrderedEffect4NoNullary(effects.Effect):
  def __init__(self, _):
    pass

@dataclasses.dataclass(eq=False)
class ForTestingOrderedEffect5NoEq(effects.Effect):
  pass


_testing_effects = dict(
  ForTestingOrderedEffect1=ForTestingOrderedEffect1(),
  ForTestingOrderedEffect2=ForTestingOrderedEffect2(),
  ForTestingUnorderedEffect1=ForTestingUnorderedEffect1(),
  ForTestingOrderedEffect4NoNullary=ForTestingOrderedEffect4NoNullary(42),
  ForTestingOrderedEffect5NoEq=ForTestingOrderedEffect5NoEq(),
)
# Register the effects
for effect in _testing_effects.values():
  effect_class = effect.__class__
  effects.lowerable_effects.add_type(effect_class)
  effects.control_flow_allowed_effects.add_type(effect_class)
  effects.remat_allowed_effects.add_type(effect_class)
  effects.custom_derivatives_allowed_effects.add_type(effect_class)
  if "Ordered" in str(effect_class):
    effects.ordered_effects.add_type(effect_class)

# A primitive that takes a effect_class_name kwarg with the name of the effect class
# and just doubles its argument.
testing_primitive_with_effect_p = core.Primitive("testing_primitive_with_effect")
testing_primitive_with_effect_p.def_effectful_abstract_eval(
  lambda aval, *x, effect_class_name: (aval, {_testing_effects[effect_class_name]}))

def lowering_testing_primitive_with_effect(ctx, a, *, effect_class_name: str):
  if "Ordered" in effect_class_name:
    token_in = ctx.tokens_in.get(_testing_effects[effect_class_name])[0]
    ctx.set_tokens_out(mlir.TokenSet({_testing_effects[effect_class_name]: (token_in,)}))
  return [mlir.hlo.add(a, a)]

mlir.register_lowering(testing_primitive_with_effect_p,
                       lowering_testing_primitive_with_effect)

## Setup for multi-platform lowering
_testing_multi_platform_to_add = dict(cpu=2., tpu=3., cuda=4., rocm=5.)

def _testing_multi_platform_func(x, *,
                                 effect_class_name: str | None = None):
  # Behaves like x + 2 * _testing_multi_platform_to_add[platform]
  def for_platform(platform: str):
    if effect_class_name is None:
      return 2. * _testing_multi_platform_to_add[platform]
    else:
      return testing_primitive_with_effect_p.bind(
        _testing_multi_platform_to_add[platform],
        effect_class_name=effect_class_name)

  return x + lax.platform_dependent(
    tpu=lambda: for_platform("tpu"),
    cuda=lambda: for_platform("cuda"),
    rocm=lambda: for_platform("rocm"),
    default=lambda: for_platform("cpu"),
  )

def _testing_multi_platform_fun_expected(x,
                                         platform: str | None = None):
  return x + 2. * _testing_multi_platform_to_add[
    xb.canonicalize_platform(platform or jtu.device_under_test())
  ]


def get_exported(fun, vjp_order=0,
                 **export_kwargs):
  """Like export.export but with serialization + deserialization."""
  def serde_exported(*fun_args, **fun_kwargs):
    exp = export.export(fun, **export_kwargs)(*fun_args, **fun_kwargs)
    serialized = export.serialize(exp, vjp_order=vjp_order)
    return export.deserialize(serialized)
  return serde_exported

class JaxExportTest(jtu.JaxTestCase):

  def override_serialization_version(self, version_override: int):
    version = config.jax_serialization_version.value
    if version != version_override:
      self.enter_context(config.jax_serialization_version(version_override))
    logging.info(
        "Using JAX serialization version %s",
        config.jax_serialization_version.value)

  @classmethod
  def setUpClass(cls):
    # Find the available platforms
    cls.platforms = []
    for backend in ["cpu", "gpu", "tpu"]:
      try:
        jax.devices(backend)
      except RuntimeError:
        continue
      cls.platforms.append(backend)
    super().setUpClass()

  def setUp(self):
    super().setUp()
    # Run tests with the maximum supported version by default
    self.override_serialization_version(
        export.maximum_supported_serialization_version)

  def test_basic_export_only(self):
    def my_fun(x):
      return jnp.sin(x)
    exp = get_exported(my_fun)(jax.ShapeDtypeStruct((4,), dtype=np.float32))
    self.assertEqual("my_fun", exp.fun_name)
    self.assertEqual((export.default_lowering_platform(),),
                     exp.lowering_platforms)
    self.assertEqual(jax.tree.flatten(((1,), {}))[1], exp.in_tree)
    self.assertEqual((core.ShapedArray((4,), dtype=np.float32),), exp.in_avals)
    self.assertEqual((core.ShapedArray((4,), dtype=np.float32),), exp.out_avals)

  def test_pytree_export_only(self):
    a = np.arange(4, dtype=np.float32)
    b = np.arange(6, dtype=np.float32)
    def f(a_b_pair, *, a, b):
      return (dict(res=a_b_pair, a=a, b=b), jnp.sin(a), jnp.cos(b))

    exp = get_exported(f, lowering_platforms=("cpu",))((a, b), a=a, b=b)
    a_aval = core.ShapedArray(a.shape, a.dtype)
    b_aval = core.ShapedArray(b.shape, b.dtype)
    self.assertEqual(exp.lowering_platforms, ("cpu",))
    args = ((a, b),)
    kwargs = dict(a=a, b=b)
    self.assertEqual(exp.in_tree, jax.tree.flatten((args, kwargs))[1])
    self.assertEqual(exp.in_avals, (a_aval, b_aval, a_aval, b_aval))
    self.assertEqual(exp.out_tree, jax.tree.flatten(f(*args, **kwargs))[1])
    self.assertEqual(exp.out_avals, (a_aval, b_aval, a_aval, b_aval, a_aval, b_aval))

  def test_basic(self):
    f = jnp.sin
    x = np.arange(4, dtype=np.float32)
    exp_f = get_exported(f)(x)

    f1 = export.call_exported(exp_f)
    self.assertAllClose(f(x), f1(x))

  def test_call_exported_lambda(self):
    # When we export a lambda, the exported.fun_name is not a valid MLIR function name
    f = lambda x: jnp.sin(x)
    x = np.arange(4, dtype=np.float32)
    exp_f = get_exported(f)(x)
    f1 = export.call_exported(exp_f)
    self.assertAllClose(f(x), f1(x))

  def test_call_twice_exported(self):
    def f(x): return jnp.sin(x)
    x = np.arange(4, dtype=np.float32)

    @jax.jit
    def f1(x):
      exp_f = get_exported(f)(x)
      return export.call_exported(exp_f)(x) + export.call_exported(exp_f)(x)

    self.assertAllClose(2. * f(x), f1(x))

  def test_unused_args(self):
    f = lambda x, y: jnp.sin(x)
    x = np.arange(4, dtype=np.float32)
    y = np.arange(6, dtype=np.float32)
    exp_f = get_exported(f)(x, y)

    f1 = export.call_exported(exp_f)
    self.assertAllClose(f(x, y), f1(x, y))

  def test_pytree(self):
    a = np.arange(4, dtype=np.float32)
    b = np.arange(6, dtype=np.float32)
    def f(a_b_pair, a, b):
      return (dict(res=a_b_pair, a=a, b=b), jnp.sin(a), jnp.cos(b))

    exp_f = get_exported(f)((a, b), a=a, b=b)
    f1 = export.call_exported(exp_f)
    self.assertAllClose(f((a, b), a=a, b=b),
                        f1((a, b), a=a, b=b))

  def test_error_wrong_intree(self):
    def f(a_b_pair, *, c):
      return jnp.sin(a_b_pair[0]) + jnp.cos(a_b_pair[1]) + c
    a = b = c = np.arange(4, dtype=np.float32)
    exp_f = get_exported(f)((a, b), c=c)

    with self.assertRaisesRegex(
        ValueError,
        "The invocation args and kwargs must have the same pytree structure"):
      export.call_exported(exp_f)(a, b, c=(a, b))

  def test_error_wrong_avals(self):
    def f(a, *, b):  # a: f32[4] and b: f32[4]
      return jnp.sin(a) + jnp.cos(b)
    f32_4 = np.arange(4, dtype=np.float32)
    exp_f = get_exported(f)(f32_4, b=f32_4)

    with self.assertRaisesRegex(ValueError,
        r"Shape mismatch for args\[0\].shape\[0\]"):
      export.call_exported(exp_f)(np.arange(6, dtype=np.float32), b=f32_4)

    with self.assertRaisesRegex(ValueError,
        r"Shape mismatch for kwargs\['b'\].shape\[0\]"):
      export.call_exported(exp_f)(f32_4, b=np.arange(6, dtype=np.float32))

    with self.assertRaisesRegex(ValueError,
        r"Rank mismatch for args\[0\]"):
      export.call_exported(exp_f)(f32_4.reshape((1, 4)), b=f32_4)

    with self.assertRaisesRegex(ValueError,
        r"Dtype mismatch for args\[0\]"):
      export.call_exported(exp_f)(f32_4.astype(np.float16), b=f32_4)

  @jtu.parameterized_filterable(
    testcase_name=lambda kw: kw["platform"],
    kwargs=[dict(platform=p)
            for p in ("cpu", "cuda", "rocm", "tpu")])
  def test_error_wrong_platform(self, platform):
    a = np.arange(4, dtype=np.float32)

    exp_f = get_exported(jnp.sin, lowering_platforms=(platform,))(a)
    if xb.canonicalize_platform(jtu.device_under_test()) == platform:
      raise unittest.SkipTest("Uninteresting scenario")

    with self.assertRaisesRegex(
        ValueError, "The exported function .* was lowered for platform"):
      export.call_exported(exp_f)(a)

    # Now try with the platform check disabled
    exp_f_no_platform_check = get_exported(
      jnp.sin, lowering_platforms=(platform,),
      disabled_checks=[export.DisabledSafetyCheck.platform()])(a)
    res = export.call_exported(exp_f_no_platform_check)(a)
    self.assertAllClose(res, jnp.sin(a))

  @jtu.parameterized_filterable(
    testcase_name=lambda kw: kw["dialect"],
    kwargs=[dict(dialect=dialect)
            for dialect in ("stablehlo",)]
  )
  def test_error_disallowed_custom_call(self, dialect):
    # If we use hlo.custom_call we detect invalid custom call targets.
    # Set up a primitive with custom lowering rules
    test_primitive = core.Primitive("_test_primitive_disallowed_custom_call")
    test_primitive.def_abstract_eval(lambda in_aval: in_aval)
    def test_primitive_lowering(ctx, arg):
      op = dict(stablehlo=hlo.CustomCallOp)[dialect]
      return op([arg.type], [arg], "disallowed_call_target").results
    mlir.register_lowering(test_primitive, test_primitive_lowering)
    self.addCleanup(lambda: mlir.register_lowering(test_primitive, None))

    a = np.arange(3, dtype=np.float32)
    with self.assertRaisesRegex(ValueError,
        "Cannot serialize code with custom calls whose targets .*"):
      get_exported(
        lambda a: a + test_primitive.bind(a)
      )(a)

    # Now try again with the safety check disabled
    exp = get_exported(
      lambda a: a + test_primitive.bind(a),
      disabled_checks=[export.DisabledSafetyCheck.custom_call("disallowed_call_target")]
    )(a)
    self.assertIn("disallowed_call_target", exp.mlir_module())

  def test_grad(self):
    f = lambda x: jnp.sum(jnp.sin(x))
    x = np.arange(4, dtype=np.float32)
    exp_f = get_exported(f, vjp_order=1)(x)

    f1 = export.call_exported(exp_f)
    self.assertAllClose(jax.grad(f)(x), jax.grad(f1)(x))

  def test_higher_order_grad(self):
    f = lambda x: x ** 3
    x = np.float32(4.)
    exp_f = get_exported(f, vjp_order=3)(x)

    f1 = export.call_exported(exp_f)
    self.assertAllClose(jax.grad(f)(x),
                        jax.grad(f1)(x))
    self.assertAllClose(jax.grad(jax.grad(f))(x),
                        jax.grad(jax.grad(f1))(x))
    self.assertAllClose(jax.grad(jax.grad(jax.grad(f)))(x),
                        jax.grad(jax.grad(jax.grad(f1)))(x))

  def test_grad_int(self):
    def f(xi, xf):
      return (2 * xi.T, xf.T * xf.T)

    xi = np.arange(6, dtype=np.int32).reshape((2, 3))
    xf = np.arange(12, dtype=np.float32).reshape((3, 4))

    # Native JAX 1st order vjp
    (f_outi, f_outf), f_vjp = jax.vjp(f, xi, xf)
    f_outi_ct = np.ones(f_outi.shape, dtype=f_outi.dtype)
    f_outf_ct = np.ones(f_outf.shape, dtype=f_outf.dtype)
    xi_ct, xf_ct = f_vjp((f_outi_ct, f_outf_ct))

    # Native JAX 2nd order vjp
    res, f_vjp2 = jax.vjp(f_vjp, (f_outi_ct, f_outf_ct))
    self.assertAllClose(res, (xi_ct, xf_ct))
    (f_outi_ct2, f_outf_ct2), = f_vjp2((xi_ct, xf_ct))

    exp = get_exported(f, vjp_order=2)(xi, xf)
    fr = export.call_exported(exp)

    res = fr(xi, xf)
    self.assertAllClose(res, (f_outi, f_outf))

    # Reloaded 1st order vjp
    (fr_outi, fr_outf), fr_vjp = jax.vjp(fr, xi, xf)
    self.assertAllClose(fr_outi, f_outi)
    self.assertAllClose(fr_outf, f_outf)
    xri_ct, xrf_ct = fr_vjp((f_outi_ct, f_outf_ct))
    self.assertAllClose(xri_ct, xi_ct)
    self.assertAllClose(xrf_ct, xf_ct)

    # Reloaded 2nd order vjp
    res, f_vjp2 = jax.vjp(fr_vjp, (f_outi_ct, f_outf_ct))
    self.assertAllClose(res, (xi_ct, xf_ct))
    (fr_outi_ct2, fr_outf_ct2), = f_vjp2((xi_ct, xf_ct))
    self.assertAllClose(fr_outi_ct2, f_outi_ct2)
    self.assertAllClose(fr_outf_ct2, f_outf_ct2)

  def test_pytree_vjp(self):
    def f(a_b_pair, *, a, b):
      return (dict(res=a_b_pair, a=2. * a, b=3. * b),
              jnp.sin(4. * a))

    a = np.arange(4, dtype=np.float32)
    b = np.arange(6, dtype=np.float32)
    exp_f = get_exported(f, vjp_order=1)((a, b), a=a, b=b)

    out_ct = f((a, b), a=a, b=b)  # The output has the right structure as the cotangent
    def f1_jax(a, b):  # For VJP, make a function without kwargs
      res = f((a, b), a=a, b=b)
      return res
    def f1_exp(a, b):  # For VJP, make a function without kwargs
      res = export.call_exported(exp_f)((a, b), a=a, b=b)
      return res
    jax_vjp = jax.vjp(f1_jax, a, b)[1](out_ct)
    exp_vjp = jax.vjp(f1_exp, a, b)[1](out_ct)
    self.assertAllClose(jax_vjp, exp_vjp)

  def test_roundtrip(self):
    def f1(x):
      return jnp.sin(x)
    a = np.arange(4, dtype=np.float32)
    exp_f1 = get_exported(f1)(a)
    def f2(x):
      res1 = export.call_exported(exp_f1)(x)
      res2 = export.call_exported(exp_f1)(res1)
      return jnp.cos(res2)
    exp_f2 = get_exported(f2)(a)

    self.assertAllClose(jnp.cos(jnp.sin(jnp.sin(a))),
                        export.call_exported(exp_f2)(a))

  def test_poly_export_only(self):
    a = np.arange(12, dtype=np.float32).reshape((3, 4))
    def f(a, b):  # a: f32[2w,h]  b: f32[w,h]
      return jnp.concatenate([a, b], axis=0)

    scope = export.SymbolicScope()
    exp = get_exported(f)(
        jax.ShapeDtypeStruct(export.symbolic_shape("(2*w, h)", scope=scope), a.dtype),
        jax.ShapeDtypeStruct(export.symbolic_shape("(w, h)", scope=scope), a.dtype))
    self.assertEqual("(2*w, h)", str(exp.in_avals[0].shape))
    self.assertEqual("(w, h)", str(exp.in_avals[1].shape))
    self.assertEqual("(3*w, h)", str(exp.out_avals[0].shape))

    # Peek at the module
    module_str = exp.mlir_module()
    self.assertEqual(config.jax_serialization_version.value >= 7,
                     "shape_assertion" in module_str)
    self.assertIn("jax.uses_shape_polymorphism = true", module_str)
    wrapped_main_expected_re = (
      r"@_wrapped_jax_export_main\("
      r"%arg0: tensor<i..> {jax.global_constant = \"h\"}.*"
      r"%arg1: tensor<i..> {jax.global_constant = \"w\"}.*"
      r"%arg2: tensor<\?x\?xf32>"
    )
    self.assertRegex(module_str, wrapped_main_expected_re)

    # Look for private inner functions that are generated to compute the
    # dimension variables and shape assertions. All those functions must
    # have jax.global_constant attributes on all the arguments.
    for func_name, func_args in re.findall(
        r"func.func private @([\w]+)\((.+)\) ->",
        module_str):
      if func_name == "_wrapped_jax_export_main":
        continue
      func_args_count = len(re.findall(r"%arg\d+", func_args))
      func_args_constant_attrs = len(re.findall(r"jax.global_constant = ",
                                                func_args))
      self.assertEqual(func_args_count, func_args_constant_attrs)

  def test_poly_pytree_export_only(self):
    a = np.arange(12, dtype=np.float32).reshape((3, 4))
    def f(a0, a1, *, ak):
      return jnp.concatenate([a0, a1, ak], axis=0)

    a_poly_spec = jax.ShapeDtypeStruct(export.symbolic_shape("(w, h)"), a.dtype)
    exp = get_exported(f)(a_poly_spec, a_poly_spec, ak=a_poly_spec)
    self.assertEqual("(w, h)", str(exp.in_avals[0].shape))
    self.assertEqual("(3*w, h)", str(exp.out_avals[0].shape))

  def test_poly_export_error_symbolic_scope(self):
    a = np.arange(12, dtype=np.float32).reshape((3, 4))
    def f(x, y):
      return jnp.concatenate([x, y], axis=1)

    x_poly_spec = jax.ShapeDtypeStruct(export.symbolic_shape("(w, h1)"), a.dtype)
    y_poly_spec = jax.ShapeDtypeStruct(export.symbolic_shape("(w, h2)"), a.dtype)
    with self.assertRaisesRegex(
        ValueError,
        re.compile(
            "Invalid mixing of symbolic scopes when exporting f.*"
            r"Expected current \(from args\[0\]\) scope .*"
            r"and found for 'w' \(args\[1\]\) scope .*", re.DOTALL)):
      get_exported(f)(x_poly_spec, y_poly_spec)


  @jtu.parameterized_filterable(
    kwargs=[
      dict(v=v)
      for v in range(export.minimum_supported_serialization_version - 1,
                     export.maximum_supported_serialization_version + 2)])
  def test_poly_basic_versions(self, v: int):
    self.override_serialization_version(v)
    with contextlib.ExitStack() as e:
      if not (export.minimum_supported_serialization_version <= v
              <= export.maximum_supported_serialization_version):
        e.enter_context(self.assertRaisesRegex(
          ValueError,
          f"The requested jax_serialization version {v} is outside the range of supported versions"))

      exp = get_exported(jnp.sin)(
          jax.ShapeDtypeStruct(export.symbolic_shape("w, h"), np.float32))
      x = np.arange(30, dtype=np.float32).reshape((5, 6))
      res = export.call_exported(exp)(x)
      self.assertAllClose(res, np.sin(x))

  # A function is exported with f32[poly_spec] and is called with different arg
  # shapes. We use export.call_exported and we also run the shape check
  # module.
  @jtu.parameterized_filterable(
    testcase_name=lambda kw:f"poly_spec={kw['poly_spec']}_arg_shape={kw['arg_shape']}",  # type: ignore
    kwargs=[
      dict(poly_spec="3,4,12", arg_shape=(3, 4, 12)),
      dict(poly_spec="3,4,12", arg_shape=(3, 4, 13),
           # The shape check module does not test constant dimensions
           expect_error=re.escape(
               r"Shape mismatch for args[0].shape[2] (expected same constant)")),
      dict(poly_spec="3,4,6*a", arg_shape=(3, 4, 12)),
      dict(poly_spec="3,a,a+8", arg_shape=(3, 4, 12)),
      dict(poly_spec="3,4,a+1", arg_shape=(3, 4, 1),
           expect_error=re.escape(
               "Expected value >= 1 for dimension variable 'a'. "
               "Using the following polymorphic shapes specifications: args[0].shape = (3, 4, a + 1). "
               "Obtained dimension variables: 'a' = 0"
          )),
      dict(poly_spec="3,4,6*a", arg_shape=(3, 4, 13),
           expect_error=re.escape(
              "Division had remainder 1 when computing the value of 'a'"
          )),
      dict(poly_spec="3,a,a+8", arg_shape=(3, 4, 13),
           expect_error=re.escape(
             "Found inconsistency between dimension size "
             "args[0].shape[2] (= 13) and the specification 'a + 8' (= 12)"
          )),
  ])
  def test_poly_shape_checks(
      self, poly_spec="3,a,a+8",
      arg_shape=(3, 4, 12), arg_dtype=np.float32,
      expect_error=None):  # If given, error from running the exported module

    def f(x):  # x: f32[poly_spec]
      return jnp.reshape(x, (-1, x.shape[1]))

    disabled_checks = ()
    exp_f = get_exported(f, disabled_checks=disabled_checks)(
        jax.ShapeDtypeStruct(export.symbolic_shape(poly_spec), np.float32))
    self.assertEqual(exp_f.uses_shape_polymorphism, poly_spec != "3,4,12")
    arg = np.arange(np.prod(arg_shape),
                    dtype=arg_dtype).reshape(arg_shape)  # arg : f32[3,4,12]

    with contextlib.ExitStack() as stack:
      if expect_error is not None:
        stack.push(self.assertRaisesRegex(Exception, expect_error))

      assert core.is_constant_shape(arg.shape)
      res = export.call_exported(exp_f)(arg)

    if not expect_error:
      self.assertAllClose(res, f(arg))

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
      # Inner has poly shapes but outer has static shapes. When we call inner
      # we do the shape constraint checking
      dict(inner_poly_spec="3,a,a+b", outer_poly_spec="3,4,12"),
      dict(inner_poly_spec="3,4,3*a", outer_poly_spec="3,4,12"),
      dict(inner_poly_spec="3,a,a", outer_poly_spec="3,4,12",
           expect_error_outer_exp=re.escape(
             "Found inconsistency between dimension size "
             "args[0].shape[2] (= 12) and the specification 'a' (= 4)")),
      dict(inner_poly_spec="3,4,5*a", outer_poly_spec="3,4,12",
           expect_error_outer_exp=re.escape(
             "Division had remainder 2 when computing the value of 'a'")),
      dict(inner_poly_spec="3,4,12+a", outer_poly_spec="3,4,12",
           expect_error_outer_exp=re.escape(
              "Expected value >= 1 for dimension variable 'a'. "
              "Using the following polymorphic shapes specifications: args[0].shape = (3, 4, a + 12). "
              "Obtained dimension variables: 'a' = 0 from specification "
              "'a + 12' for dimension args[0].shape[2] (= 12)")),
      # Both inner and outer have poly shapes.
      dict(inner_poly_spec="3,a,b", outer_poly_spec="3,4,c"),
      dict(inner_poly_spec="3,4,3*a", outer_poly_spec="3,4,6*c"),
      dict(inner_poly_spec="3,a,a+8", outer_poly_spec="3,c+2,c+10"),
      dict(inner_poly_spec="3,a,a+b", outer_poly_spec="3,4,c",
           expect_error_outer_exp=re.escape(
             "Expected value >= 1 for dimension variable 'b'. "
             "Using the following polymorphic shapes specifications: args[0].shape = (3, a, b + a). "
             "Obtained dimension variables: 'a' = 4 from specification "
             "'a' for dimension args[0].shape[1] (= 4), "
             "'b' = c - 4 from specification 'b + a' for dimension args[0].shape[2] (= c),")),
      dict(inner_poly_spec="3,a,a", outer_poly_spec="3,4,c",
           expect_error_outer_exp=re.escape(
             "Found inconsistency between dimension size "
             "args[0].shape[2] (= c) and the specification 'a' (= 4)")),
      dict(inner_poly_spec="3,a,a", arg_shape=(3, 4),
           outer_poly_spec="3,c",
           expect_error_outer_exp=r"Rank mismatch for args\[0\]"),
      dict(inner_poly_spec="3,a,a+b", arg_dtype=np.int32,
           outer_poly_spec="3,c,d",
           expect_error_outer_exp=r"Dtype mismatch for args\[0\]"),
      dict(inner_poly_spec="3,4,5*a", outer_poly_spec="3,4,c",
           expect_error_outer_exp=re.escape(
              "Division had remainder mod(c, 5) when computing the value of 'a'")),
      dict(inner_poly_spec="3,a,a+b", outer_poly_spec="3,c,c",
           expect_error_outer_exp=re.escape(
               "Expected value >= 1 for dimension variable 'b'. "
               "Using the following polymorphic shapes specifications: args[0].shape = (3, a, b + a). "
               "Obtained dimension variables: 'a' = c from "
               "specification 'a' for dimension args[0].shape[1] (= c), "
               "'b' = 0 from specification 'b + a' for dimension args[0].shape[2] (= c)")),
      dict(inner_poly_spec="3,a,a+b", outer_poly_spec="c,4,12",
           expect_error_outer_exp=re.escape(
               "Shape mismatch for args[0].shape[0] (expected same constant)")),
      dict(inner_poly_spec="3,4,5*a", outer_poly_spec="3,4,25*c",
           expect_error_run=re.escape(
              "Division had remainder 12 when computing the value of 'c'")),
      dict(inner_poly_spec="3,a,b", outer_poly_spec="3,c+4,12",
           expect_error_run=re.escape(
               "Expected value >= 1 for dimension variable 'c'. "
               "Using the following polymorphic shapes specifications: args[0].shape = (3, c + 4, 12). "
               "Obtained dimension variables: 'c' = 0")),
      dict(inner_poly_spec="3,a,a", outer_poly_spec="3,a,a",
           expect_error_run=re.escape(
               "Found inconsistency between dimension size "
               "args[0].shape[2] (= 12) and the specification 'a' (= 4)")),
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
    inner_exp = get_exported(inner)(
        jax.ShapeDtypeStruct(export.symbolic_shape(inner_poly_spec), np.float32))

    self.assertEqual(inner_exp.uses_shape_polymorphism,
                     (inner_poly_spec != "3,4,12"))
    def outer(x):  # x: outer_poly_spec
      # Use an addition to test that the shapes are refined properly for the
      # result of the call_exported.
      return export.call_exported(inner_exp)(x) + inner(x)

    with contextlib.ExitStack() as stack:
      if expect_error_outer_exp is not None:
        stack.push(self.assertRaisesRegex(ValueError, expect_error_outer_exp))

      # Call it after exporting again, with polymorphic shapes
      outer_exp = get_exported(outer)(
          jax.ShapeDtypeStruct(export.symbolic_shape(outer_poly_spec), arg.dtype))

    if expect_error_outer_exp is not None:
      return

    self.assertEqual(outer_exp.uses_shape_polymorphism,
                     (inner_poly_spec != "3,4,12" or outer_poly_spec != "3,4,12"))

    with contextlib.ExitStack() as stack:
      if expect_error_run is not None:
        stack.push(self.assertRaisesRegex(Exception, expect_error_run))

      res = export.call_exported(outer_exp)(arg)

    if expect_error_run is not None:
      return
    self.assertAllClose(2. * inner(arg), res)

  # Tests details of the shape constraints errors
  # This test exists also in shape_poly_test.py. Here we test the
  # call_exported error reporting.
  @jtu.parameterized_filterable(
    testcase_name=lambda kw: kw["shape"],  # assume "shape" is unique
    kwargs=[
      dict(shape=(8, 2, 9),  # a = 2, b = 3, c = 4
           poly_spec="(a + 2*b, a, a + b + c)"),
      dict(shape=(2, 2, 6),  # a = 2, b = 0, c = 4
           poly_spec="(a + 2*b, a, a + b + c)",
           expect_error=(
             "Input shapes do not match the polymorphic shapes specification. "
             "Expected value >= 1 for dimension variable 'b'. "
             "Using the following polymorphic shapes specifications: args[0].shape = (2*b + a, a, c + b + a). "
             "Obtained dimension variables: 'a' = 2 from specification 'a' for dimension args[0].shape[1] (= 2), "
             "'b' = 0 from specification '2*b + a' for dimension args[0].shape[0] (= 2), . "
             "Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#shape-assertion-errors for more details."
           )),
      dict(shape=(3, 2, 6),  # a = 2, b = 0.5, c = 4 - b is not integer
           poly_spec="(a + 2*b, a, a + b + c)",
           expect_error=(
             "Input shapes do not match the polymorphic shapes specification. "
             "Division had remainder 1 when computing the value of 'b'. "
             "Using the following polymorphic shapes specifications: args[0].shape = (2*b + a, a, c + b + a). "
             "Obtained dimension variables: 'a' = 2 from specification 'a' for dimension args[0].shape[1] (= 2), . "
             "Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#shape-assertion-errors for more details."
           )),
      dict(shape=(8, 2, 6),  # a = 2, b = 3 - inconsistency
           poly_spec="(a + 2*b, a, a + b)",
           expect_error=(
             "Input shapes do not match the polymorphic shapes specification. "
             "Found inconsistency between dimension size args[0].shape[0] (= 8) and the specification '2*b + a' (= 10). "
             "Using the following polymorphic shapes specifications: args[0].shape = (2*b + a, a, b + a). "
             "Obtained dimension variables: 'a' = 2 from specification 'a' for dimension args[0].shape[1] (= 2), "
             "'b' = 4 from specification 'b + a' for dimension args[0].shape[2] (= 6), . "
             "Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#shape-assertion-errors for more details."
           )),
      dict(shape=(7, 2, 36),  # a = 2, b = 3, c = 6 - cannot solve c
           poly_spec="(2 * a + b, a, c * c)",
           expect_error=(
             "Cannot solve for values of dimension variables {'c'}. "
             "We can only solve linear uni-variate constraints. "
             "Using the following polymorphic shapes specifications: args[0].shape = (b + 2*a, a, c^2). "
             "Unprocessed specifications: 'c^2' for dimension size args[0].shape[2]. "
             "Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#dimension-variables-must-be-solvable-from-the-input-shapes for more details."
           )),
  ])
  def test_shape_constraints_errors(self, *,
      shape, poly_spec: str, expect_error: str | None = None):
    def f_jax(x):  # x: f32[a + 2*b, a, a + b + c]
      return 0.

    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    with contextlib.ExitStack() as stack:
      if expect_error is not None:
        stack.push(self.assertRaisesRegex(Exception, re.escape(expect_error)))
      exp = get_exported(f_jax)(
          jax.ShapeDtypeStruct(export.symbolic_shape(poly_spec), x.dtype))
      export.call_exported(exp)(x)

  def test_poly_booleans(self):
    # For booleans we use a special case ConvertOp to cast to and from
    # dynamic shapes arguments.
    def f_jax(x):  # x: bool[b]
      return jnp.logical_not(x)

    x = np.array([True, False, True, False], dtype=np.bool_)
    exp = get_exported(f_jax)(jax.ShapeDtypeStruct(export.symbolic_shape("b"),
                                                   x.dtype))
    res = export.call_exported(exp)(x)
    self.assertAllClose(f_jax(x), res)

  @jtu.parameterized_filterable(
    kwargs=[
      dict(dtype=dtype)
      for dtype in dtypes._jax_types if dtype != np.dtype("bool")
  ])
  def test_poly_numeric_dtypes(self, dtype=np.int32):
    if str(dtype) in {"float8_e4m3b11fnuz",
                      "float8_e4m3fnuz",
                      "float8_e5m2fnuz",
                      "int4",
                      "uint4"}:
      self.skipTest(f"TODO: serialization not supported for {str(dtype)}")
    def f_jax(x):
      return x + x

    x = np.arange(6, dtype=dtype)
    exp = get_exported(f_jax)(jax.ShapeDtypeStruct(export.symbolic_shape("b"),
                                                   x.dtype))
    res = export.call_exported(exp)(x)
    self.assertAllClose(f_jax(x), res)

  def test_poly_expressions(self):
    # Calling an Exported module whose output shape contains symbolic
    # expressions
    def output_shape(b):
      return (b + b, b - b, b * b,
              (b + 13) // b, (b + 13) % b,
              core.max_dim(b - 5, 0))
    def f(x):  # x: f32[b]
      b = x.shape[0]
      return jnp.ones(output_shape(b), dtype=x.dtype)
    x = np.arange(5, dtype=np.float32)
    exp = get_exported(f)(jax.ShapeDtypeStruct(export.symbolic_shape("b"),
                                                x.dtype))
    # Call with static shapes
    res = export.call_exported(exp)(x)
    self.assertAllClose(res, f(x))

    # Now re-export with shape polymorphism
    x_spec = jax.ShapeDtypeStruct(export.symbolic_shape("a"), x.dtype)
    exp2 = get_exported(export.call_exported(exp))(x_spec)
    a = exp2.in_avals[0].shape[0]
    self.assertEqual(exp2.out_avals[0].shape, output_shape(a))

  def test_with_sharding(self):
    nr_devices = 2
    if len(jax.devices()) < nr_devices:
      self.skipTest("Need at least 2 devices")
    export_devices = jax.devices()[0:nr_devices]
    export_mesh = Mesh(export_devices, axis_names=("x",))
    a = np.arange(16 * 4, dtype=np.float32).reshape((16, 4))
    @functools.partial(
        jax.jit,
        in_shardings=(jax.sharding.NamedSharding(export_mesh, P("x", None),),),
        out_shardings=jax.sharding.NamedSharding(export_mesh, P(None, "x")))
    def f_jax(b):  # b: f32[16 // DEVICES, 4]
      return b * 2.

    res_native = f_jax(a)
    exp = get_exported(f_jax)(a)

    self.assertEqual(exp.nr_devices, len(export_devices))
    run_devices = export_devices[::-1]  # We can use other devices
    run_mesh = Mesh(run_devices, "y")
    a_device = jax.device_put(a, jax.sharding.NamedSharding(run_mesh, P()))

    expected_re = re.compile(
      # The top-level input it replicated
      r"func.func .* @main\(%arg0: tensor<16x4xf32>.*mhlo.sharding = \"{replicated}\"}\).*"
      # We apply the in_shardings for f_jax
      r".*custom_call @Sharding\(%arg0\).*mhlo.sharding = \"{devices=\[2,1\]<=\[2\]}\"}.*"
      r"%1 = .*call @call_exported_f_jax.*"
      # We apply the out_shardings for f_jax
      r".*custom_call @Sharding\(%1\).*mhlo.sharding = \"{devices=\[1,2\]<=\[2\]}\"}.*",
      re.DOTALL)
    hlo = jax.jit(export.call_exported(exp)).lower(a_device).as_text()
    self.assertRegex(hlo, expected_re)

    res_exported = export.call_exported(exp)(a_device)
    self.assertAllClose(res_native, res_exported)

    # Test error reporting
    with self.assertRaisesRegex(
        NotImplementedError,
        "Exported module .* was lowered for 2 devices and is called in a context with 1 device"):
      _ = export.call_exported(exp)(a)

    with self.assertRaisesRegex(
        NotImplementedError,
        "Exported module .* was lowered for 2 devices and is called in a context with 1 device"):
      mesh1 = Mesh(jax.devices()[0:1], axis_names=("x",))
      _ = jax.jit(
        export.call_exported(exp),
        in_shardings=(jax.sharding.NamedSharding(mesh1, P("x", None)),)
      )(a)

  @jtu.parameterized_filterable(
    kwargs=[
      dict(testcase_name=f"_poly={poly}", poly=poly)
      for poly in (None, "2*b1,_", "_,b2", "2*b1,b2")
    ])
  def test_shard_map_collective_permute(self, poly=None):
    if len(jax.devices()) < 2:
      self.skipTest("Test requires at least 2 local devices")
    devices = np.array(jax.devices()[:2])  # use 2 devices
    mesh = Mesh(devices, axis_names=("x",))
    a = np.arange(4 * 4, dtype=np.float32).reshape((4, 4))

    @functools.partial(
      pjit.pjit,
      in_shardings=NamedSharding(mesh, P("x", None),),
      out_shardings=NamedSharding(mesh, P("x", None)))
    @functools.partial(
        shard_map, mesh=mesh,
        in_specs=(P("x", None),), out_specs=P("x", None))
    def f_jax(b):  # b: f32[2, 4]
      axis_size = lax.psum(1, "x")
      perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
      return lax.ppermute(b, "x", perm=perm)

    args_specs = export.symbolic_args_specs((a,), polymorphic_shapes=poly)
    exp = get_exported(f_jax)(*args_specs)

    # Test JAX native execution
    res_jax = f_jax(a)
    b0, b1 = np.split(a, 2, axis=0)  # The shard_map splits on axis 0
    b0, b1 = b1, b0
    expected = np.concatenate([b0, b1], axis=0)  # out_specs concatenates on axis 0
    self.assertAllClose(res_jax, expected)
    self.assertLen(res_jax.addressable_shards, len(devices))

    # Test reloaded execution.
    f_r = export.call_exported(exp)
    with self.assertRaisesRegex(
        Exception,
        "Exported module .* was lowered for 2 devices and is "
        "called in a context with 1 devices"):
      _ = f_r(a)  # A is all on the default device

    # Replicate the input so that the execution knows
    # that we are using multiple devices
    a_replicated = jax.device_put(a, NamedSharding(mesh, None))
    res_r = f_r(a_replicated)
    self.assertAllClose(res_r, expected)
    self.assertLen(res_r.addressable_shards, len(devices))
    for i in range(len(devices)):
      self.assertEqual(res_jax.addressable_shards[i].device,
                       res_r.addressable_shards[i].device)
      self.assertEqual(res_jax.addressable_shards[i].index,
                       res_r.addressable_shards[i].index)
      self.assertAllClose(res_jax.addressable_shards[i].data,
                          res_r.addressable_shards[i].data)

  @jtu.parameterized_filterable(
    kwargs=[
      dict(in_shardings=in_shardings, out_shardings=out_shardings,
           with_mesh=with_mesh)
      for in_shardings in ("missing", None, "P")
      for out_shardings in ("missing", None, "P")
      for with_mesh in (True, False)
  ])
  def test_grad_with_sharding(self, in_shardings="P", out_shardings=None,
                              with_mesh=False):
    if len(jax.devices()) < 2:
      self.skipTest("Test requires at least 2 devices")
    x_shape = (10, 20)
    x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)
    def f_jax(x):  # x: f32[10,20] -> f32[20,10]
      return jnp.sin(x.T)

    mesh = Mesh(jax.devices()[:2], "d")
    pjit_kwargs = {}
    # Use NamedShardings if we don't have a mesh_context
    if with_mesh:
      sharding_None_d = P(None, "d")
      sharding_d_None = P("d", None)
    else:
      sharding_None_d = NamedSharding(mesh, P(None, "d"))
      sharding_d_None = NamedSharding(mesh, P("d", None))

    if in_shardings != "missing":
      pjit_kwargs["in_shardings"] = (
        sharding_None_d if in_shardings == "P" else None)
    if out_shardings != "missing":
      pjit_kwargs["out_shardings"] = (
        sharding_d_None if out_shardings == "P" else None)
    f_jax_pjit = pjit.pjit(f_jax, **pjit_kwargs)

    with contextlib.ExitStack() as stack:
      if with_mesh:
        stack.enter_context(mesh)
      # Serialize higher-order gradiends
      exp = get_exported(f_jax_pjit, vjp_order=2)(x)
      exp_vjp = exp.vjp()
      # Try 2nd order grad as well
      exp_vjp2 = exp_vjp.vjp()

    vjp_module_str = str(exp_vjp.mlir_module())

    if in_shardings == "P":
      primal_in_sharding = "{devices=[1,2]<=[2]}"
    else:
      primal_in_sharding = "{replicated}"
    if out_shardings == "P":
      primal_out_sharding = "{devices=[2,1]<=[2]}"
    else:
      primal_out_sharding = "{replicated}"

    # TODO(b/326476605): Change the condition below if required.
    if in_shardings == "P":
      main = re.compile(
        r"func.func public @main\(%arg0: tensor<10x20xf32>.*"
        "mhlo.sharding = \"" + re.escape(primal_in_sharding) + "\""
        r".*%arg1: tensor<20x10xf32>.*"
        "mhlo.sharding = \"" + re.escape(primal_out_sharding) + "\""
        # result
        r".*->.*\(tensor<10x20xf32>.*"
        "mhlo.sharding = \"" + re.escape(primal_in_sharding) + "\"")
      self.assertRegex(vjp_module_str, main)

    # Custom calls for the primal input shape all match primal_in_sharding
    primal_in_calls = re.findall(
      r"custom_call @Sharding.*mhlo.sharding = \"(.+)\".*:.*tensor<10x20xf32>",
      vjp_module_str)
    self.assertTrue(
      all(s == primal_in_sharding for s in primal_in_calls),
      primal_in_calls
    )

    # Custom calls for the primal output shape all match primal_out_sharding
    primal_out_calls = re.findall(
      r"custom_call @Sharding.*mhlo.sharding = \"(.+)\".*:.*tensor<20x10xf32>",
      vjp_module_str)
    self.assertTrue(
      all(s == primal_out_sharding for s in primal_out_calls),
      primal_in_calls
    )

    # Call the exported gradient functions. In order to set the device context
    # we replicate the inputs. If we don't use a mesh context and there are
    # no shardings on inputs or outputs, then we have serialized for one
    # device.
    if in_shardings != "P" and out_shardings != "P" and not with_mesh:
      self.assertEqual(exp_vjp.nr_devices, 1)
      self.assertEqual(exp_vjp2.nr_devices, 1)
      call_mesh = Mesh(jax.devices()[:1], "e")
    else:
      self.assertEqual(exp_vjp.nr_devices, 2)
      self.assertEqual(exp_vjp2.nr_devices, 2)
      call_mesh = Mesh(jax.devices()[:2], "e")

    g1 = pjit.pjit(export.call_exported(exp_vjp),
                   in_shardings=(NamedSharding(call_mesh, None),
                                 NamedSharding(call_mesh, None)))(x, x.T)
    _, f_jax_vjp = jax.vjp(f_jax, x)
    xbar = f_jax_vjp(x.T)
    self.assertAllClose(xbar, g1)

    g2 = pjit.pjit(export.call_exported(exp_vjp2),
                   in_shardings=(NamedSharding(call_mesh, None),
                                 NamedSharding(call_mesh, None),
                                 NamedSharding(call_mesh, None)))(x, x.T, x)
    _, f_jax_vjp2 = jax.vjp(f_jax_vjp, x.T)
    xbar2, = f_jax_vjp2((x,))
    self.assertAllClose(xbar2, g2[1])

  def test_multi_platform(self):
    x = np.arange(8, dtype=np.float32)
    exp = get_exported(_testing_multi_platform_func,
                        lowering_platforms=("tpu", "cpu", "cuda"))(x)
    self.assertEqual(exp.lowering_platforms, ("tpu", "cpu", "cuda"))
    module_str = str(exp.mlir_module())
    expected_main_re = (
      r"@main\("
      r"%arg0: tensor<i..>.*jax.global_constant = \"_platform_index\".*, "
      r"%arg1: tensor<8xf32>.*->")
    self.assertRegex(module_str, expected_main_re)

    self.assertIn("jax.uses_shape_polymorphism = true",
                  module_str)

    # Call with argument placed on different plaforms
    for platform in self.__class__.platforms:
      x_device = jax.device_put(x, jax.devices(platform)[0])
      res_exp = export.call_exported(exp)(x_device)
      self.assertAllClose(
        res_exp,
        _testing_multi_platform_fun_expected(x, platform=platform))

  def test_multi_platform_nested(self):
    x = np.arange(5, dtype=np.float32)
    exp = get_exported(lambda x: _testing_multi_platform_func(jnp.sin(x)),
                        lowering_platforms=("cpu", "tpu", "cuda"))(x)
    self.assertEqual(exp.lowering_platforms, ("cpu", "tpu", "cuda"))

    # Now serialize the call to the exported using a different sequence of
    # lowering platforms, but included in the lowering platforms for the
    # nested exported.
    exp2 = get_exported(export.call_exported(exp),
                         lowering_platforms=("cpu", "cuda"))(x)

    # Ensure that we do not have multiple lowerings of the exported function
    exp2_module_str = str(exp2.mlir_module())
    count_sine = len(re.findall("stablehlo.sine", exp2_module_str))
    self.assertEqual(1, count_sine)

    # Call with argument placed on different plaforms
    for platform in self.__class__.platforms:
      if platform == "tpu": continue
      x_device = jax.device_put(x, jax.devices(platform)[0])
      res_exp = export.call_exported(exp2)(x_device)
      self.assertAllClose(
        res_exp,
        _testing_multi_platform_fun_expected(np.sin(x), platform=platform))

  def test_multi_platform_nested_inside_single_platform_export(self):
    x = np.arange(5, dtype=np.float32)
    exp = get_exported(_testing_multi_platform_func,
                        lowering_platforms=("cpu", "tpu", "cuda"))(x)
    self.assertEqual(exp.lowering_platforms, ("cpu", "tpu", "cuda"))

    # Now serialize the call for the current platform.
    exp2 = get_exported(export.call_exported(exp))(x)
    module_str = str(exp2.mlir_module())
    self.assertIn("jax.uses_shape_polymorphism = true",
                  module_str)
    res2 = export.call_exported(exp2)(x)
    self.assertAllClose(res2, _testing_multi_platform_fun_expected(x))

  def test_multi_platform_and_poly(self):
    if jtu.test_device_matches(["gpu"]):
      # The export is not applicable to GPU
      raise unittest.SkipTest("Not intended for running on GPU")
    exp = get_exported(lambda x: jnp.reshape(_testing_multi_platform_func(x), (-1,)),
                        lowering_platforms=("cpu", "tpu"))(
        jax.ShapeDtypeStruct(export.symbolic_shape("b1, b2"), np.float32)
    )
    x = np.arange(12, dtype=np.float32).reshape((3, 4))
    res = export.call_exported(exp)(x)
    self.assertAllClose(res, _testing_multi_platform_fun_expected(x).reshape((-1,)))
    # Now serialize the call to the exported
    exp2 = get_exported(export.call_exported(exp))(x)
    res2 = export.call_exported(exp2)(x)
    self.assertAllClose(res2, _testing_multi_platform_fun_expected(x).reshape((-1,)))

  def test_multi_platform_and_sharding(self):
    export_devices = jax.devices()[0:2]
    export_mesh = Mesh(export_devices, axis_names=("x",))
    a = np.arange(16 * 4, dtype=np.float32).reshape((16, 4))
    @functools.partial(
        jax.jit,
        in_shardings=(jax.sharding.NamedSharding(export_mesh, P("x", None),),),
        out_shardings=jax.sharding.NamedSharding(export_mesh, P(None, "x")))
    def f_jax(b):  # b: f32[16 // DEVICES, 4]
      return b * 2.

    res_native = f_jax(a)
    exp = get_exported(f_jax,
                        lowering_platforms=("cpu", "tpu", "cuda"))(a)

    # Call with argument placed on different plaforms
    for platform in self.__class__.platforms:
      run_devices = jax.devices(platform)[0:len(export_devices)]
      if len(run_devices) != len(export_devices):
        continue
      run_mesh = Mesh(run_devices, ("x",))
      a_device = jax.device_put(a, jax.sharding.NamedSharding(run_mesh, None))
      res_exp = export.call_exported(exp)(a_device)
      self.assertArraysAllClose(res_native, res_exp)

  @jtu.parameterized_filterable(
    kwargs=[
      dict(v=v)
      for v in range(export.minimum_supported_serialization_version,
                     export.maximum_supported_serialization_version + 1)])
  def test_ordered_effects_basic(self, *, v: int):
    self.override_serialization_version(v)
    x = np.arange(3, dtype=np.float32)
    def f_jax(x):  # x: f32[3]
      # Test also the calling convention for inner functions
      def f_jax_inner(x):
        return (
          testing_primitive_with_effect_p.bind(x, effect_class_name="ForTestingOrderedEffect2") +
          testing_primitive_with_effect_p.bind(x, effect_class_name="ForTestingUnorderedEffect1"))
      return (
        10. +
        jax.jit(f_jax_inner)(x) +
        testing_primitive_with_effect_p.bind(x, effect_class_name="ForTestingOrderedEffect1") +
        testing_primitive_with_effect_p.bind(x, effect_class_name="ForTestingOrderedEffect2")
      )

    exp = get_exported(f_jax)(x)
    if exp.mlir_module_serialization_version >= _export._VERSION_START_SUPPORT_EFFECTS_WITH_REAL_TOKENS:
      self.assertEqual(["ForTestingOrderedEffect1()", "ForTestingOrderedEffect2()"],
                       sorted(str(e) for e in exp.ordered_effects))
      self.assertEqual(["ForTestingUnorderedEffect1()"],
                       [str(e) for e in exp.unordered_effects])
    else:
      self.assertEqual([], [str(e) for e in exp.ordered_effects])
      self.assertEqual([], [str(e) for e in exp.unordered_effects])
    mlir_module_str = str(exp.mlir_module())

    # Inner functions use stablehlo.token for all versions
    inner_fun_expected_re = (
      r"func.func private @f_jax_inner\("
      r"%arg0: !stablehlo.token .*jax.token = true.*"
      r"%arg1: tensor<3xf32>.*->.*"
      # Results
      r"!stablehlo.token .*jax.token = true.*"
      r"tensor<3xf32>"
    )
    self.assertRegex(mlir_module_str, inner_fun_expected_re)

    # The wrapped_main function takens tokens after version 9, and takes
    # i1[0] before version 9.
    wrapped_main_expected_re = (
      r"@_wrapped_jax_export_main\("
      r"%arg0: !stablehlo.token .*jax.token = true.*"
      r"%arg1: !stablehlo.token .*jax.token = true.*->.*"
      # Results
      r"!stablehlo.token .*jax.token = true.*"
      r"!stablehlo.token .*jax.token = true.*")
    if exp.mlir_module_serialization_version < _export._VERSION_START_SUPPORT_EFFECTS_WITH_REAL_TOKENS:
      wrapped_main_expected_re = wrapped_main_expected_re.replace("!stablehlo.token", "tensor<0xi1>")
    self.assertRegex(mlir_module_str, wrapped_main_expected_re)

    if exp.mlir_module_serialization_version < _export._VERSION_START_SUPPORT_EFFECTS_WITH_REAL_TOKENS:
      # The main function does not have tokens
      self.assertNotRegex(mlir_module_str, r"@main.*token")
    else:
      # The main function takes tokens and has the same type as the wrapped main
      main_expected_re = wrapped_main_expected_re.replace("@_wrapped_jax_export_main", "@main")
      self.assertRegex(mlir_module_str, main_expected_re)

    # Now call the exported from a function that uses its own effects
    def f_outer(x):
      return (
        testing_primitive_with_effect_p.bind(
          x, effect_class_name="ForTestingOrderedEffect2") +
        testing_primitive_with_effect_p.bind(
          x, effect_class_name="ForTestingUnorderedEffect1") +
        export.call_exported(exp)(x))

    lowered_outer = jax.jit(f_outer).lower(x)
    if exp.mlir_module_serialization_version < _export._VERSION_START_SUPPORT_EFFECTS_WITH_REAL_TOKENS:
      self.assertEqual(["ForTestingOrderedEffect2()"],
                       [str(e) for e in lowered_outer._lowering.compile_args["ordered_effects"]])
    else:
      self.assertEqual(["ForTestingOrderedEffect1()", "ForTestingOrderedEffect2()"],
                       sorted(str(e) for e in lowered_outer._lowering.compile_args["ordered_effects"]))
    self.assertEqual(["ForTestingUnorderedEffect1()"],
                     sorted([str(e) for e in lowered_outer._lowering.compile_args["unordered_effects"]]))

    mlir_outer_module_str = str(lowered_outer.compiler_ir())
    if exp.mlir_module_serialization_version >= _export._VERSION_START_SUPPORT_EFFECTS_WITH_REAL_TOKENS:
      main_expected_re = main_expected_re.replace("!stablehlo.token", "tensor<0xi1>")
      self.assertRegex(mlir_outer_module_str, main_expected_re)

    res = jax.jit(f_outer)(x)
    self.assertAllClose(2. * 2. * x + 10. + 4. * 2. * x, res)

  @jtu.parameterized_filterable(
    kwargs=[
      dict(v=v)
      for v in range(export.minimum_supported_serialization_version,
                     export.maximum_supported_serialization_version + 1)])
  def test_ordered_effects_poly(self, *, v: int):
    self.override_serialization_version(v)
    x = np.arange(12, dtype=np.float32).reshape((3, 4))
    def f_jax(x):  # x: f32[b1, b2]
      return 10. + testing_primitive_with_effect_p.bind(x, effect_class_name="ForTestingOrderedEffect1")
    exp = get_exported(f_jax)(jax.ShapeDtypeStruct(
        export.symbolic_shape("b2, b1"), x.dtype))
    mlir_module_str = str(exp.mlir_module())
    wrapped_main_expected_re = (
      r"@_wrapped_jax_export_main\("
      r"%arg0: tensor<i..> {jax.global_constant = \"b1\"}.*, "
      r"%arg1: tensor<i..> {jax.global_constant = \"b2\"}.*, "
      r"%arg2: !stablehlo.token {jax.token = true}.*, "
      r"%arg3: tensor<\?x\?xf32>.*\) -> \("
      # Results
      r"!stablehlo.token {jax.token = true}, tensor<\?x\?xf32>.*\)")
    if exp.mlir_module_serialization_version < _export._VERSION_START_SUPPORT_EFFECTS_WITH_REAL_TOKENS:
      wrapped_main_expected_re = wrapped_main_expected_re.replace("!stablehlo.token", "tensor<0xi1>")
    self.assertRegex(mlir_module_str, wrapped_main_expected_re)

    if exp.mlir_module_serialization_version < _export._VERSION_START_SUPPORT_EFFECTS_WITH_REAL_TOKENS:
      # The main function does not have tokens
      self.assertNotRegex(mlir_module_str, r"@main.*token")
    else:
      main_expected_re = (
        r"@main\("
        r"%arg0: !stablehlo.token {jax.token = true}.*, "
        r"%arg1: tensor<\?x\?xf32>.*\) -> \("
        # Results
        r"!stablehlo.token {jax.token = true}, tensor<\?x\?xf32>.*\)")
      self.assertRegex(mlir_module_str, main_expected_re)

    res = export.call_exported(exp)(x)
    self.assertAllClose(10. + 2. * x, res)

  @jtu.parameterized_filterable(
    kwargs=[
      dict(v=v)
      for v in range(export.minimum_supported_serialization_version,
                     export.maximum_supported_serialization_version + 1)])
  def test_ordered_effects_multi_platform_and_poly(self, *, v: int):
    self.override_serialization_version(v)
    if jtu.device_under_test() == "gpu":
      # The export is not applicable to GPU
      raise unittest.SkipTest("Not intended for running on GPU")
    x = np.ones((3, 4), dtype=np.float32)
    def f_jax(x):  # x: f32[b1, b2]
      return 10. + _testing_multi_platform_func(x,
                                                effect_class_name="ForTestingOrderedEffect1")
    exp = get_exported(
        f_jax,
        lowering_platforms=("cpu", "tpu")
        )(jax.ShapeDtypeStruct(export.symbolic_shape("b1, b2"), x.dtype))
    mlir_module_str = str(exp.mlir_module())
    wrapped_main_expected_re = (
      r"@_wrapped_jax_export_main\("
      r"%arg0: tensor<i..> {jax.global_constant = \"_platform_index\"}.*, "
      r"%arg1: tensor<i..> {jax.global_constant = \"b1\"}.*, "
      r"%arg2: tensor<i..> {jax.global_constant = \"b2\"}.*, "
      r"%arg3: !stablehlo.token {jax.token = true}.*, "
      r"%arg4: tensor<\?x\?xf32>.*\) -> \("
      # Results
      r"!stablehlo.token {jax.token = true}, tensor<\?x\?xf32>.*\)")
    if exp.mlir_module_serialization_version < _export._VERSION_START_SUPPORT_EFFECTS_WITH_REAL_TOKENS:
      wrapped_main_expected_re = wrapped_main_expected_re.replace("!stablehlo.token", "tensor<0xi1>")
    self.assertRegex(mlir_module_str, wrapped_main_expected_re)

    if exp.mlir_module_serialization_version < _export._VERSION_START_SUPPORT_EFFECTS_WITH_REAL_TOKENS:
      # The main function does not have tokens
      self.assertNotRegex(mlir_module_str, r"@main.*token")
    else:
      main_expected_re = (
        r"@main\("
        r"%arg0: tensor<i..> {jax.global_constant = \"_platform_index\"}.*, "
        r"%arg1: !stablehlo.token {jax.token = true}.*, "
        r"%arg2: tensor<\?x\?xf32>.*\) -> \("
        # Results
        r"!stablehlo.token {jax.token = true}, tensor<\?x\?xf32>.*\)")
      self.assertRegex(mlir_module_str, main_expected_re)
    res = export.call_exported(exp)(x)
    self.assertAllClose(10. + _testing_multi_platform_fun_expected(x),
                        res)

  @jtu.parameterized_filterable(
    kwargs=[
      dict(v=v)
      for v in range(export.minimum_supported_serialization_version,
                     export.maximum_supported_serialization_version + 1)])
  def test_ordered_effects_with_donation(self, *, v: int):
    self.override_serialization_version(v)
    x = np.arange(3, dtype=np.float32)

    def f_jax(x):
      return testing_primitive_with_effect_p.bind(
          x, effect_class_name="ForTestingOrderedEffect1"
      )

    f_jax = jax.jit(f_jax, donate_argnums=(0,))
    exp = export.export(f_jax)(x)
    mlir_module_str = str(exp.mlir_module())
    if exp.mlir_module_serialization_version < _export._VERSION_START_SUPPORT_EFFECTS_WITH_REAL_TOKENS:
      self.assertRegex(mlir_module_str, r"@main.*tf.aliasing_output = 0")
      self.assertRegex(mlir_module_str, r"@_wrapped_jax_export_main.*tf.aliasing_output = 1")
    else:
      self.assertRegex(mlir_module_str, r"@main.*tf.aliasing_output = 1")
      self.assertRegex(mlir_module_str, r"@_wrapped_jax_export_main.*tf.aliasing_output = 1")

  @jtu.parameterized_filterable(
    kwargs=[
      dict(name=name, expect_error=expect_error)
      # name is the suffix for event name: ForTestingOrderedEffectxxx
      for name, expect_error in (
        ("4NoNullary", "must have a nullary constructor"),
        ("5NoEq", "must have a nullary class constructor that produces an "
                  "equal effect object"),
      )
    ])
  def test_ordered_effects_error(self, *, name: str, expect_error: str):
    x = np.ones((3, 4), dtype=np.float32)
    def f_jax(x):
      return 10. + _testing_multi_platform_func(
        x,
        effect_class_name="ForTestingOrderedEffect" + name)
    with self.assertRaisesRegex(Exception, expect_error):
      _ = get_exported(f_jax)(jax.ShapeDtypeStruct((3, 4), x.dtype))

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
