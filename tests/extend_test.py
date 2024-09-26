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

import os
import unittest

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import lax
import jax.extend as jex
import jax.numpy as jnp

from jax._src import abstract_arrays
from jax._src import api
from jax._src import core
from jax._src import linear_util
from jax._src import prng
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.interpreters import mlir
from jax._src.layout import DeviceLocalLayout
from jax._src.lib.mlir.dialects import hlo

jax.config.parse_flags_with_absl()


class ExtendTest(jtu.JaxTestCase):

  def test_symbols(self):
    # Assume these are tested in random_test.py, only check equivalence
    self.assertIs(jex.random.seed_with_impl, prng.seed_with_impl)
    self.assertIs(jex.random.threefry2x32_p, prng.threefry2x32_p)
    self.assertIs(jex.random.threefry_2x32, prng.threefry_2x32)
    self.assertIs(jex.random.threefry_prng_impl, prng.threefry_prng_impl)
    self.assertIs(jex.random.rbg_prng_impl, prng.rbg_prng_impl)
    self.assertIs(jex.random.unsafe_rbg_prng_impl, prng.unsafe_rbg_prng_impl)

    # Assume these are tested elsewhere, only check equivalence
    self.assertIs(jex.backend.backends, xla_bridge.backends)
    self.assertIs(jex.backend.backend_xla_version, xla_bridge.backend_xla_version)
    self.assertIs(jex.backend.clear_backends, api.clear_backends)
    self.assertIs(jex.backend.get_backend, xla_bridge.get_backend)
    self.assertIs(jex.backend.register_backend_factory, xla_bridge.register_backend_factory)
    self.assertIs(jex.core.array_types, abstract_arrays.array_types)
    self.assertIs(jex.linear_util.StoreException, linear_util.StoreException)
    self.assertIs(jex.linear_util.WrappedFun, linear_util.WrappedFun)
    self.assertIs(jex.linear_util.cache, linear_util.cache)
    self.assertIs(jex.linear_util.merge_linear_aux, linear_util.merge_linear_aux)
    self.assertIs(jex.linear_util.transformation, linear_util.transformation)
    self.assertIs(jex.linear_util.transformation_with_aux, linear_util.transformation_with_aux)
    self.assertIs(jex.linear_util.wrap_init, linear_util.wrap_init)


class RandomTest(jtu.JaxTestCase):

  def test_key_make_with_custom_impl(self):
    shape = (4, 2, 7)

    def seed_rule(_):
      return jnp.ones(shape, dtype=jnp.dtype('uint32'))

    def no_rule(*args, **kwargs):
      assert False, 'unreachable'

    impl = jex.random.define_prng_impl(
        key_shape=shape, seed=seed_rule, split=no_rule, fold_in=no_rule,
        random_bits=no_rule)
    k = jax.random.key(42, impl=impl)
    self.assertEqual(k.shape, ())
    self.assertEqual(impl, jax.random.key_impl(k))

  def test_key_wrap_with_custom_impl(self):
    def no_rule(*args, **kwargs):
      assert False, 'unreachable'

    shape = (4, 2, 7)
    impl = jex.random.define_prng_impl(
        key_shape=shape, seed=no_rule, split=no_rule, fold_in=no_rule,
        random_bits=no_rule)
    data = jnp.ones((3, *shape), dtype=jnp.dtype('uint32'))
    k = jax.random.wrap_key_data(data, impl=impl)
    self.assertEqual(k.shape, (3,))
    self.assertEqual(impl, jax.random.key_impl(k))


class FfiTest(jtu.JaxTestCase):

  def find_custom_call_in_module(self, module):
    for func in module.body.operations:
      for block in func.body.blocks:
        for op in block.operations:
          if op.OPERATION_NAME == "stablehlo.custom_call":
            return op
    self.fail("No custom_call found in the lowered IR")

  def testHeadersExist(self):
    base_dir = os.path.join(jex.ffi.include_dir(), "xla", "ffi", "api")
    for header in ["c_api.h", "api.h", "ffi.h"]:
      self.assertTrue(os.path.exists(os.path.join(base_dir, header)))

  @parameterized.parameters([
    (tuple(range(3)), tuple(range(3))),
    (None, tuple(reversed(range(3)))),
    (DeviceLocalLayout(tuple(range(3))), tuple(reversed(range(3)))),
  ])
  def testLoweringLayouts(self, layout_spec, expected_layout):
    # Regression test to ensure that the lowering rule properly captures
    # layouts.
    def lowering_rule(ctx, x):
      aval, = ctx.avals_in
      ndim = len(aval.shape)
      return jex.ffi.ffi_lowering("test_ffi", operand_layouts=[layout_spec],
                                  result_layouts=[layout_spec])(ctx, x)
    prim = core.Primitive("test_ffi")
    prim.def_impl(lambda x: x)
    prim.def_abstract_eval(lambda x: x)
    mlir.register_lowering(prim, lowering_rule)

    x = jnp.ones((3,) * len(expected_layout))
    lowered = jax.jit(prim.bind).lower(x)
    module = lowered.compiler_ir("stablehlo")
    op = self.find_custom_call_in_module(module)
    self.assertIn("operand_layouts", op.attributes)
    self.assertIn("result_layouts", op.attributes)

    text = lowered.as_text()
    expected = ", ".join(map(str, expected_layout))
    pattern = rf"operand_layouts = \[dense<\[{expected}\]>"
    self.assertRegex(text, pattern)
    pattern = rf"result_layouts = \[dense<\[{expected}\]>"
    self.assertRegex(text, pattern)

  @parameterized.parameters([
      (True, mlir.ir.BoolAttr.get),
      (1, mlir.i64_attr),
      (5.0, lambda x: mlir.ir.FloatAttr.get(mlir.ir.F64Type.get(), x)),
      ("param", mlir.ir.StringAttr.get),
      (np.float32(0.5),
       lambda x: mlir.ir.FloatAttr.get(mlir.ir.F32Type.get(), x)),
  ])
  def testParams(self, param, expected_builder):
    def fun(x):
      return jex.ffi.ffi_call("test_ffi", x, x, param=param)

    # Here we inspect the lowered IR to test that the parameter has been
    # serialized with the appropriate type.
    module = jax.jit(fun).lower(0.5).compiler_ir("stablehlo")
    op = self.find_custom_call_in_module(module)
    config = op.attributes["mhlo.backend_config"]
    self.assertIsInstance(config, mlir.ir.DictAttr)
    self.assertIn("param", config)
    with mlir.make_ir_context(), mlir.ir.Location.unknown():
      expected = expected_builder(param)
    self.assertEqual(type(config["param"]), type(expected))
    self.assertTrue(expected.type.isinstance(config["param"].type))

  def testToken(self):
    def fun():
      token = lax.create_token()
      return jex.ffi.ffi_call("test_ffi", core.abstract_token, token)

    # Ensure that token inputs and outputs are translated to the correct type
    module = jax.jit(fun).lower().compiler_ir("stablehlo")
    op = self.find_custom_call_in_module(module)
    self.assertTrue(hlo.TokenType.isinstance(op.operands[0].type))
    self.assertTrue(hlo.TokenType.isinstance(op.results[0].type))

  def testEffectsHlo(self):
    # The target name must exist on the current platform, but we don't actually
    # need to call it with the correct syntax, because we're only checking the
    # compiled HLO.
    if jtu.test_device_matches(["cpu"]):
      target_name = "lapack_sgetrf_ffi"
    elif jtu.test_device_matches(["rocm"]):
      target_name = "hipsolver_getrf_ffi"
    elif jtu.test_device_matches(["cuda", "gpu"]):
      target_name = "cusolver_getrf_ffi"
    else:
      raise unittest.SkipTest("Unsupported device")
    def fun():
      jex.ffi.ffi_call(target_name, (), has_side_effect=True)
    hlo = jax.jit(fun).lower()
    self.assertIn(target_name, hlo.as_text())
    self.assertIn("has_side_effect = true", hlo.as_text())
    self.assertIn(target_name, hlo.compile().as_text())

  def testJvpError(self):
    def fun(x):
      return jex.ffi.ffi_call("test_ffi", x, x, non_hashable_arg={"a": 1})
    with self.assertRaisesRegex(
        ValueError, "The FFI call to `.+` cannot be differentiated."):
      jax.jvp(fun, (0.5,), (0.5,))

  def testNonHashableAttributes(self):
    def fun(x):
      return jex.ffi.ffi_call("test_ffi", x, x, non_hashable_arg={"a": 1})

    self.assertIn("HashableDict", str(jax.make_jaxpr(fun)(jnp.ones(5))))
    hlo = jax.jit(fun).lower(jnp.ones(5)).as_text()
    self.assertIn("non_hashable_arg = {a = 1", hlo)

    # If non-hashable arguments aren't handled properly, this will raise a
    # TypeError. We make sure it doesn't.
    with self.assertRaises(Exception) as manager:
      fun(jnp.ones(5))
    self.assertNotIsInstance(manager.exception, TypeError)

    def fun(x):
      return jex.ffi.ffi_call("test_ffi", x, x, non_hashable_arg=np.arange(3))
    self.assertIn("HashableArray", str(jax.make_jaxpr(fun)(jnp.ones(5))))
    hlo = jax.jit(fun).lower(jnp.ones(5)).as_text()
    self.assertIn("non_hashable_arg = array<i64: 0, 1, 2>", hlo)
    with self.assertRaises(Exception) as manager:
      fun(jnp.ones(5))
    self.assertNotIsInstance(manager.exception, TypeError)

  @jtu.sample_product(
    shape=[(1,), (4,), (5,)],
    dtype=(np.int32,),
  )
  @jtu.run_on_devices("gpu")
  def testFfiCall(self, shape, dtype):
    pivots_size = shape[-1]
    permutation_size = 2 * pivots_size
    pivots = jnp.arange(permutation_size - 1, pivots_size - 1, -1, dtype=dtype)
    pivots = jnp.broadcast_to(pivots, shape)
    expected = lax.linalg.lu_pivots_to_permutation(pivots, permutation_size)
    actual = ffi_call_lu_pivots_to_permutation(pivots, permutation_size)
    self.assertArraysEqual(actual, expected)

  @jtu.sample_product(
      shape=[(1,), (4,), (5,)],
      dtype=(np.int32,),
      vectorized=(False, True),
  )
  @jtu.run_on_devices("gpu")
  def testFfiCallBatching(self, shape, dtype, vectorized):
    shape = (10,) + shape
    pivots_size = shape[-1]
    permutation_size = 2 * pivots_size
    pivots = jnp.arange(permutation_size - 1, pivots_size - 1, -1, dtype=dtype)
    pivots = jnp.broadcast_to(pivots, shape)
    expected = lax.linalg.lu_pivots_to_permutation(pivots, permutation_size)
    actual = jax.vmap(lambda x: ffi_call_lu_pivots_to_permutation(
        x, permutation_size, vectorized=vectorized))(pivots)
    self.assertArraysEqual(actual, expected)


# TODO(dfm): For now this test uses the `cu_lu_pivots_to_permutation`
# custom call target because that's the only one in jaxlib that uses the
# new FFI interface. Once more are available, consider using something that
# can be run on multiple platforms.
def ffi_call_lu_pivots_to_permutation(pivots, permutation_size, vectorized=True):
  return jex.ffi.ffi_call(
      "cu_lu_pivots_to_permutation",
      jax.ShapeDtypeStruct(
          shape=pivots.shape[:-1] + (permutation_size,),
          dtype=pivots.dtype,
      ),
      pivots,
      vectorized=vectorized,
  )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
