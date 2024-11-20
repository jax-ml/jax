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
from functools import partial

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import lax
import jax.extend as jex
import jax.numpy as jnp
import jax.sharding as shd

from jax._src import abstract_arrays
from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import linear_util
from jax._src import prng
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.interpreters import mlir
from jax._src.layout import DeviceLocalLayout
from jax._src.lib import lapack
from jax._src.lib.mlir.dialects import hlo
from jax._src.lax import linalg as lax_linalg_internal
from jax.experimental.shard_map import shard_map

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

  def make_custom_impl(self, shape, seed=False, split=False, fold_in=False,
                       random_bits=False):
    assert not split and not fold_in and not random_bits  # not yet implemented
    def seed_rule(_):
      return jnp.ones(shape, dtype=jnp.dtype('uint32'))

    def no_rule(*args, **kwargs):
      assert False, 'unreachable'

    return jex.random.define_prng_impl(
        key_shape=shape, seed=seed_rule if seed else no_rule, split=no_rule,
        fold_in=no_rule, random_bits=no_rule)

  def test_key_make_with_custom_impl(self):
    impl = self.make_custom_impl(shape=(4, 2, 7), seed=True)
    k = jax.random.key(42, impl=impl)
    self.assertEqual(k.shape, ())
    self.assertEqual(impl, jax.random.key_impl(k))

  def test_key_wrap_with_custom_impl(self):
    shape = (4, 2, 7)
    impl = self.make_custom_impl(shape=shape)
    data = jnp.ones((3, *shape), dtype=jnp.dtype('uint32'))
    k = jax.random.wrap_key_data(data, impl=impl)
    self.assertEqual(k.shape, (3,))
    self.assertEqual(impl, jax.random.key_impl(k))

  def test_key_impl_is_spec(self):
    # this is counterpart to random_test.py:
    # KeyArrayTest.test_key_impl_builtin_is_string_name
    spec_ref = self.make_custom_impl(shape=(4, 2, 7), seed=True)
    key = jax.random.key(42, impl=spec_ref)
    spec = jax.random.key_impl(key)
    self.assertEqual(repr(spec), f"PRNGSpec({spec_ref._impl.name!r})")


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
      return jex.ffi.ffi_call("test_ffi", x)(x, param=param)

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
      return jex.ffi.ffi_call("test_ffi", core.abstract_token)(token)

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
      jex.ffi.ffi_call(target_name, (), has_side_effect=True)()
    hlo = jax.jit(fun).lower()
    self.assertIn(target_name, hlo.as_text())
    self.assertIn("has_side_effect = true", hlo.as_text())
    self.assertIn(target_name, hlo.compile().as_text())

  def testJvpError(self):
    def fun(x):
      return jex.ffi.ffi_call("test_ffi", x)(x, non_hashable_arg={"a": 1})
    with self.assertRaisesRegex(
        ValueError, "The FFI call to `.+` cannot be differentiated."):
      jax.jvp(fun, (0.5,), (0.5,))

  def testNonHashableAttributes(self):
    def fun(x):
      return jex.ffi.ffi_call("test_ffi", x)(x, non_hashable_arg={"a": 1})

    self.assertIn("HashableDict", str(jax.make_jaxpr(fun)(jnp.ones(5))))
    hlo = jax.jit(fun).lower(jnp.ones(5)).as_text()
    self.assertIn("non_hashable_arg = {a = 1", hlo)

    # If non-hashable arguments aren't handled properly, this will raise a
    # TypeError. We make sure it doesn't.
    with self.assertRaises(Exception) as manager:
      fun(jnp.ones(5))
    self.assertNotIsInstance(manager.exception, TypeError)

    def fun(x):
      return jex.ffi.ffi_call("test_ffi", x)(x, non_hashable_arg=np.arange(3))
    self.assertIn("HashableArray", str(jax.make_jaxpr(fun)(jnp.ones(5))))
    hlo = jax.jit(fun).lower(jnp.ones(5)).as_text()
    self.assertIn("non_hashable_arg = array<i64: 0, 1, 2>", hlo)
    with self.assertRaises(Exception) as manager:
      fun(jnp.ones(5))
    self.assertNotIsInstance(manager.exception, TypeError)

  @jtu.sample_product(shape=[(6, 5), (4, 5, 6)])
  @jtu.run_on_devices("gpu", "cpu")
  def testFfiCall(self, shape):
    x = self.rng().randn(*shape).astype(np.float32)
    expected = lax_linalg_internal.geqrf(x)
    actual = ffi_call_geqrf(x)
    for a, b in zip(actual, expected):
      self.assertArraysEqual(a, b)

  @jtu.sample_product(
      shape=[(6, 5), (4, 5, 6)],
      vmap_method=["expand_dims", "broadcast_all", "sequential"],
  )
  @jtu.run_on_devices("gpu", "cpu")
  def testFfiCallBatching(self, shape, vmap_method):
    shape = (10,) + shape
    x = self.rng().randn(*shape).astype(np.float32)
    expected = lax_linalg_internal.geqrf(x)
    actual = jax.vmap(partial(ffi_call_geqrf, vmap_method=vmap_method))(x)
    for a, b in zip(actual, expected):
      if vmap_method == "sequential" and len(shape) == 3:
        # On GPU, the batched FFI call to geqrf uses an algorithm with
        # different numerics than the unbatched version (which is used when
        # vmap_method="sequential"). Therefore, we need to include floating
        # point tolerance for this check.
        self.assertArraysAllClose(a, b)
      else:
        self.assertArraysEqual(a, b)

  @jtu.run_on_devices("gpu", "cpu")
  def testVectorizedDeprecation(self):
    x = self.rng().randn(3, 5, 4).astype(np.float32)
    with self.assertWarns(DeprecationWarning):
      ffi_call_geqrf(x, vectorized=True)
    with self.assertWarns(DeprecationWarning):
      jax.vmap(ffi_call_geqrf)(x)

  def testBackwardCompatSyntax(self):
    def fun(x):
      return jex.ffi.ffi_call("test_ffi", x, x, param=0.5)
    with self.assertWarns(DeprecationWarning):
      jax.jit(fun).lower(jnp.ones(5))

  def testInputOutputAliases(self):
    def fun(x):
      return jex.ffi.ffi_call("test", x, input_output_aliases={0: 0})(x)
    hlo = jax.jit(fun).lower(jnp.ones(5)).as_text()
    self.assertRegex(hlo, r"output_operand_aliases = \[.*operand_index = 0.*\]")

  def testInvalidInputOutputAliases(self):
    def fun(x):
      return jex.ffi.ffi_call("test", x, input_output_aliases={1: 0})(x)
    with self.assertRaisesRegex(ValueError, "with input index"):
      jax.jit(fun).lower(jnp.ones(5)).as_text()

    def fun(x):
      return jex.ffi.ffi_call("test", x, input_output_aliases={0: 1})(x)
    with self.assertRaisesRegex(ValueError, "with output index"):
      jax.jit(fun).lower(jnp.ones(5)).as_text()

    def fun(x):
      return jex.ffi.ffi_call("test", jax.ShapeDtypeStruct(x.shape, np.int32),
                              input_output_aliases={0: 0})(x)
    with self.assertRaisesRegex(ValueError,
                                "referring to an input with abstract value"):
      jax.jit(fun).lower(jnp.ones(5)).as_text()

    def fun(x):
      return jex.ffi.ffi_call("test", jax.ShapeDtypeStruct(x.shape + x.shape,
                                                           x.dtype),
                              input_output_aliases={0: 0})(x)
    with self.assertRaisesRegex(ValueError,
                                "referring to an input with abstract value"):
      jax.jit(fun).lower(jnp.ones(5)).as_text()

  def testLegacyBackendConfig(self):
    def fun(x):
      return jex.ffi.ffi_call("test", x, custom_call_api_version=2,
                              legacy_backend_config="12345")(x)
    hlo = jax.jit(fun).lower(jnp.ones(5)).as_text()
    self.assertRegex(hlo, 'backend_config = "12345"')

  def testInvalidBackendConfig(self):
    def fun(x):
      return jex.ffi.ffi_call("test", x, legacy_backend_config="12345")(x)
    with self.assertRaisesRegex(ValueError,
                                "The use of the legacy_backend_config"):
      jax.jit(fun).lower(jnp.ones(5)).as_text()

    def fun(x):
      return jex.ffi.ffi_call("test", x,
                              custom_call_api_version=2)(x, attribute=1)
    with self.assertRaisesRegex(ValueError,
                                "The use of ffi_call attributes requires"):
      jax.jit(fun).lower(jnp.ones(5)).as_text()

  def testAllow64(self):
    if config.enable_x64.value:
      self.skipTest("Requires enable_x64=False")
    def fun():
      return jex.ffi.ffi_call("test", jax.ShapeDtypeStruct((), np.int64))()
    self.assertIn("tensor<i64>", jax.jit(fun).lower().as_text())

  def testInvalidResultType(self):
    with self.assertRaisesRegex(
        ValueError, "All elements of result_shape_dtypes.*position 0"):
      jex.ffi.ffi_call("test", None)()
    with self.assertRaisesRegex(
        ValueError, "All elements of result_shape_dtypes.*position 1"):
      jex.ffi.ffi_call("test", (jax.ShapeDtypeStruct((), np.float32), ()))()

  @jtu.run_on_devices("gpu", "cpu")
  def testShardMap(self):
    mesh = jtu.create_mesh((1,), ("i",))
    x = self.rng().randn(8, 4, 5).astype(np.float32)

    @partial(shard_map, mesh=mesh, in_specs=shd.PartitionSpec('i'),
             out_specs=shd.PartitionSpec('i'))
    def f(x):
      return ffi_call_geqrf(x)

    f(x)  # eager mode doesn't crash
    jax.jit(f)(x)  # neither does JIT
    self.assertNotIn("all-gather", jax.jit(f).lower(x).compile().as_text())


def ffi_call_geqrf(x, **kwargs):
  if jtu.test_device_matches(["cpu"]):
    lapack._lapack.initialize()

  assert x.dtype == np.float32
  ndim = x.ndim
  x_major_to_minor = tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
  output_types = [
      x, jax.ShapeDtypeStruct(x.shape[:-2] + (min(*x.shape[-2:]),), x.dtype)]

  def call(platform, x):
    target_name = dict(
        cpu="lapack_sgeqrf_ffi",
        rocm="hipsolver_geqrf_ffi",
        cuda="cusolver_geqrf_ffi",
    )[platform]
    return jex.ffi.ffi_call(
        target_name, output_types, input_output_aliases={0: 0},
        input_layouts=[x_major_to_minor],
        output_layouts=[x_major_to_minor, None],
        **kwargs)(x)

  return lax.platform_dependent(
      x, cpu=partial(call, "cpu"), rocm=partial(call, "rocm"),
      cuda=partial(call, "cuda"))


class MlirRegisterLoweringTest(jtu.JaxTestCase):

  def test_unknown_platform_error(self):
    with self.assertRaisesRegex(
        NotImplementedError,
        "Registering an MLIR lowering rule for primitive .+ for an unknown "
        "platform foo. Known platforms are: .+."):
      mlir.register_lowering(prim=None, rule=None, platform="foo")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
