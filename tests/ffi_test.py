# Copyright 2024 The JAX Authors.
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
from jax.sharding import PartitionSpec as P

from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.layout import DeviceLocalLayout
from jax._src.lib import lapack, xla_extension_version
from jax._src.lib.mlir.dialects import hlo
from jax._src.lax import linalg as lax_linalg_internal
from jax.experimental.shard_map import shard_map

jax.config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


class FfiTest(jtu.JaxTestCase):

  def find_custom_call_in_module(self, module):
    for func in module.body.operations:
      for block in func.body.blocks:
        for op in block.operations:
          if op.OPERATION_NAME == "stablehlo.custom_call":
            return op
    self.fail("No custom_call found in the lowered IR")

  def test_headers_exist(self):
    base_dir = os.path.join(jax.ffi.include_dir(), "xla", "ffi", "api")
    for header in ["c_api.h", "api.h", "ffi.h"]:
      self.assertTrue(os.path.exists(os.path.join(base_dir, header)))

  @parameterized.parameters([
    (tuple(range(3)), tuple(range(3))),
    (None, tuple(reversed(range(3)))),
    (DeviceLocalLayout(tuple(range(3))), tuple(reversed(range(3)))),
  ])
  def test_lowering_layouts(self, layout_spec, expected_layout):
    # Regression test to ensure that the lowering rule properly captures
    # layouts.
    def lowering_rule(ctx, x):
      return jax.ffi.ffi_lowering("test_ffi", operand_layouts=[layout_spec],
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
  def test_params(self, param, expected_builder):
    def fun(x):
      return jax.ffi.ffi_call("test_ffi", x)(x, param=param)

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

  def test_token(self):
    def fun():
      token = lax.create_token()
      return jax.ffi.ffi_call("test_ffi", core.abstract_token)(token)

    # Ensure that token inputs and outputs are translated to the correct type
    module = jax.jit(fun).lower().compiler_ir("stablehlo")
    op = self.find_custom_call_in_module(module)
    self.assertTrue(hlo.TokenType.isinstance(op.operands[0].type))
    self.assertTrue(hlo.TokenType.isinstance(op.results[0].type))

  def test_effects_hlo(self):
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
      jax.ffi.ffi_call(target_name, (), has_side_effect=True)()
    hlo = jax.jit(fun).lower()
    self.assertIn(target_name, hlo.as_text())
    self.assertIn("has_side_effect = true", hlo.as_text())
    self.assertIn(target_name, hlo.compile().as_text())

  def test_jvp_error(self):
    def fun(x):
      return jax.ffi.ffi_call("test_ffi", x)(x, non_hashable_arg={"a": 1})
    with self.assertRaisesRegex(
        ValueError, "The FFI call to `.+` cannot be differentiated."):
      jax.jvp(fun, (0.5,), (0.5,))

  def test_non_hashable_attributes(self):
    def fun(x):
      return jax.ffi.ffi_call("test_ffi", x)(x, non_hashable_arg={"a": 1})

    self.assertIn("HashableDict", str(jax.make_jaxpr(fun)(jnp.ones(5))))
    hlo = jax.jit(fun).lower(jnp.ones(5)).as_text()
    self.assertIn("non_hashable_arg = {a = 1", hlo)

    # If non-hashable arguments aren't handled properly, this will raise a
    # TypeError. We make sure it doesn't.
    with self.assertRaises(Exception) as manager:
      fun(jnp.ones(5))
    self.assertNotIsInstance(manager.exception, TypeError)

    def fun(x):
      return jax.ffi.ffi_call("test_ffi", x)(x, non_hashable_arg=np.arange(3))
    self.assertIn("HashableArray", str(jax.make_jaxpr(fun)(jnp.ones(5))))
    hlo = jax.jit(fun).lower(jnp.ones(5)).as_text()
    self.assertIn("non_hashable_arg = dense<[0, 1, 2]> : tensor<3xi64>", hlo)
    with self.assertRaises(Exception) as manager:
      fun(jnp.ones(5))
    self.assertNotIsInstance(manager.exception, TypeError)

  @jtu.sample_product(shape=[(6, 5), (4, 5, 6)])
  @jtu.run_on_devices("gpu", "cpu")
  def test_ffi_call(self, shape):
    x = self.rng().randn(*shape).astype(np.float32)
    expected = lax_linalg_internal.geqrf(x)
    actual = ffi_call_geqrf(x)
    for a, b in zip(actual, expected):
      self.assertArraysEqual(a, b)

  @jtu.sample_product(
      shape=[(6, 5), (4, 5, 6)],
      vmap_method=["expand_dims", "broadcast_all", "sequential",
                   "sequential_unrolled"],
  )
  @jtu.run_on_devices("gpu", "cpu")
  def test_ffi_call_batching(self, shape, vmap_method):
    shape = (10,) + shape
    x = self.rng().randn(*shape).astype(np.float32)
    expected = lax_linalg_internal.geqrf(x)
    actual = jax.vmap(partial(ffi_call_geqrf, vmap_method=vmap_method))(x)
    for a, b in zip(actual, expected):
      if vmap_method.startswith("sequential") and len(shape) == 3:
        # On GPU, the batched FFI call to geqrf uses an algorithm with
        # different numerics than the unbatched version (which is used when
        # vmap_method="sequential"). Therefore, we need to include floating
        # point tolerance for this check.
        self.assertArraysAllClose(a, b)
      else:
        self.assertArraysEqual(a, b)

  @jtu.run_on_devices("gpu", "cpu")
  def test_vectorized_deprecation(self):
    x = self.rng().randn(3, 5, 4).astype(np.float32)
    with self.assertWarns(DeprecationWarning):
      ffi_call_geqrf(x, vectorized=True)
    with self.assertWarns(DeprecationWarning):
      jax.vmap(ffi_call_geqrf)(x)

  def test_backward_compat_syntax(self):
    def fun(x):
      return jax.ffi.ffi_call("test_ffi", x, x, param=0.5)
    msg = "Calling ffi_call directly with input arguments is deprecated"
    with self.assertDeprecationWarnsOrRaises("jax-ffi-call-args", msg):
      jax.jit(fun).lower(jnp.ones(5))

  def test_input_output_aliases(self):
    def fun(x):
      return jax.ffi.ffi_call("test", x, input_output_aliases={0: 0})(x)
    hlo = jax.jit(fun).lower(jnp.ones(5)).as_text()
    self.assertRegex(hlo, r"output_operand_aliases = \[.*operand_index = 0.*\]")

  def test_invalid_input_output_aliases(self):
    def fun(x):
      return jax.ffi.ffi_call("test", x, input_output_aliases={1: 0})(x)
    with self.assertRaisesRegex(ValueError, "with input index"):
      jax.jit(fun).lower(jnp.ones(5)).as_text()

    def fun(x):
      return jax.ffi.ffi_call("test", x, input_output_aliases={0: 1})(x)
    with self.assertRaisesRegex(ValueError, "with output index"):
      jax.jit(fun).lower(jnp.ones(5)).as_text()

    def fun(x):
      return jax.ffi.ffi_call("test", jax.ShapeDtypeStruct(x.shape, np.int32),
                              input_output_aliases={0: 0})(x)
    with self.assertRaisesRegex(ValueError,
                                "referring to an input with abstract value"):
      jax.jit(fun).lower(jnp.ones(5)).as_text()

    def fun(x):
      return jax.ffi.ffi_call("test", jax.ShapeDtypeStruct(x.shape + x.shape,
                                                           x.dtype),
                              input_output_aliases={0: 0})(x)
    with self.assertRaisesRegex(ValueError,
                                "referring to an input with abstract value"):
      jax.jit(fun).lower(jnp.ones(5)).as_text()

  def test_legacy_backend_config(self):
    def fun(x):
      return jax.ffi.ffi_call("test", x, custom_call_api_version=2,
                              legacy_backend_config="12345")(x)
    hlo = jax.jit(fun).lower(jnp.ones(5)).as_text()
    self.assertRegex(hlo, 'backend_config = "12345"')

  def test_invalid_backend_config(self):
    def fun(x):
      return jax.ffi.ffi_call("test", x, legacy_backend_config="12345")(x)
    with self.assertRaisesRegex(ValueError,
                                "The use of the legacy_backend_config"):
      jax.jit(fun).lower(jnp.ones(5)).as_text()

    def fun(x):
      return jax.ffi.ffi_call("test", x,
                              custom_call_api_version=2)(x, attribute=1)
    with self.assertRaisesRegex(ValueError,
                                "The use of ffi_call attributes requires"):
      jax.jit(fun).lower(jnp.ones(5)).as_text()

  def test_allow_x64(self):
    if config.enable_x64.value:
      self.skipTest("Requires enable_x64=False")
    def fun():
      return jax.ffi.ffi_call("test", jax.ShapeDtypeStruct((), np.int64))()
    self.assertIn("tensor<i64>", jax.jit(fun).lower().as_text())

  def test_invalid_result_type(self):
    with self.assertRaisesRegex(
        ValueError, "All elements of result_shape_dtypes.*position 0"):
      jax.ffi.ffi_call("test", None)()
    with self.assertRaisesRegex(
        ValueError, "All elements of result_shape_dtypes.*position 1"):
      jax.ffi.ffi_call("test", (jax.ShapeDtypeStruct((), np.float32), ()))()

  @jtu.run_on_devices("gpu", "cpu")
  def test_shard_map(self):
    mesh = jtu.create_mesh((len(jax.devices()),), ("i",))
    x = self.rng().randn(8, 4, 5).astype(np.float32)

    @partial(shard_map, mesh=mesh, in_specs=P("i"), out_specs=P("i"))
    def f(x):
      return ffi_call_geqrf(x)

    f(x)  # eager mode doesn't crash
    jax.jit(f)(x)  # neither does JIT
    self.assertNotIn("all-gather", jax.jit(f).lower(x).compile().as_text())

  @jtu.run_on_devices("gpu", "cpu")
  @jtu.ignore_warning(category=DeprecationWarning)
  def test_extend_import_shim(self):
    ffi_call_geqrf(jnp.ones((4, 5), dtype=np.float32), _use_extend=True)


def ffi_call_geqrf(x, _use_extend=False, **kwargs):
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
    f = jex.ffi.ffi_call if _use_extend else jax.ffi.ffi_call
    return f(
        target_name, output_types, input_output_aliases={0: 0},
        input_layouts=[x_major_to_minor],
        output_layouts=[x_major_to_minor, None],
        **kwargs)(x)

  return lax.platform_dependent(
      x, cpu=partial(call, "cpu"), rocm=partial(call, "rocm"),
      cuda=partial(call, "cuda"))


class BatchPartitioningTest(jtu.JaxTestCase):
  def setUp(self):
    super().setUp()
    if xla_extension_version < 313:
      self.skipTest("Requires XLA extension version >= 313")
    # Register callbacks before checking the number of devices to make sure
    # that we're testing the registration path, even if we can't run the tests.
    for target_name in ["lapack_sgeqrf_ffi", "cusolver_geqrf_ffi",
                        "hipsolver_geqrf_ffi"]:
      jax.ffi.register_ffi_target_as_batch_partitionable(target_name)
    if jax.device_count() < 2:
      self.skipTest("Requires multiple devices")
    if jtu.test_device_matches(["cpu"]):
      lapack._lapack.initialize()

  @jtu.run_on_devices("gpu", "cpu")
  def test_shard_map(self):
    mesh = jtu.create_mesh((len(jax.devices()),), ("i",))
    x = self.rng().randn(8, 4, 5).astype(np.float32)

    @partial(shard_map, mesh=mesh, in_specs=P("i"), out_specs=P("i"),
             check_rep=False)
    def f(x):
      return batch_partitionable_ffi_call(x)

    f(x)  # eager mode doesn't crash
    jax.jit(f)(x)  # neither does JIT
    self.assertNotIn("all-gather", jax.jit(f).lower(x).compile().as_text())

  @jtu.run_on_devices("gpu", "cpu")
  def test_batch_partitioning(self):
    def f(x):
      return batch_partitionable_ffi_call(x)

    mesh = jtu.create_mesh((len(jax.devices()),), ("i",))
    x = self.rng().randn(8, 4, 5).astype(np.float32)
    x_sharding = jax.NamedSharding(mesh, P("i"))
    x = jax.device_put(x, x_sharding)
    f_jit = jax.jit(f, out_shardings=x_sharding)

    f(x)  # eager mode doesn't crash
    f_jit(x)  # neither does JIT
    self.assertNotIn("all-gather", f_jit.lower(x).compile().as_text())


def batch_partitionable_ffi_call(x):
  return batch_partitionable_p.bind(x)


batch_partitionable_p = core.Primitive("batch_partitionable")
batch_partitionable_p.multiple_results = True
dispatch.simple_impl(batch_partitionable_p)


@batch_partitionable_p.def_abstract_eval
def _batch_partitionable_abstract_eval(x):
  return x, core.ShapedArray(x.shape[:-1], x.dtype)


def _batch_partitionable_lowering(target_name, ctx, x):
  x_aval, = ctx.avals_in
  num_batch_dims = len(x_aval.shape) - 2
  frontend_attrs = mlir.ir_attribute({"num_batch_dims": str(num_batch_dims)})
  return jax.ffi.ffi_lowering(
      target_name,
      extra_attributes={"mhlo.frontend_attributes": frontend_attrs}
  )(ctx, x)


mlir.register_lowering(
    batch_partitionable_p,
    partial(_batch_partitionable_lowering, "lapack_sgeqrf_ffi"),
    platform="cpu",
)
mlir.register_lowering(
    batch_partitionable_p,
    partial(_batch_partitionable_lowering, "cusolver_geqrf_ffi"),
    platform="cuda",
)
mlir.register_lowering(
    batch_partitionable_p,
    partial(_batch_partitionable_lowering, "hipsolver_geqrf_ffi"),
    platform="rocm",
)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
