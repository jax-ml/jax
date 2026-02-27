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

# ruff: noqa: F401

import functools
import math
import os
import sys

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.pallas import core
from jax._src.pallas import pallas_call as pallas_call_lib
if sys.platform != "win32":
  from jax.experimental.pallas import triton as plgpu
else:
  plgpu = None

try:
  from jax._src.lib import triton
except ImportError:
  triton = None
try:
  from jax.experimental.pallas import tpu as pltpu
except ImportError:
  pltpu = None
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax import export
import numpy as np


config.update("jax_traceback_filtering", "off")
config.parse_flags_with_absl()


def matmul_kernel(x_ref, y_ref, o_ref):
  # x shape: (m, l), y shape (l, n), o shape: (m ,n)
  block_m, block_l = x_ref.shape
  block_l2, block_n = y_ref.shape
  assert block_l2 == block_l
  assert o_ref.shape == (block_m, block_n)
  @pl.when(pl.program_id(axis=2) == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref)

  o_ref[...] += x_ref[...] @ y_ref[...]


@functools.partial(jax.jit, static_argnames=['block_shape'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    block_shape=(128, 128, 128)
):
  m, l = x.shape
  l2, n = y.shape
  assert l2 == l
  block_m, block_n, block_l = block_shape
  assert l % block_l == 0, f"{l=}, {block_l=}"
  assert m % block_m == 0, f"{m=}, {block_m=}"
  assert n % block_n == 0, f"{n=}, {block_n=}"
  grid = (m // block_m, n // block_n, l // block_l)
  fused_matmul = pl.pallas_call(
      functools.partial(matmul_kernel),
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      in_specs=[
          pl.BlockSpec((block_m, block_l), lambda i, j, k: (i, k)),
          pl.BlockSpec((block_l, block_n), lambda i, j, k: (k, j)),
      ],
      out_specs=pl.BlockSpec((block_m, block_n), lambda i, j, k: (i, j)),
      grid=grid,
      interpret=jtu.test_device_matches(["cpu"]),
  )
  return fused_matmul(x, y)


class ShapePolyTest(jtu.JaxTestCase, parameterized.TestCase):

  def setUp(self):
    if jax.config.x64_enabled:
      self.skipTest("Only works in 32-bit")
    if (jtu.test_device_matches(["cuda"]) and
        not jtu.is_cuda_compute_capability_at_least("8.0")):
      self.skipTest("Only works on GPU with capability >= sm80")
    if plgpu is None:
      self.skipTest("Triton is not available on this platform")
    super().setUp()
    # TODO(bchetioui): Remove this for H100+ once tests are all compatible with
    # Pallas/Mosaic GPU.
    self.enter_context(pallas_call_lib._PALLAS_USE_MOSAIC_GPU(False))

  def test_copy(self):
    # The blocks are static, but the input and the grid are of polymorphic
    # dimensions.
    block_shape = (8, 128)
    def f(x, *, eager=False, compiler_params=None):  # x: i32[w, h]
      def copy_kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]
      # Use both pl.cdiv and // for specifying the grid
      grid = (pl.cdiv(x.shape[0], block_shape[0]),
              (x.shape[1] + 1) // block_shape[1])
      return pl.pallas_call(
          copy_kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[pl.BlockSpec(block_shape, lambda i, j: (i, j))],
          out_specs=pl.BlockSpec(block_shape, lambda i, j: (i, j)),
          grid=grid,
          compiler_params=compiler_params,
          interpret=eager and jtu.test_device_matches(["cpu"]))(x)

    shape1 = (128, 256)
    x1 = jnp.arange(math.prod(shape1), dtype=np.int32).reshape(shape1)
    res = f(x1, eager=True)
    self.assertAllClose(res, x1)

    w, h = export.symbolic_shape("w, h")
    exp = export.export(jax.jit(f), platforms=["tpu"])(
        jax.ShapeDtypeStruct((w, h), jnp.int32)
    )

    if jtu.test_device_matches(["tpu"]):
      res_exp_1 = exp.call(x1)
      self.assertAllClose(res_exp_1, x1)

      shape2 = block_shape
      x2 = jnp.arange(math.prod(shape2), dtype=np.int32).reshape(shape2)
      res_exp_2 = exp.call(x2)
      self.assertAllClose(res_exp_2, x2)

    # TODO(necula): support shape polymorphism for GPU
    with self.assertRaisesRegex(
        NotImplementedError,
        "dynamic grid bounds not supported in the Triton backend"):
      if plgpu is None:
        self.skipTest("Triton not available")
      export.export(
          jax.jit(functools.partial(f, compiler_params=plgpu.CompilerParams())),
          platforms=["cuda"])(jax.ShapeDtypeStruct((w, h), jnp.int32))

  def test_block_sizes_must_be_static_no_grid(self):
    def f(x, *, eager=False):  # x: f32[w, h]
      def copy_one(x_ref, o_ref):
        o_ref[...] = x_ref[...]
      return pl.pallas_call(
          copy_one,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          interpret=eager and jtu.test_device_matches(["cpu"]))(x)
    shape1 = (128, 256)
    x1 = jnp.arange(math.prod(shape1), dtype=np.int32).reshape(shape1)
    res = f(x1, eager=True)
    self.assertAllClose(res, x1)

    w, h = export.symbolic_shape("w, h")
    with self.assertRaisesRegex(
        ValueError,
        "shape polymorphism for Pallas does not support dynamically-shaped blocks"):
      export.export(
          jax.jit(f),
          platforms=["tpu"])(jax.ShapeDtypeStruct((w, h), jnp.int32))

  def test_block_sizes_must_be_static(self):
    def f(x, *, eager=False):  # x: f32[w, h]
      def copy_one(x_ref, o_ref):
        o_ref[...] = x_ref[...]
      grid = (2, 2)
      block_shape = (x.shape[0] // grid[0], x.shape[1] // grid[1])
      return pl.pallas_call(
          copy_one,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[pl.BlockSpec(block_shape, lambda i, j: (i, j))],
          out_specs=pl.BlockSpec(block_shape, lambda i, j: (i, j)),
          grid=grid,
          interpret=eager and jtu.test_device_matches(["cpu"]))(x)
    shape1 = (128, 256)
    x1 = jnp.arange(math.prod(shape1), dtype=np.int32).reshape(shape1)
    res = f(x1, eager=True)
    self.assertAllClose(res, x1)

    w, h = export.symbolic_shape("w, h")
    with self.assertRaisesRegex(
        ValueError,
        "shape polymorphism for Pallas does not support dynamically-shaped blocks"):

      export.export(
          jax.jit(f),
          platforms=["tpu"])(jax.ShapeDtypeStruct((w, h), jnp.int32))

  @jtu.run_on_devices("tpu")
  def test_matmul(self):
    x_shape = (1024, 256)
    y_shape = (256, 2048)

    key = jax.random.key(42)
    key1, key2 = jax.random.split(key, 2)
    x = jax.random.normal(key1, x_shape, dtype=np.float32)
    y = jax.random.normal(key2, y_shape, dtype=np.float32)

    res = matmul(x, y)
    self.assertAllClose(res, x @ y, atol=1e-4)

    m, n, l = export.symbolic_shape("m, n, l",
                                    constraints=["mod(m, 128) == 0",
                                                 "mod(n, 128) == 0",
                                                 "mod(l, 128) == 0"])
    exp = export.export(matmul, platforms=["tpu"])(
        jax.ShapeDtypeStruct((m, l), jnp.float32),
        jax.ShapeDtypeStruct((l, n), jnp.float32),
    )
    if jtu.test_device_matches(["tpu"]):
      res_exp = exp.call(x, y)
      self.assertAllClose(res_exp, x @ y, atol=1e-4)

  def test_simple_symbolic_matmul_export(self):
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("Not supported on GPU.")

    def sym_matmul(x, y, symbolic_grid):
      symbolic_grid = symbolic_grid.shape[0]
      symbolic_x_0 = x.shape[0] // symbolic_grid
      symbolic_y_1 = y.shape[1] // symbolic_grid

      def x_ref_block_spec_mapping(i, j):
        return (i, 0)

      def y_ref_block_spec_mapping(i, j):
        return (0, j)

      def sym_matmul_kernel(x_ref, y_ref, z_ref):
        z_ref[...] = x_ref[...] @ y_ref[...]

      return pl.pallas_call(
          sym_matmul_kernel,
          out_shape=jax.ShapeDtypeStruct((symbolic_x_0, symbolic_y_1), x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              in_specs=[
                  pl.BlockSpec(
                      (symbolic_x_0, x.shape[1]), x_ref_block_spec_mapping
                  ),
                  pl.BlockSpec(
                      (y.shape[0], symbolic_y_1),
                      y_ref_block_spec_mapping,
                  ),
              ],
              out_specs=pl.BlockSpec(
                  (symbolic_x_0, symbolic_y_1),
                  lambda i, j: (i, j),
              ),
              grid=(symbolic_grid, symbolic_grid),
          ),
      )(x, y)

    a, b, c, d, e = jax.export.symbolic_shape(
        "m_dim, k_dim, n_dim, grid_size, unused_dim",
        constraints=(
            "mod(floordiv(m_dim, grid_size), 8) == 0",
            "mod(k_dim, 128) == 0",
            "mod(floordiv(n_dim, grid_size), 128) == 0",
        ),
    )
    x = jax.ShapeDtypeStruct((a, b), jax.numpy.float32)
    y = jax.ShapeDtypeStruct((b, c), jax.numpy.float32)

    dummy_d = jax.ShapeDtypeStruct((d, e), jax.numpy.float32)

    exported_module = pl.lower_as_mlir(
        jax.jit(sym_matmul),
        x,
        y,
        dummy_d,
        dynamic_shapes=True,
        platforms=["tpu"],
    )
    assert exported_module is not None
    self.assertIn(
        "%arg0: tensor<?x?xf32> loc(unknown), %arg1: tensor<?x?xf32>"
        " loc(unknown), %arg2: tensor<?x?xf32>",
        str(exported_module),
    )
    x = jax.ShapeDtypeStruct((128, 1024), jax.numpy.float32)
    y = jax.ShapeDtypeStruct((1024, 512), jax.numpy.float32)
    dummy_d = jax.ShapeDtypeStruct((1, 1), jax.numpy.float32)
    exported_module = pl.lower_as_mlir(
        jax.jit(sym_matmul),
        x,
        y,
        dummy_d,
        dynamic_shapes=False,
        platforms=["tpu"],
    )
    assert exported_module is not None
    self.assertIn(
        "call @sym_matmul(%arg0, %arg1)",
        str(exported_module),
    )

  def test_pallas_shape_poly_no_cache_collision(self):

    def kernel(x, y):
      y[:] = x[:]

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )
    f = jax.vmap(f)

    x1_shape = jax.ShapeDtypeStruct(
        jax.export.symbolic_shape("b1, 8, 128"), jnp.float32
    )
    exported_module1 = pl.lower_as_mlir(
        jax.jit(f), x1_shape, dynamic_shapes=True
    )
    self.assertIn("(b1, 8, 128)", str(exported_module1))
    x2_shape = jax.ShapeDtypeStruct(
        jax.export.symbolic_shape("b2, 8, 128"), jnp.float32
    )
    exported_module2 = pl.lower_as_mlir(
        jax.jit(f), x2_shape, dynamic_shapes=True
    )
    self.assertIn("(b2, 8, 128)", str(exported_module2))

  def test_cdiv(self):
    def kernel(x, y):
      c = pl.cdiv(x.shape[0], 2)
      assert c == (x.shape[0] + 1) // 2
      assert c == y.shape[0]
      y[:] = x[:c]

    (m,) = jax.export.symbolic_shape("m")
    x_shape = jax.ShapeDtypeStruct((m, 128), jnp.float32)
    y_shape = jax.ShapeDtypeStruct(((m + 1) // 2, 128), jnp.float32)
    f = pl.pallas_call(kernel, out_shape=y_shape)

    exported_module = pl.lower_as_mlir(
        jax.jit(f), x_shape, dynamic_shapes=True, platforms=["tpu"]
    )
    self.assertIn("(m, 128)", str(exported_module))

  def test_dynamic_shapes_export(self):
    def add_vectors_kernel(x_ref, y_ref, o_ref):
      block_b = x_ref.shape[0]

      for batch_idx in range(block_b):
        x_b = x_ref[batch_idx]
        y_b = y_ref[batch_idx]
        o_ref[batch_idx] = x_b + y_b

    def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
      batch_block = 4
      x_block = 128
      grid = (x.shape[0] // batch_block, x.shape[1] // x_block)
      in_specs = [
          pl.BlockSpec(
              (batch_block, x_block, x.shape[2]),
              lambda batch_idx, x_idx: (batch_idx, x_idx, 0),
          ),
          pl.BlockSpec(
              (batch_block, x_block, y.shape[2]),
              lambda batch_idx, x_idx: (batch_idx, x_idx, 0),
          ),
      ]
      out_specs = [
          pl.BlockSpec(
              (batch_block, x_block, x.shape[2]),
              lambda batch_idx, x_idx: (batch_idx, x_idx, 0),
          ),
      ]
      out_shape = [jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)]
      return pl.pallas_call(
          add_vectors_kernel,
          out_shape=out_shape,
          name="my_custom_kernel_name",
          grid_spec=pltpu.PrefetchScalarGridSpec(
              grid=grid,
              in_specs=in_specs,
              out_specs=out_specs,
              num_scalar_prefetch=0,
          ),
      )(x, y)

    batch_size_sym, a_sym, b_sym = jax.export.symbolic_shape(
        "batch_size,a_size,b_size"
    )
    x_shape = jax.ShapeDtypeStruct((batch_size_sym, a_sym, b_sym), jnp.float32)
    y_shape = jax.ShapeDtypeStruct((batch_size_sym, a_sym, b_sym), jnp.float32)

    f_j = jax.jit(add_vectors)
    f_e = jax.export.export(f_j, platforms=["tpu"])

    with core.pallas_export_experimental(dynamic_shapes=True):
      f_k = f_e(x_shape, y_shape)

    self.assertRegex(
        f_k.mlir_module(),
        r"stablehlo.custom_call"
        r" @tpu_custom_call.+kernel_name\s*=\s*\"my_custom_kernel_name\"",
    )

  def test_dynamic_shapes_export_requires_flag(self):
    def add_vectors_kernel(x_ref, y_ref, o_ref):
      o_ref[...] = x_ref[...] + y_ref[...]

    def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
      return pl.pallas_call(
          add_vectors_kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              grid=(1,),
              in_specs=[
                  pl.BlockSpec(x.shape, lambda i: (0, 0)),
                  pl.BlockSpec(y.shape, lambda i: (0, 0)),
              ],
              out_specs=pl.BlockSpec(x.shape, lambda i: (0, 0)),
              num_scalar_prefetch=0,
          ),
      )(x, y)

    m, n = jax.export.symbolic_shape("m,n")
    x_shape = jax.ShapeDtypeStruct((m, n), jnp.float32)

    f_e = jax.export.export(jax.jit(add_vectors), platforms=["tpu"])

    with self.assertRaisesRegex(ValueError, "pallas_export_experimental"):
      f_e(x_shape, x_shape)

  def test_export_vmap(self):
    if not jtu.is_cloud_tpu_at_least(2026, 2, 24):
      self.skipTest("Requires a newer libTPU")
    def add_vectors_kernel(x_ref, y_ref, o_ref):
      o_ref[...] = x_ref[...] + y_ref[...]

    def add_vectors(x, y):
      block_size = 128
      # Grid depends on input shape, which will be symbolic
      grid = (
          pl.cdiv(x.shape[0], block_size),
          pl.cdiv(x.shape[1], block_size),
      )
      block_spec = pl.BlockSpec((block_size, block_size), lambda i, j: (i, j))
      return pl.pallas_call(
          add_vectors_kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=grid,
          in_specs=[block_spec, block_spec],
          out_specs=block_spec,
      )(x, y)

    b, m, n = jax.export.symbolic_shape("b,m,n")
    x_info = jax.ShapeDtypeStruct((b, m, n), jnp.float32)

    exporter = jax.export.export(
        jax.jit(jax.vmap(add_vectors)), platforms=["tpu"]
    )

    with core.pallas_export_experimental(dynamic_shapes=True):
      exp = exporter(x_info, x_info)  # No crash

    if jtu.device_under_test() == "tpu":
      x = y = jnp.ones((4, 128, 128))
      res = exp.call(x, y)
      self.assertAllClose(res, x + y)

      x = y = jnp.ones((4, 192, 192))  # Not multiple of 128
      res = exp.call(x, y)
      self.assertAllClose(res, x + y)


class ExportTestWithTriton(jtu.JaxTestCase):

  def setUp(self):
    if triton is None:
      self.skipTest("Triton is not available on this platform")
    self.enter_context(pallas_call_lib._PALLAS_USE_MOSAIC_GPU(False))
    super().setUp()

  def _check_cuda_export(self, exp):
    self.assertRegex(
        exp.mlir_module(),
        r"stablehlo.custom_call"
        r" @__gpu\$xla\.gpu\.triton.+name\s*=\s*\"my_custom_kernel_name\"",
    )

  def test_cross_platform(self):
    def add_vectors_kernel(x_ref, y_ref, o_ref):
      x, y = x_ref[...], y_ref[...]
      o_ref[...] = x + y

    @jax.jit
    def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
      return pl.pallas_call(
          add_vectors_kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          name="my_custom_kernel_name",
      )(x, y)

    platforms = ["tpu"]
    # TODO(b/394629193): Remove True once the bug is fixed.
    if True or (triton is not None and triton.has_compilation_handler("cuda")):
      # Only include CUDA if GPU support is linked in.
      platforms.append("cuda")

    a = np.arange(8 * 16, dtype=np.int32).reshape((8, 16))
    exp = export.export(
        add_vectors,
        platforms=platforms,
        # The Pallas GPU custom call is not enabled for export by default.
        disabled_checks=[
            export.DisabledSafetyCheck.custom_call("triton_kernel_call"),
            export.DisabledSafetyCheck.custom_call("__gpu$xla.gpu.triton"),
        ],
    )(a, a)

    if jtu.device_under_test() == "tpu" or (
        jtu.device_under_test() == "gpu"
        and jtu.is_cuda_compute_capability_at_least("8.0")
    ):
      res = exp.call(a, a)
      self.assertAllClose(res, a + a)

    # Check that we use the proper kernels names
    if "tpu" in platforms:
      self.assertRegex(
          exp.mlir_module(),
          r"stablehlo.custom_call"
          r" @tpu_custom_call.+kernel_name\s*=\s*\"my_custom_kernel_name\"",
      )
    if "cuda" in platforms:
      self._check_cuda_export(exp)


class ExportTestWithMosaicGpu(ExportTestWithTriton):

  def setUp(self):
    # TODO(b/432678342): remove once this is fixed.
    if jtu.is_device_cuda() and not jtu.is_cuda_compute_capability_at_least(
        "9.0"
    ):
      self.skipTest(
          "LLVM seems to care about the compute capability if a GPU is present"
      )
    super().setUp()
    self.enter_context(pallas_call_lib._PALLAS_USE_MOSAIC_GPU(True))

  def _check_cuda_export(self, exp):
    self.assertRegex(
        exp.mlir_module(),
        r"stablehlo.custom_call @mosaic_gpu_v2.*my_custom_kernel_name",
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
