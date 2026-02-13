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

"""Test exporting Pallas kernels."""

from jax.experimental.pallas import tpu as pltpu
import sys

from absl.testing import absltest
import jax
from jax import export
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax._src.pallas import core
from jax._src.pallas import pallas_call as pallas_call_lib
from jax.experimental import pallas as pl
import numpy as np


try:
  from jax._src.lib import triton
except ImportError:
  triton = None  # Windows builds don't have Triton.


jax.config.parse_flags_with_absl()

pallas_export_experimental = core.pallas_export_experimental

class ExportTestWithTriton(jtu.JaxTestCase):

  def setUp(self):
    if sys.platform == "win32":
      self.skipTest("Only works on non-Windows platforms")
    self.enter_context(pallas_call_lib._PALLAS_USE_MOSAIC_GPU(False))
    super().setUp()

  def _check_cuda_export(self, exp):
    self.assertRegex(
        exp.mlir_module(),
        r"stablehlo.custom_call @__gpu\$xla\.gpu\.triton.+name\s*=\s*\"my_custom_kernel_name\"")

  def test_cross_platform(self):
    def add_vectors_kernel(x_ref, y_ref, o_ref):
      x, y = x_ref[...], y_ref[...]
      o_ref[...] = x + y

    @jax.jit
    def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
      return pl.pallas_call(add_vectors_kernel,
                            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                            name="my_custom_kernel_name",
                            )(x, y)

    platforms = ["tpu"]
    # TODO(b/394629193): Remove True once the bug is fixed.
    if True or triton.has_compilation_handler("cuda"):
      # Only include CUDA if GPU support is linked in.
      platforms.append("cuda")

    a = np.arange(8 * 16, dtype=np.int32).reshape((8, 16))
    exp = export.export(
        add_vectors,
        platforms=platforms,
        # The Pallas GPU custom call is not enabled for export by default.
        disabled_checks=[
            export.DisabledSafetyCheck.custom_call("triton_kernel_call"),
            export.DisabledSafetyCheck.custom_call("__gpu$xla.gpu.triton")
        ]
    )(a, a)

    if (jtu.device_under_test() == "tpu" or
        (jtu.device_under_test() == "gpu" and
         jtu.is_cuda_compute_capability_at_least("8.0"))):
      res = exp.call(a, a)
      self.assertAllClose(res, a + a)

    # Check that we use the proper kernels names
    if "tpu" in platforms:
      self.assertRegex(
          exp.mlir_module(),
          r"stablehlo.custom_call @tpu_custom_call.+kernel_name\s*=\s*\"my_custom_kernel_name\"")
    if "cuda" in platforms:
      self._check_cuda_export(exp)


class ExportTestWithMosaicTPU(jtu.JaxTestCase):
  def test_dynamic_shapes_export(self):
    if jtu.device_under_test() != "tpu":
      self.skipTest("Mosaic TPU test only runs on TPU")

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
          pl.BlockSpec((batch_block, x_block, x.shape[2]), lambda batch_idx, x_idx: (batch_idx, x_idx, 0)),
          pl.BlockSpec((batch_block, x_block, y.shape[2]), lambda batch_idx, x_idx: (batch_idx, x_idx, 0)),
      ]
      out_specs = [
          pl.BlockSpec((batch_block, x_block, x.shape[2]), lambda batch_idx, x_idx: (batch_idx, x_idx, 0)),
      ]
      out_shape = [jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)]
      return pl.pallas_call(add_vectors_kernel,
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
        "batch_size,a_size,b_size")
    x_shape = jax.ShapeDtypeStruct((batch_size_sym, a_sym, b_sym), jnp.float32)
    y_shape = jax.ShapeDtypeStruct((batch_size_sym, a_sym, b_sym), jnp.float32)

    f_j = jax.jit(add_vectors)
    f_e = jax.export.export(f_j, platforms=["tpu"])

    with pallas_export_experimental(dynamic_shapes=True):
      f_k = f_e(x_shape, y_shape)

    print(f_k.mlir_module())
    self.assertRegex(
        f_k.mlir_module(),
        r"stablehlo.custom_call @tpu_custom_call.+kernel_name\s*=\s*\"my_custom_kernel_name\"")


class ExportTestWithMosaicGpu(ExportTestWithTriton):

  def setUp(self):
    # TODO(b/432678342): remove once this is fixed.
    if jtu.is_device_cuda() and not jtu.is_cuda_compute_capability_at_least("9.0"):
      self.skipTest(
          "LLVM seems to care about the compute capability if a GPU is present"
      )
    super().setUp()
    self.enter_context(pallas_call_lib._PALLAS_USE_MOSAIC_GPU(True))

  def _check_cuda_export(self, exp):
    self.assertRegex(
        exp.mlir_module(),
        r"stablehlo.custom_call @mosaic_gpu_v2.*my_custom_kernel_name")

  def test_cross_platform(self):
    # Skip test_cross_platform on ROCm: Mosaic GPU not supported on ROCm.
    # TODO(GulsumGudukbay): Unskip once Mosaic is enabled. Issues #34669
    # and # 34711.
    if jtu.is_device_rocm():
      self.skipTest("Mosaic GPU not supported on ROCm.")
    return super().test_cross_platform()


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
