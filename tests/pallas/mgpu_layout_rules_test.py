# Copyright 2026 The JAX Authors.
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

from absl.testing import absltest

import jax
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.pallas.mosaic_gpu import layout_rules
from jax._src.pjit import explicit_layout
from jax.experimental.mosaic.gpu import fragmented_array as fa
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jax.experimental.mosaic.gpu.mma import MMALayouts
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp


GPUTiledLayout = layout_rules.GPUTiledLayout

jax.config.parse_flags_with_absl()


def _from_fa_layout(layout: fa.FragmentedLayout):
  if not isinstance(layout, fa.TiledLayout):
    raise NotImplementedError(
        f"Unsupported layout type: {type(layout)}"
    )
  return GPUTiledLayout(
      layout.tiling, layout.warp_dims, layout.lane_dims, layout.vector_dim
  )


class LayoutInTypesTest(jtu.JaxTestCase):

  def setUp(self):
    if (jtu.device_under_test() == "gpu" and
        not jtu.is_cuda_compute_capability_at_least("8.0")):
      self.skipTest("Can only run on compute capability >= 8.0")

  def test_elementwise_layout_is_propagated(self):
    arr1 = jnp.arange(64 * 8, dtype=jnp.float32).reshape(64, 8)
    arr2 = jnp.arange(64 * 8, dtype=jnp.float32).reshape(64, 8)

    @explicit_layout
    def add_val(x_val, y_val):
      self.assertEqual(x_val.aval.layout, _from_fa_layout(fa.WGMMA_LAYOUT))
      self.assertEqual(y_val.aval.layout, _from_fa_layout(fa.WGMMA_LAYOUT))
      out_val = x_val + y_val
      self.assertEqual(out_val.aval.layout, _from_fa_layout(fa.WGMMA_LAYOUT))
      return out_val

    @jax.jit
    def f(x, y):
      def add(x_gmem, y_gmem, o_gmem):
        x_val = plgpu.load(x_gmem, optimized=False)
        y_val = plgpu.load(y_gmem, optimized=False)

        x_layout = GPUTiledLayout.for_array(x_val, plgpu.Layout.WGMMA)
        y_layout = GPUTiledLayout.for_array(y_val, plgpu.Layout.WGMMA)

        o_val = add_val(x_val, y_val, in_layouts=(x_layout, y_layout))
        o_gmem[...] = o_val
      return plgpu.kernel(add, out_type=jax.typeof(x))(x, y)

    f.trace(arr1, arr2)
    if jtu.device_under_test() == "gpu":
      f(arr1, arr2)

  def test_invalid_layout_assignment_raises(self):
    arr1 = jnp.arange(64 * 128, dtype=jnp.bfloat16).reshape(64, 128)
    arr2 = jnp.arange(128 * 256, dtype=jnp.bfloat16).reshape(128, 256)

    @explicit_layout
    def mma(acc, x_val, y_val):
      out_val = plgpu.mma(acc, x_val, y_val)
      return out_val

    @jax.jit
    def f(x, y):
      def matmul(x_gmem, y_gmem, o_gmem):
        x = plgpu.load(x_gmem, optimized=False)
        y = plgpu.load(y_gmem, optimized=False)
        acc = jnp.zeros((x.shape[0], y.shape[1]), dtype=jnp.float32)

        x_layout = GPUTiledLayout.for_array(x, plgpu.Layout.WGMMA)
        y_layout = GPUTiledLayout.for_array(y, plgpu.Layout.WGMMA)
        acc_layout = GPUTiledLayout.for_array(acc, plgpu.Layout.WGMMA)

        o_val = mma(acc, x, y, in_layouts=(x_layout, y_layout, acc_layout))
        o_gmem[...] = o_val

      return plgpu.kernel(
          matmul,
          out_type=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), jnp.float32),
      )(x, y)

    with self.assertRaisesRegex(ValueError, "Failed to infer layouts"):
      f.trace(arr1, arr2)

  def test_mma_layout_is_propagated(self):
    arr1 = jnp.arange(64 * 128, dtype=jnp.bfloat16).reshape(64, 128)
    arr2 = jnp.arange(128 * 256, dtype=jnp.bfloat16).reshape(128, 256)

    with mlir.make_ir_context():
      layouts = MMALayouts(mgpu_utils.dtype_to_ir_type(jnp.bfloat16))

    @explicit_layout
    def mma(acc, x_val, y_val):
      self.assertEqual(acc.layout, _from_fa_layout(layouts.acc))
      self.assertEqual(x_val.layout, _from_fa_layout(layouts.lhs))
      self.assertEqual(y_val.layout, _from_fa_layout(layouts.rhs))
      out_val = plgpu.mma(acc, x_val, y_val)
      self.assertEqual(out_val.layout, _from_fa_layout(layouts.acc))
      return out_val

    @jax.jit
    def f(x, y):
      def matmul(x_gmem, y_gmem, o_gmem):
        x = plgpu.load(x_gmem, optimized=False)
        y = plgpu.load(y_gmem, optimized=False)
        acc = jnp.zeros((x.shape[0], y.shape[1]), dtype=jnp.float32)

        x_layout = GPUTiledLayout.for_array(x, plgpu.Layout.MMA_LHS(jnp.bfloat16))
        y_layout = GPUTiledLayout.for_array(y, plgpu.Layout.MMA_RHS(jnp.bfloat16))
        acc_layout = GPUTiledLayout.for_array(acc, plgpu.Layout.MMA_ACC(jnp.float32))

        o_val = mma(acc, x, y, in_layouts=(acc_layout, x_layout, y_layout))
        o_gmem[...] = o_val

      return plgpu.kernel(
          matmul,
          out_type=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), jnp.float32),
      )(x, y)

    f.trace(arr1, arr2)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
