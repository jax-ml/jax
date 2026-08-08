# Copyright 2025 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import functools

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax._src import config
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import numpy as np

try:
  import torch
except ImportError:
  torch = None
try:
  # We only import this to see if Mosaic is available.
  import jax.experimental.mosaic.gpu  # noqa: F401
except ImportError:
  attention_mgpu = None
else:
  from jax.experimental.pallas.ops.gpu import attention_mgpu

config.parse_flags_with_absl()


class TorchTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jtu.test_device_matches(["rocm"]):
      self.skipTest("Mosaic GPU is not supported on ROCm.")
    if torch is None:
      self.skipTest("Test requires PyTorch")
    if attention_mgpu is None:
      self.skipTest("Mosaic GPU not available.")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("Only works on GPU with capability sm90a+")

  def test_simple_pl_kernel_pipeline(self):
    size = 1024
    tile = 128
    k = size // tile

    @plgpu.as_torch_kernel
    @plgpu.kernel(
        out_type=(jax.ShapeDtypeStruct((size,), jnp.int32)),
    )
    def kernel(x_ref, y_ref, o_ref):
      def do_sum(indices, x_smem, y_smem, o_smem):
        o_smem[...] = x_smem[...] + y_smem[...]

      plgpu.emit_pipeline(
        do_sum,
        grid=(k,),
        in_specs=[
            plgpu.BlockSpec((tile, ), lambda ki: (ki,)),
            plgpu.BlockSpec((tile,), lambda ki: (ki,)),
        ],
        out_specs=[
            plgpu.BlockSpec((tile, ), lambda ki: (ki,)),
        ]
      )(x_ref, y_ref, o_ref)

    x = torch.arange(size, dtype=torch.int32, device="cuda")
    y = torch.arange(size, dtype=torch.int32, device="cuda")
    np.testing.assert_array_equal(kernel(x, y).cpu(), (x + y).cpu())

  def test_simple_plgpu_kernel(self):
    @plgpu.as_torch_kernel
    @functools.partial(
        plgpu.kernel, out_type=jax.ShapeDtypeStruct([128], jnp.int32)
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = x_ref[...] + y_ref[0]

    x = torch.arange(128, dtype=torch.int32, device="cuda")
    y = torch.arange(128, dtype=torch.int32, device="cuda")
    np.testing.assert_array_equal(kernel(x, y).cpu(), (x + y[0]).cpu())

  def test_flip(self):
    @functools.partial(
        pl.kernel,
        mesh=plgpu.Mesh(),
        out_type=(jax.ShapeDtypeStruct([128], jnp.int32),) * 2,
        compiler_params=plgpu.CompilerParams(),
    )
    def kernel(x_ref, y_ref, x_o_ref, y_o_ref):
      x_o_ref[...] = x_ref[...]
      y_o_ref[...] = y_ref[...]

    x = torch.arange(128, dtype=torch.int32, device="cuda")
    y = torch.arange(128, dtype=torch.int32, device="cuda")
    yo, xo = plgpu.as_torch_kernel(lambda x, y: kernel(x, y)[::-1])(x, y)
    np.testing.assert_array_equal(xo.cpu(), x.cpu())
    np.testing.assert_array_equal(yo.cpu(), y.cpu())

  def test_not_all_returned(self):
    @functools.partial(
        plgpu.kernel,
        out_type=(jax.ShapeDtypeStruct([128], jnp.int32),) * 2,
    )
    def kernel(x_ref, y_ref, x_o_ref, y_o_ref):
      x_o_ref[...] = x_ref[...]
      y_o_ref[...] = y_ref[...]

    x = torch.arange(128, dtype=torch.int32, device="cuda")
    y = torch.arange(128, dtype=torch.int32, device="cuda")
    xo = plgpu.as_torch_kernel(lambda x, y: kernel(x, y)[0])(x, y)
    np.testing.assert_array_equal(xo.cpu(), x.cpu())

  def test_invalid(self):
    @functools.partial(
        plgpu.kernel,
        out_type=(jax.ShapeDtypeStruct([128], jnp.int32),) * 2,
    )
    def kernel(x_ref, y_ref, x_o_ref, y_o_ref):
      x_o_ref[...] = x_ref[...]
      y_o_ref[...] = y_ref[...]

    x = torch.arange(128, dtype=torch.int32, device="cuda")
    y = torch.arange(128, dtype=torch.int32, device="cuda")

    with self.assertRaisesRegex(ValueError, "Unsupported operation .* stablehlo.add"):
      plgpu.as_torch_kernel(lambda x, y: x + y)(x, y)

    with self.assertRaisesRegex(ValueError, "Multiple Mosaic GPU kernels"):
      plgpu.as_torch_kernel(lambda x, y: kernel(*kernel(x, y)))(x, y)
    with self.assertRaisesRegex(ValueError, "Unsupported operation .* stablehlo.add"):
      plgpu.as_torch_kernel(lambda x, y: kernel(x, y + jnp.ones_like(x)))(x, y)
    with self.assertRaisesRegex(ValueError, "The function can only return kernel results"):
      plgpu.as_torch_kernel(lambda x, y: (kernel(x, y), x, y))(x, y)

  def test_attention(self):
    if not jtu.is_cuda_compute_capability_equal("9.0"):
      self.skipTest("Test requires compute capability == 9.0")
    batch_size = 1
    q_seq_len = 4096
    kv_seq_len = 4096
    head_dim = 64
    num_q_heads, num_kv_heads = 4, 1
    block_q = block_kv = 64
    q = torch.randn(
        (batch_size, q_seq_len, num_q_heads, head_dim),
        dtype=torch.float16,
        device="cuda",
    )
    k = torch.randn(
        (batch_size, kv_seq_len, num_kv_heads, head_dim),
        dtype=torch.float16,
        device="cuda",
    )
    v = torch.randn(
        (batch_size, kv_seq_len, num_kv_heads, head_dim),
        dtype=torch.float16,
        device="cuda",
    )
    kernel_fn = functools.partial(
        attention_mgpu.attention,
        config=attention_mgpu.TuningConfig(
            block_q=block_q,
            block_kv=block_kv,
            max_concurrent_steps=2,
        ),
    )
    np.testing.assert_array_equal(
        plgpu.as_torch_kernel(kernel_fn)(q, k, v).cpu(),
        kernel_fn(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)),
    )

  def test_torch_aliasing(self):
    @pl.kernel(mesh=plgpu.Mesh(), out_type=(), compiler_params=plgpu.CompilerParams())
    def kernel(x_ref):
      x_ref[...] = jnp.ones_like(x_ref)

    x = torch.zeros(128, dtype=torch.float32, device="cuda")
    plgpu.as_torch_kernel(kernel)(x)  # Run for side effects
    np.testing.assert_array_equal(
        x.cpu(), torch.ones((128,), dtype=torch.float32, device="cpu")
    )

  @parameterized.parameters(
      plgpu.LoweringSemantics.Lane,
      plgpu.LoweringSemantics.Warpgroup,
  )
  def test_simple_matmul(self, lowering_semantics):
    # TODO: put CC version constraints: CC >= 12.0
    m, n, k = 128, 128, 64
    acc_dtype = jnp.float32
    dtype = jnp.bfloat16

    to_torch_dtypes = {
      jnp.float32: torch.float32,
      jnp.float16: torch.float16,
      jnp.bfloat16: torch.bfloat16,
    }

    @plgpu.kernel(
        out_type=jax.ShapeDtypeStruct((m, n), acc_dtype),
        compiler_params=plgpu.CompilerParams(lowering_semantics=lowering_semantics),
    )
    def kernel(x_ref, y_ref, o_ref):
      acc = plgpu.layout_cast(
          jnp.zeros((m, n), acc_dtype), plgpu.Layout.MMA_ACC(dtype)
      )
      x = plgpu.load(
          x_ref, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False
      )
      y = plgpu.load(
          y_ref.T, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False
      )
      o_ref[...] = plgpu.mma(acc, x, y)

    gen = torch.Generator(device="cuda").manual_seed(123)
    x = torch.rand(m, k, generator=gen, dtype=to_torch_dtypes[dtype], device="cuda")
    y = torch.rand(n, k, generator=gen, dtype=to_torch_dtypes[dtype], device="cuda")
    out = plgpu.as_torch_kernel(kernel)(x, y)
    expected = (x.to(to_torch_dtypes[acc_dtype]) @ y.T.to(to_torch_dtypes[acc_dtype]))
    np.testing.assert_allclose(out.cpu(), expected.cpu(), atol=1e-2, rtol=1e-2)

  @parameterized.parameters(
      plgpu.LoweringSemantics.Lane,
      plgpu.LoweringSemantics.Warpgroup,
  )
  def test_matmul_with_smem(self, lowering_semantics):
    # TODO: put CC version constraints: CC >= 12.0
    m, n, k = 128, 128, 64
    acc_dtype = jnp.float32
    dtype = jnp.bfloat16

    to_torch_dtypes = {
      jnp.float32: torch.float32,
      jnp.float16: torch.float16,
      jnp.bfloat16: torch.bfloat16,
    }

    @plgpu.kernel(
        out_type=jax.ShapeDtypeStruct((m, n), acc_dtype),
        compiler_params=plgpu.CompilerParams(lowering_semantics=lowering_semantics),
        scratch_types=[
            plgpu.SMEM((m, k), dtype),
            plgpu.SMEM((k, n), dtype),
            plgpu.Barrier(num_arrivals=2),
        ],
    )
    def kernel(x_ref, y_ref, o_ref, x_smem, y_smem, barrier_ref):
      acc = plgpu.layout_cast(
          jnp.zeros((m, n), acc_dtype), plgpu.Layout.MMA_ACC(dtype)
      )

      plgpu.copy_gmem_to_smem(x_ref, x_smem, barrier_ref)
      plgpu.copy_gmem_to_smem(y_ref, y_smem, barrier_ref)
      plgpu.barrier_wait(barrier_ref)

      x = plgpu.load(
          x_smem, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False
      )
      y = plgpu.load(
          y_smem, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False
      )
      o_ref[...] = plgpu.mma(acc, x, y)

    gen = torch.Generator(device="cuda").manual_seed(123)
    x = torch.rand(m, k, generator=gen, dtype=to_torch_dtypes[dtype], device="cuda")
    y = torch.rand(k, n, generator=gen, dtype=to_torch_dtypes[dtype], device="cuda")
    out = plgpu.as_torch_kernel(kernel)(x, y)
    expected = (x.to(to_torch_dtypes[acc_dtype]) @ y.to(to_torch_dtypes[acc_dtype]))
    np.testing.assert_allclose(out.cpu(), expected.cpu(), atol=1e-2, rtol=1e-2)

  @parameterized.parameters([
      plgpu.LoweringSemantics.Lane,
      plgpu.LoweringSemantics.Warpgroup,
  ])
  def test_matmul_with_pipeline(self, lowering_semantics):
    # TODO: put CC version constraints: CC >= 12.0
    m, n, k = 512, 512, 256
    tile_m = tile_n = 64
    tile_k = 64
    out_dtype = jnp.float32
    dtype = jnp.bfloat16

    to_torch_dtypes = {
      jnp.float32: torch.float32,
      jnp.float16: torch.float16,
      jnp.bfloat16: torch.bfloat16,
    }

    @plgpu.kernel(
        out_type=jax.ShapeDtypeStruct((m, n), out_dtype),
        compiler_params=plgpu.CompilerParams(lowering_semantics=lowering_semantics),
        grid=(m // tile_m, n // tile_n),
        grid_names=('m', 'n'),
        scratch_types=[
            plgpu.SMEM((tile_m, tile_n), out_dtype),  # output
        ],
    )
    def kernel(x_gmem, y_gmem, o_gmem, o_smem):
      pid_m = jax.lax.axis_index('m')
      pid_n = jax.lax.axis_index('n')

      acc = plgpu.layout_cast(
          jnp.zeros((tile_m, tile_n), out_dtype), plgpu.Layout.MMA_ACC(dtype)
      )

      def body(_, x_smem, y_smem, carry):
        x = plgpu.load(
            x_smem, layout=plgpu.Layout.MMA_LHS(dtype), optimized=False
        )
        y = plgpu.load(
            y_smem, layout=plgpu.Layout.MMA_RHS(dtype), optimized=False
        )
        return plgpu.mma(carry, x, y)

      acc = plgpu.emit_pipeline(
        body,
        grid=(k // tile_k,),
        in_specs=[
            plgpu.BlockSpec(
                (tile_m, tile_k), lambda ki: (pid_m, ki), delay_release=1
            ),
            plgpu.BlockSpec(
                (tile_k, tile_n), lambda ki: (ki, pid_n), delay_release=1
            ),
        ],
        max_concurrent_steps=2,
        init_carry=acc,
      )(x_gmem, y_gmem)

      o_smem[...] = acc.astype(out_dtype)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(
          o_smem,
          o_gmem.at[
              pl.ds(pid_m * tile_m, tile_m),
              pl.ds(pid_n * tile_n, tile_n)
          ],
      )
      plgpu.wait_smem_to_gmem(0)  # Wait for all copies to finish.

    gen = torch.Generator(device="cuda").manual_seed(123)
    x = torch.rand(m, k, generator=gen, dtype=to_torch_dtypes[dtype], device="cuda")
    y = torch.rand(k, n, generator=gen, dtype=to_torch_dtypes[dtype], device="cuda")
    out = plgpu.as_torch_kernel(kernel)(x, y)
    expected = (x.to(to_torch_dtypes[out_dtype]) @ y.to(to_torch_dtypes[out_dtype]))
    np.testing.assert_allclose(out.cpu(), expected.cpu(), atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
