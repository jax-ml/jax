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

import functools
import math

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
import jax._src.pallas.mosaic_gpu as plgpu
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np


jax.config.parse_flags_with_absl()


class PallasTest(jtu.JaxTestCase):

  def setUp(self):
    if config.enable_x64.value:
      self.skipTest("Only works on x32 at the moment")
    if not jtu.is_cuda_compute_capability_at_least("9.0"):
      self.skipTest("Only works on a GPU with capability >= sm90")

    super().setUp()


class PallasCallTest(PallasTest):

  def test_add_one(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    x = jnp.arange(256).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  def test_add_xy(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = x_ref[...] + y_ref[...]

    x = jnp.arange(256).astype(jnp.float32)
    y = x + 1
    np.testing.assert_array_equal(kernel(x, y), x + y)

  def test_add_one_grid(self):
    @functools.partial(
        pl.pallas_call,
        in_specs=[pl.BlockSpec((128,), lambda *i: i)],
        out_specs=pl.BlockSpec((128,), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct([128 * 2], jnp.float32),
        grid=2,
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    x = jnp.arange(128 * 2).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  def test_add_one_grid_with_scratch(self):

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128 * 2], jnp.float32),
        in_specs=[pl.BlockSpec((128,), lambda *i: i)],
        out_specs=pl.BlockSpec((128,), lambda *i: i),
        scratch_shapes=[plgpu.SMEM((128,), jnp.float32)],
        grid=2,
    )
    def kernel(x_ref, o_ref, scratch_ref):
      scratch_ref[...] = x_ref[...] + 1
      o_ref[...] = scratch_ref[...]

    x = jnp.arange(256).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  @parameterized.product(num_stages=[1, 2, 3])
  def test_add_one_grid_pipelined(self, num_stages):

    @functools.partial(
        pl.pallas_call,
        in_specs=[pl.BlockSpec((128, 16), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((128, 16), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct([128 * 2, 64], jnp.float32),
        compiler_params=plgpu.GPUCompilerParams(
            dimension_semantics=["parallel", "sequential"],
            num_stages=num_stages,
        ),
        grid=(2, 1),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    x = jnp.arange(128 * 2 * 64).reshape((128 * 2, 64)).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  def test_add_one_with_async_copy_smem_to_gmem(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        scratch_shapes=[plgpu.SMEM((128,), jnp.float32)],
    )
    def kernel(x_ref, o_ref_gmem, scratch_ref):
      scratch_ref[...] = x_ref[...] + 1
      plgpu.async_copy_smem_to_gmem(scratch_ref, o_ref_gmem)
      plgpu.wait_smem_to_gmem(0)

    x = jnp.arange(128).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  def test_add_one_with_async_copy_gmem_to_smem(self):

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
        in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),),
        scratch_shapes=[
            plgpu.SMEM((128,), jnp.float32),
            plgpu.Barrier(num_arrivals=1),
        ],
    )
    def kernel(x_ref_gmem, o_ref, scratch_ref, barrier_ref):
      plgpu.async_copy_gmem_to_smem(
          x_ref_gmem, scratch_ref, barrier=barrier_ref
      )
      plgpu.wait_barrier(barrier_ref)
      o_ref[...] = scratch_ref[...] + 1

    x = jnp.arange(128).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  def test_add_doubled_sum(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + jnp.sum(x_ref[...]) + jnp.sum(x_ref[...])

    x = jnp.arange(128).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + x.sum()*2)

  @parameterized.parameters(False, True)
  def test_rsqrt(self, approx_math):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
        compiler_params=plgpu.GPUCompilerParams(approx_math=approx_math),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = jax.lax.rsqrt(x_ref[...])

    x = jnp.arange(128).astype(jnp.float32)
    np.testing.assert_allclose(kernel(x), jax.lax.rsqrt(x))

  @parameterized.product(input_factor=[0.001, 1, 10, 100, 100])
  def test_layer_norm(self, input_factor):
    eps = 1e-5
    gamma = 1.0
    beta = 1.0

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def layer_norm(x_ref, o_ref):
      x_mean = jnp.mean(x_ref[...])
      x_centered = x_ref[...] - x_mean
      o_ref[...] = (
          x_centered * jax.lax.rsqrt(jnp.mean(x_centered**2) + eps) * gamma
          + beta
      )

    def layer_norm_np(x):
      x_mean = np.mean(x)
      x_centered = x - x_mean
      return (x_centered / np.sqrt(np.mean(x_centered**2) + eps) * gamma) + beta

    # Ones are always fully precise
    x = jnp.ones((256,)).astype(jnp.float32) * input_factor
    np.testing.assert_allclose(layer_norm(x), layer_norm_np(x))

    # random (and anything else is not)
    x = (
        jax.random.uniform(jax.random.key(42), shape=(256,), dtype=jnp.float32)
        * input_factor
    )
    # TODO(cperivol): find out why in this particular case we have a small-ish error.
    rtol = 1e-07 if input_factor > 10 else 5e-5
    np.testing.assert_allclose(layer_norm(x), layer_norm_np(x), rtol=rtol)

  def test_print(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, o_ref):
      del x_ref, o_ref
      pl.debug_print("It works!")

    x = jnp.arange(256).astype(jnp.float32)
    with jtu.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertEqual(output(), "It works!\n")

  def test_print_with_values(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, o_ref):
      del o_ref
      pl.debug_print("x[0] = {}", x_ref[0])

    x = jnp.arange(256).astype(jnp.float32)
    with self.assertRaises(Exception):
      # TODO(slebedev): Remove assertRaises() once we support indexing.
      kernel(x)

  def test_print_array(self):
    in_shape = [2, 1, 64, 64]
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct(in_shape, jnp.float32),
    )
    def kernel(x_ref, o_ref):
      del o_ref
      pl.debug_print("x: {}", x_ref[...])

    x = jnp.arange(math.prod(in_shape)).reshape(in_shape).astype(jnp.float32)
    with jtu.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertIn(f"x: [1, 0, 43, 23]/{list(in_shape)}: 6871.000000\n", output())

  def test_scoped_allocation(self):
    def kernel(x_ref, o_ref):
      def body(tmp_ref):
        self.assertEqual(tmp_ref.shape, (8, 128))
        tmp_ref[...] = x_ref[...] + 1.0
        return tmp_ref[...]

      tmp = pl.run_scoped(body, plgpu.SMEM((8, 128), jnp.float32))
      self.assertEqual(tmp.shape, (8, 128))
      o_ref[...] = tmp

    inp = np.ones((8, 128))
    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )
    o = f(inp)
    np.testing.assert_array_equal(o, inp + 1.0)

  def test_program_id(self):
    @functools.partial(
        pl.pallas_call,
        in_specs=(),
        out_specs=pl.BlockSpec((128,), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct([128 * 2], jnp.int32),
        grid=2,
    )
    def kernel(o_ref):
      o_ref[...] = jnp.full(o_ref.shape, pl.program_id(0))

    np.testing.assert_array_equal(
        kernel(),
        jnp.array([0] * 128 + [1] * 128, dtype=jnp.int32),
    )

  def test_num_programs(self):
    @functools.partial(
        pl.pallas_call,
        in_specs=(),
        out_specs=pl.BlockSpec((128,), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct([128 * 2], jnp.int32),
        grid=2,
    )
    def kernel(o_ref):
      o_ref[...] = jnp.full(o_ref.shape, pl.num_programs(0))

    np.testing.assert_array_equal(
        kernel(),
        jnp.full([256], 2, dtype=jnp.int32),
    )

  def test_swizzled_blockspec_shapes(self):
    @functools.partial(
        pl.pallas_call,
        in_specs=[
            plgpu.GPUBlockSpec(
                (128, 64), lambda *i: i, tiling=(64, 64), swizzle=128
            ),
        ],
        out_specs=pl.BlockSpec((2, 1, 64, 64), lambda i, j: (i, j, 64, 64)),
        out_shape=jax.ShapeDtypeStruct((4, 2, 64, 64), jnp.float16),
        grid=(2, 2),
    )
    def kernel(x_ref, o_ref):
      assert x_ref.shape == (2, 1, 64, 64), x_ref.shape
      o_ref[...] = x_ref[...]

    x = jnp.zeros((256, 128), dtype=jnp.float16)
    result = kernel(x)
    self.assertEqual(result.shape, (4, 2, 64, 64))

  def test_fori_loop(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, o_ref):
      # Equivalent to x_ref[...] + 2 + 3.
      o_ref[...] = jax.lax.fori_loop(2, 4, lambda i, x: x + i, x_ref[...])

    x = jnp.arange(256).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 2.0 + 3.0)

  def test_wgmma(self):
    dtype = jnp.float16
    swizzle = 128
    elems_128b = swizzle // jnp.dtype(dtype).itemsize
    def kernel(a_ref, b_ref, o_ref):
      acc = plgpu.zero_accumulator((64, 128), jnp.float32)
      acc = plgpu.wgmma(acc, a_ref, b_ref, rhs_transpose=False)
      plgpu.wgmma_wait(0)
      # TODO(cperivol): turn acc into a reference so we can reason about effects.
      o_ref[...] = acc.as_array()

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(64, 128), dtype=dtype)
    b = jax.random.uniform(key2, shape=(128, 128), dtype=dtype)

    res = pl.pallas_call(
        kernel,
        in_specs=[
            plgpu.GPUBlockSpec(
                (64, 128),
                lambda i, j: (i, j),
                tiling=(64, elems_128b),
                swizzle=128,
            ),
            plgpu.GPUBlockSpec(
                (128, 128),
                lambda *i: i,
                tiling=(elems_128b, elems_128b),
                swizzle=128,
            ),
        ],
        out_specs=plgpu.GPUBlockSpec((64, 128), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct((64, 128), jnp.float32),
        grid=(1, 1),
    )(a, b)
    np.testing.assert_allclose(
        res, a @ b, rtol=1e-3
    )

if __name__ == "__main__":
  absltest.main()
