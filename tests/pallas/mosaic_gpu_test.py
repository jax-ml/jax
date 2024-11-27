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

import contextlib
import functools
import math
import os
import re
import tempfile
import traceback

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
  from jax._src.lib import mosaic_gpu as mosaic_gpu_lib
except ImportError:
  mosaic_gpu_lib = None


jax.config.parse_flags_with_absl()


class PallasTest(jtu.JaxTestCase):

  def setUp(self):
    if config.enable_x64.value:
      self.skipTest("Only works on x32 at the moment")
    if not jtu.is_cuda_compute_capability_at_least("9.0"):
      self.skipTest("Only works on a GPU with capability >= sm90")

    super().setUp()

  @contextlib.contextmanager
  def capture_stdout(self):
    if mosaic_gpu_lib is None:
      raise ValueError("Running tests but missing Mosaic GPU extension")
    with jtu.capture_stdout() as stdout:
      yield stdout
      # We need to cudaDeviceSynchronize to make sure printfs are flushed.
      mosaic_gpu_lib._mosaic_gpu_ext._sync_all_devices()

  def skip_unless_sm90a(self):
    if not jtu.is_cuda_compute_capability_equal("9.0"):
      self.skipTest("Only works on GPU with capability sm90a")


class PallasCallTest(PallasTest):

  @parameterized.named_parameters(
      ("add_one", lambda x:  x + 1.),
      ("logistic", jax.lax.logistic),
      ("exp", jax.lax.exp),
      ("square", lambda x: x ** 2),
      ("rsqrt", jax.lax.rsqrt),
      ("tanh", jax.lax.tanh, 1e-6),
  )
  def test_unary_ops(self, unary, rtol=1e-7):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = unary(x_ref[...])

    x = jnp.arange(256).astype(jnp.float32)
    np.testing.assert_allclose(kernel(x), unary(x), rtol=rtol)

  @parameterized.named_parameters(
      ("add", lambda x, y: x + y),
      ("mul", lambda x, y: x * y),
      ("div", lambda x, y: x / y),
      ("min", lambda x, y: jnp.minimum(x, y)),
      ("max", lambda x, y: jnp.maximum(x, y)),
  )
  def test_binary_op(self, bop):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = bop(x_ref[...], y_ref[...])

    x = jnp.arange(256).astype(jnp.float32)
    y = x + 1
    np.testing.assert_array_equal(kernel(x, y), bop(x, y))

  def test_add_first(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
    )
    def kernel(x_ref, y_ref, o_ref):
      o_ref[...] = x_ref[...] + y_ref[0]

    x = jnp.arange(256).astype(jnp.float32)
    y = jnp.flip(x).reshape(1, 256)
    np.testing.assert_array_equal(kernel(x, y), x + y[0])

  def test_reshape(self):
    shape1, shape2 = (128,), (2, 16, 4)

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape2, jnp.float32),
    )
    def kernel(x_ref, out_ref):
      x_ref_reshaped = x_ref.reshape(shape2)
      self.assertEqual(x_ref.shape, shape1)
      self.assertEqual(x_ref_reshaped.shape, shape2)
      out_ref[...] = x_ref_reshaped[...]

    x = jnp.arange(math.prod(shape1)).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x.reshape(shape2))

  def test_add_xy_indexed(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
    )
    def kernel(x_ref, y_ref, o_ref):
      idx = jnp.sum(y_ref[...])
      o_ref[...] = x_ref[idx]

    x = jnp.arange(4 * 128).reshape(4, 128).astype(jnp.float32)
    y = jnp.zeros(128, dtype=jnp.int32)
    np.testing.assert_array_equal(kernel(x, y), x[jnp.sum(y)])

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

  @parameterized.product(max_concurrent_steps=[1, 2, 3, 4, 16])
  def test_add_one_grid_pipelined(self, max_concurrent_steps):

    @functools.partial(
        pl.pallas_call,
        in_specs=[pl.BlockSpec((128, 16), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((128, 16), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct([128 * 2, 64], jnp.float32),
        compiler_params=plgpu.GPUCompilerParams(
            dimension_semantics=["parallel", "sequential"],
            max_concurrent_steps=max_concurrent_steps,
        ),
        grid=(2, 4),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    x = jnp.arange(128 * 2 * 64).reshape((128 * 2, 64)).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  def test_add_one_grid_pipelined_program_id(self):

    @functools.partial(
        pl.pallas_call,
        out_specs=pl.BlockSpec((16, 16), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct([16, 64], jnp.int32),
        compiler_params=plgpu.GPUCompilerParams(
            dimension_semantics=["parallel", "sequential"],
            max_concurrent_steps=2,
        ),
        grid=(4, 4),
    )
    def kernel(o_ref):
      o_ref[...] = jnp.broadcast_to(pl.program_id(1), o_ref.shape)

    np.testing.assert_array_equal(
        kernel(),
        jnp.repeat(jnp.repeat(jnp.arange(4), 16)[None], 16, axis=0),
    )

  def test_add_one_grid_pipelined_sequential_invariant_output(self):
    @functools.partial(
        pl.pallas_call,
        in_specs=[pl.BlockSpec((32, 16), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((32, 16), lambda i, j: (i, 0)),
        out_shape=jax.ShapeDtypeStruct([32 * 2, 64], jnp.float32),
        compiler_params=plgpu.GPUCompilerParams(
            dimension_semantics=["parallel", "sequential"],
            max_concurrent_steps=2,
        ),
        grid=(2, 4),
    )
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    x = jnp.arange(32 * 2 * 64).reshape((32 * 2, 64)).astype(jnp.float32)
    y = jnp.empty_like(x)
    for i in range(2):
      i_slice = slice(32 * i, 32 * (i + 1))
      for j in range(4):
        j_slice = slice(16 * j, 16 * (j + 1))
        y = y.at[i_slice, :16].set(x[i_slice, j_slice] + 1)

    # We only compare the elements in the first 16 columns, because the rest
    # are never written to.
    np.testing.assert_array_equal(kernel(x)[:, :16], y[:, :16])

  def test_iota(self):
    dtype, dimension = jnp.int8, 1
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((128, 128), dtype),
    )
    def kernel(o_ref):
      o_ref[...] = plgpu.broadcasted_iota(dtype, (128, 128), dimension, layout=plgpu.Layout.WGMMA)

    np.testing.assert_array_equal(kernel(), jax.lax.broadcasted_iota(dtype, (128, 128), dimension))

  @parameterized.product(indexer=[..., slice(128), slice(None, 128)])
  def test_copy_smem_to_gmem(self, indexer):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        scratch_shapes=[plgpu.SMEM((256,), jnp.float32)],
    )
    def kernel(x_ref, o_ref_gmem, scratch_ref):
      scratch_ref[...] = x_ref[...] + 1
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(scratch_ref.at[indexer], o_ref_gmem.at[indexer])
      plgpu.wait_smem_to_gmem(0)

    x = jnp.arange(256).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x)[indexer], x[indexer] + 1.0)

  @parameterized.product(indexer=[..., slice(128), slice(None, 128)])
  def test_copy_gmem_to_smem(self, indexer):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
        in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),),
        scratch_shapes=[
            plgpu.SMEM((256,), jnp.float32),
            plgpu.Barrier(num_arrivals=1),
        ],
    )
    def kernel(x_ref_gmem, o_ref, scratch_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(
          x_ref_gmem.at[indexer], scratch_ref.at[indexer], barrier_ref
      )
      plgpu.barrier_wait(barrier_ref)
      o_ref[...] = scratch_ref[...] + 1

    x = jnp.arange(256).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x)[indexer], x[indexer] + 1.0)

  @parameterized.product(indexer=[0, 1, 2, 3])
  def test_copy_gmem_to_smem_with_indexed_barrier(self, indexer):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([128], jnp.float32),
        in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),),
        scratch_shapes=[
            plgpu.SMEM((128,), jnp.float32),
            plgpu.Barrier(num_arrivals=1, num_barriers=4),
        ],
    )
    def kernel(x_ref_gmem, o_ref, scratch_ref, barrier_ref):
      plgpu.copy_gmem_to_smem(
          x_ref_gmem, scratch_ref, barrier_ref.at[indexer]
      )
      plgpu.barrier_wait(barrier_ref.at[indexer])
      o_ref[...] = scratch_ref[...] + 1

    x = jnp.arange(128).astype(jnp.float32)
    np.testing.assert_array_equal(kernel(x), x + 1.0)

  @parameterized.named_parameters(("_g2s", False), ("_s2g", True))
  def test_copy_with_transforms(self, to_smem):
    def kernel(x_ref, o_ref, barrier_ref):
      if to_smem:
        plgpu.copy_gmem_to_smem(x_ref, o_ref, barrier_ref)
        plgpu.barrier_wait(barrier_ref)
      else:
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(x_ref, o_ref)
        plgpu.wait_smem_to_gmem(0)

    in_spec = pl.BlockSpec(memory_space=plgpu.GMEM)
    out_spec = plgpu.GPUBlockSpec(
        (128, 128),
        lambda: (0, 0),
        transforms=(
            plgpu.TilingTransform((64, 32)),
            plgpu.SwizzleTransform(128),
        ),
        memory_space=plgpu.SMEM,
    )
    if not to_smem:
      in_spec, out_spec = out_spec, in_spec
    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct([128, 128], jnp.float32),
        in_specs=(in_spec,),
        out_specs=out_spec,
        scratch_shapes=[plgpu.Barrier(num_arrivals=1)],
    )
    x = jnp.arange(128 * 128, dtype=jnp.float32).reshape(128, 128)
    np.testing.assert_array_equal(f(x), x)

  def test_scoped_copy_with_transforms(self):
    ts = (plgpu.TilingTransform((64, 32)), plgpu.SwizzleTransform(128))
    def kernel(x_ref, o_ref, barrier_ref):
      def body(tmp_ref):
        plgpu.copy_gmem_to_smem(x_ref, tmp_ref, barrier_ref)
        plgpu.barrier_wait(barrier_ref)
        o_ref[...] = tmp_ref[...] * 2
      pl.run_scoped(body, plgpu.SMEM((128, 128), jnp.float32, transforms=ts))

    in_spec = pl.BlockSpec(memory_space=plgpu.GMEM)
    out_spec = plgpu.GPUBlockSpec(
        (128, 128), lambda: (0, 0), transforms=ts, memory_space=plgpu.SMEM,
    )
    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct([128, 128], jnp.float32),
        in_specs=(in_spec,),
        out_specs=out_spec,
        scratch_shapes=[plgpu.Barrier(num_arrivals=1)],
    )
    x = jnp.arange(128 * 128, dtype=jnp.float32).reshape(128, 128)
    np.testing.assert_array_equal(f(x), x * 2)

  def test_copy_with_transforms_and_indexing(self):
    def kernel(x_ref, o_ref, barrier_ref):
      for i in range(2):
        plgpu.copy_gmem_to_smem(x_ref, o_ref.at[i], barrier_ref)
        plgpu.barrier_wait(barrier_ref)

    in_spec = pl.BlockSpec(memory_space=plgpu.GMEM)
    out_spec = plgpu.GPUBlockSpec(
        (2, 128, 128),
        lambda: (0, 0, 0),
        transforms=(
            plgpu.TilingTransform((64, 32)),
            plgpu.TransposeTransform((0, 2, 1, 3, 4)),
            plgpu.SwizzleTransform(128),
        ),
        memory_space=plgpu.SMEM,
    )
    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct([2, 128, 128], jnp.float32),
        in_specs=(in_spec,),
        out_specs=out_spec,
        scratch_shapes=[plgpu.Barrier(num_arrivals=1)],
    )
    x = jnp.arange(128 * 128, dtype=jnp.float32).reshape(128, 128)
    np.testing.assert_array_equal(f(x), np.stack([x, x], axis=0))

  def test_indexing_before_transpose(self):
    def kernel(x_ref, o_ref, barrier_ref):
      for i in range(2):
        plgpu.copy_gmem_to_smem(
            x_ref, plgpu.transpose_ref(o_ref.at[i], (1, 0, 2)), barrier_ref
        )
        plgpu.barrier_wait(barrier_ref)

    in_spec = pl.BlockSpec(memory_space=plgpu.GMEM)
    out_spec = plgpu.GPUBlockSpec(
        (2, 64, 2, 128), lambda: (0, 0, 0, 0), memory_space=plgpu.SMEM,
    )
    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct([2, 64, 2, 128], jnp.float32),
        in_specs=(in_spec,),
        out_specs=out_spec,
        scratch_shapes=[plgpu.Barrier(num_arrivals=1)],
    )
    x = jnp.arange(2 * 64 * 128, dtype=jnp.float32).reshape(2, 64, 128)
    xt = x.transpose((1, 0, 2))
    np.testing.assert_array_equal(f(x), np.stack([xt, xt], axis=0))

  def test_copy_gmem_to_smem_in_run_scoped(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
        in_specs=(pl.BlockSpec(memory_space=plgpu.GMEM),),
    )
    def kernel(x_ref_gmem, o_ref):
      def body(barrier_ref):
        def inner_body(scratch_ref):
          plgpu.copy_gmem_to_smem(x_ref_gmem, scratch_ref, barrier_ref)
          plgpu.barrier_wait(barrier_ref)
          o_ref[...] = scratch_ref[...] + 1
        pl.run_scoped(inner_body, plgpu.SMEM((256,), jnp.float32))
      pl.run_scoped(body, plgpu.Barrier(num_arrivals=1))

    x = jnp.arange(256).astype(jnp.float32)
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
    with self.capture_stdout() as output:
      jax.block_until_ready(kernel(x))
    self.assertEqual(output(), "It works!\n")

  def test_print_wgmma_tiled_layout(self):
    shape = (128, 64)
    size = math.prod(shape)
    def kernel(x_ref, o_ref):
      pl.debug_print("{}", x_ref[...])
    spec = plgpu.GPUBlockSpec(shape, lambda: (0, 0), transforms=(plgpu.TilingTransform((64, 32)), plgpu.SwizzleTransform(128)))
    x = jnp.arange(size, dtype=jnp.float32).reshape(shape)
    f = pl.pallas_call(kernel, out_shape=x, in_specs=[spec], out_specs=spec)

    with self.capture_stdout() as get_output:
      jax.block_until_ready(f(x))

    output = get_output()
    results = re.findall(r"\[(\d+), (\d+)\]/\[128, 64\]: (\d+)", output)
    self.assertLen(results, size)
    for i, j, v in results:
      i, j, v = map(int, (i, j, v))
      self.assertEqual(v, i * shape[1] + j)

  def test_print_scalar(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
    )
    def kernel(x_ref, o_ref):
      del o_ref
      pl.debug_print("x.sum() = {}", x_ref[...].sum())

    x = jnp.arange(256)
    with self.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertIn(f"x.sum() = {x.sum()}", output())

  def test_print_scalar_array(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
    )
    def kernel(x_ref, o_ref):
      del o_ref
      pl.debug_print("x.sum() = {}", x_ref[...].sum() + 1)

    x = jnp.arange(256)
    with self.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertIn(f"x.sum() = {x.sum() + 1}", output())

  def test_print_array(self):
    in_shape = [2, 1, 64, 64]

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct(in_shape, jnp.int32),
    )
    def kernel(x_ref, o_ref):
      del o_ref
      pl.debug_print("x: {}", x_ref[...])

    x = jnp.arange(math.prod(in_shape)).reshape(in_shape)
    with self.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertIn(f"x: [1, 0, 43, 23]/{in_shape}: 6871\n", output())

  def test_run_scoped(self):
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

  def test_program_id_in_block_spec(self):
    @functools.partial(
        pl.pallas_call,
        out_specs=pl.BlockSpec((128,), lambda *_: pl.program_id(0)),
        out_shape=jax.ShapeDtypeStruct([128 * 2], jnp.int32),
        grid=2,
    )
    def kernel(o_ref):
      del o_ref

    # ``assertRaises`` have no way of asserting against the cause, so we
    # have to use ``traceback.format_exception`` manually.
    with self.assertRaises(Exception) as exc_info:
      kernel()
    self.assertIn(
        "not supported in this context",
        "".join(traceback.format_exception(exc_info.exception)),
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

    spec = plgpu.GPUBlockSpec(
        (128, 64),
        lambda *i: i,
        transforms=(
            plgpu.TilingTransform((64, 64)),
            plgpu.SwizzleTransform(128),
        ),
    )
    @functools.partial(
        pl.pallas_call,
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float16),
        grid=(2, 2),
    )
    def kernel(x_ref, o_ref):
      assert x_ref.shape == (128, 64), x_ref.shape
      o_ref[...] = x_ref[...]

    x = jnp.arange(128 * 128).astype(jnp.float16).reshape(128, 128)
    np.testing.assert_array_equal(kernel(x), x)

  def test_fori_loop_array(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
    )
    def kernel(x_ref, o_ref):
      # Equivalent to x_ref[...] + 2 + 3.
      o_ref[...] = jax.lax.fori_loop(2, 4, lambda i, x: x + i, x_ref[...])

    x = jnp.arange(256).astype(jnp.int32)
    np.testing.assert_array_equal(kernel(x), x + 2 + 3)

  def test_fori_loop_scalar(self):

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
    )
    def kernel(o_ref):
      # Equivalent to 2 + 3.
      o_ref[...] = jax.lax.broadcast(
          jax.lax.fori_loop(2, 4, lambda i, x: x + i, 0), o_ref.shape
      )

    np.testing.assert_array_equal(kernel(), jnp.full([256], 5, dtype=jnp.int32))

  def test_fori_loop_dynamic_bounds(self):

    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
        grid=(1,)
    )
    def kernel(o_ref):
      zero = pl.program_id(0)
      # Equivalent to 2 + 3.
      o_ref[...] = jax.lax.broadcast(
          jax.lax.fori_loop(2 + zero, 4 + zero, lambda i, x: x + i, 0), o_ref.shape
      )

    np.testing.assert_array_equal(kernel(), jnp.full([256], 5, dtype=jnp.int32))

  def test_fori_loop_tuple(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
    )
    def kernel(o_ref):
      def body(step, xs):
        return tuple(
            jax.lax.cond(step % 2 == 0, lambda x: x + 1, lambda x: x, x)
            for x in xs
        )

      # Equivalent to 3 * (0 + 1).
      o_ref[...] = jax.lax.broadcast(
          sum(jax.lax.fori_loop(2, 4, body, (0, 0, 0))), o_ref.shape
      )

    np.testing.assert_array_equal(
        kernel(), jnp.full([256], 3 * (0 + 1), dtype=jnp.int32)
    )

  def test_fori_loop_indexed_store(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([4, 128], jnp.float32),
    )
    def kernel(x_ref, y_ref, o_ref):
      def body(idx, _):
        o_ref[idx] = x_ref[idx] + y_ref[idx]
        return ()

      jax.lax.fori_loop(0, 4, body, ())

    x = jnp.arange(4 * 128).reshape(4, 128).astype(jnp.float32)
    y = x + 1
    np.testing.assert_array_equal(kernel(x, y), x + y)

  def test_cond(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
    )
    def kernel(x_ref, o_ref):
      acc = x_ref[...].sum()
      jax.lax.cond(
          acc % 2 == 0,
          lambda: pl.debug_print("acc * 2: {}", acc * 2),
          lambda: pl.debug_print("acc: {}", acc),
      )
      o_ref[...] = jnp.broadcast_to(acc, o_ref.shape)

    x = jnp.arange(256)
    with self.capture_stdout() as output:
      jax.block_until_ready(kernel(x))

    self.assertIn("acc * 2:", output())

  def test_cond_returning_array(self):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct([256], jnp.int32),
    )
    def kernel(x_ref, o_ref):
      acc = x_ref[...].sum()
      acc2, acc = jax.lax.cond(
          acc % 2 == 0,
          lambda: (acc * 2, acc),
          lambda: (acc, acc * 2),
      )
      o_ref[...] = jnp.broadcast_to(acc + acc2, o_ref.shape)

    x = jnp.arange(256)
    np.testing.assert_array_equal(kernel(x), jnp.broadcast_to(jnp.sum(x) * 3, [256]))


  @parameterized.parameters(jnp.float16, jnp.float32)
  def test_wgmma(self, dtype):
    self.skip_unless_sm90a()
    # TensorCores can only fuse transposes of 16-bit values, and RHS
    # is expected to be column major by default.
    rhs_transpose = jnp.dtype(dtype).itemsize != 2
    swizzle = 128
    elems_128b = swizzle // jnp.dtype(dtype).itemsize
    def kernel(a_ref, b_ref, o_ref):
      if rhs_transpose:
        b_ref = plgpu.transpose_ref(b_ref, (1, 0))
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref, b_ref)
        return acc_ref[...]

      o_ref[...] = pl.run_scoped(scope, plgpu.ACC((64, 192), jnp.float32))

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(64, 128), dtype=dtype)
    b_shape = (128, 192)
    if rhs_transpose:
      b_shape = b_shape[::-1]
    b = jax.random.uniform(key2, shape=b_shape, dtype=dtype)

    rhs_transforms = (plgpu.TilingTransform((elems_128b, elems_128b)),)
    if rhs_transpose:
      rhs_transforms += (plgpu.TransposeTransform((1, 0, 2, 3)),)
    res = pl.pallas_call(
        kernel,
        in_specs=[
            plgpu.GPUBlockSpec(
                (64, 128),
                lambda i, j: (i, j),
                transforms=(
                    plgpu.TilingTransform((64, elems_128b)),
                    plgpu.SwizzleTransform(128),
                ),
            ),
            plgpu.GPUBlockSpec(
                b_shape,
                lambda *i: i,
                transforms=(*rhs_transforms, plgpu.SwizzleTransform(128)),
            ),
        ],
        out_specs=plgpu.GPUBlockSpec((64, 192), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct((64, 192), jnp.float32),
        grid=(1, 1),
    )(a, b)
    np.testing.assert_allclose(
        res, a @ (b.T if rhs_transpose else b), rtol=1e-3
    )

  def test_wgmma_registers(self):
    self.skip_unless_sm90a()
    def kernel(a_ref, b_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref[...], b_ref)
        return acc_ref[...]
      o_ref[...] = pl.run_scoped(scope, plgpu.ACC((64, 192), jnp.float32))

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(64, 128), dtype=jnp.float16)
    b = jax.random.uniform(key2, shape=(128, 192), dtype=jnp.float16)

    transforms = (plgpu.TilingTransform((64, 64)), plgpu.SwizzleTransform(128))
    res = pl.pallas_call(
        kernel,
        in_specs=[
            plgpu.GPUBlockSpec((64, 128), lambda: (0, 0), transforms=transforms),
            plgpu.GPUBlockSpec((128, 192), lambda: (0, 0), transforms=transforms),
        ],
        out_specs=plgpu.GPUBlockSpec((64, 192), lambda: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((64, 192), jnp.float32),
    )(a, b)
    np.testing.assert_allclose(res, a @ b, rtol=1e-3)

  def test_wgmma_registers_init(self):
    self.skip_unless_sm90a()
    def kernel(a_ref, b_ref, i_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref[...], b_ref)
      o_ref[...] = pl.run_state(scope)(plgpu.ACC.init(i_ref[...]))

    key1, key2, key3 = jax.random.split(jax.random.key(42), 3)
    a = jax.random.uniform(key1, shape=(64, 128), dtype=jnp.float16)
    b = jax.random.uniform(key2, shape=(128, 192), dtype=jnp.float16)
    i = jax.random.uniform(key3, shape=(64, 192), dtype=jnp.float16) * 10

    transforms = (plgpu.TilingTransform((64, 64)), plgpu.SwizzleTransform(128))
    res = pl.pallas_call(
        kernel,
        in_specs=[
            plgpu.GPUBlockSpec((64, 128), lambda: (0, 0), transforms=transforms),
            plgpu.GPUBlockSpec((128, 192), lambda: (0, 0), transforms=transforms),
            plgpu.GPUBlockSpec((64, 192), lambda: (0, 0), transforms=transforms),
        ],
        out_specs=plgpu.GPUBlockSpec((64, 192), lambda: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((64, 192), jnp.float16),
    )(a, b, i)
    np.testing.assert_allclose(res, i + a @ b, rtol=2e-3)

  def test_wgmma_sliced_ref(self):
    self.skip_unless_sm90a()
    def kernel(a_ref, b_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref.at[0], b_ref.at[0])
        return acc_ref[...]

      o_ref[...] = pl.run_scoped(scope, plgpu.ACC((64, 192), jnp.float32))

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(2, 64, 128), dtype=jnp.float16)
    b = jax.random.uniform(key2, shape=(2, 128, 192), dtype=jnp.float16)

    res = pl.pallas_call(
        kernel,
        in_specs=[
            plgpu.GPUBlockSpec(
                (2, 64, 128), lambda: (0, 0, 0),
                transforms=(
                    plgpu.TilingTransform((64, 64)),
                    plgpu.SwizzleTransform(128),
                ),
            ),
            plgpu.GPUBlockSpec(
                (2, 128, 192), lambda: (0, 0, 0),
                transforms=(
                    plgpu.TilingTransform((64, 64)),
                    plgpu.SwizzleTransform(128),
                ),
            ),
        ],
        out_specs=plgpu.GPUBlockSpec((64, 192), lambda: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((64, 192), jnp.float32),
    )(a, b)
    np.testing.assert_allclose(res, a[0] @ b[0], rtol=1e-3)

  def test_wgmma_sliced_acc(self):
    self.skip_unless_sm90a()
    swizzle = 128
    elems_128b = swizzle // jnp.dtype(jnp.float16).itemsize
    def kernel(a_ref, b_ref, o_ref):
      def scope(acc_ref):
        plgpu.wgmma(acc_ref, a_ref, b_ref)
        return acc_ref[:, :64], acc_ref[:, 64:]

      o_ref[:, :64], o_ref[:, 64:] = pl.run_scoped(scope, plgpu.ACC((64, 128), jnp.float32))

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(64, 128), dtype=jnp.float16)
    b = jax.random.uniform(key2, shape=(128, 128), dtype=jnp.float16)
    res = pl.pallas_call(
        kernel,
        in_specs=[
            plgpu.GPUBlockSpec(
                (64, 128),
                lambda i, j: (i, j),
                transforms=(
                    plgpu.TilingTransform((64, elems_128b)),
                    plgpu.SwizzleTransform(128),
                ),
            ),
            plgpu.GPUBlockSpec(
                (128, 128),
                lambda *i: i,
                transforms=(
                    plgpu.TilingTransform((elems_128b, elems_128b)),
                    plgpu.SwizzleTransform(128),
                ),
            ),
        ],
        out_specs=plgpu.GPUBlockSpec((64, 128), lambda *i: i),
        out_shape=jax.ShapeDtypeStruct((64, 128), jnp.float32),
        grid=(1, 1),
    )(a, b)
    np.testing.assert_allclose(res, a @ b, rtol=1e-3)

  def test_input_output_aliases(self):
    # Note that we're writing to the input pointer, which should alias b_ptr.
    def kernel(a_ref, b_ref):
      del b_ref
      a_ref[...] = jnp.ones_like(a_ref)

    a = np.zeros((64, 64), dtype=jnp.float32)
    b = pl.pallas_call(
        kernel,
        in_specs=[plgpu.GPUBlockSpec(memory_space=plgpu.GPUMemorySpace.GMEM)],
        out_specs=plgpu.GPUBlockSpec(memory_space=plgpu.GPUMemorySpace.GMEM),
        input_output_aliases={0: 0},
        out_shape=a,
    )(a)
    np.testing.assert_array_equal(b, np.ones_like(a))

  def test_realistic_matmul(self):
    self.skip_unless_sm90a()
    dtype = jnp.float16
    swizzle = 128
    elems_128b = swizzle // jnp.dtype(dtype).itemsize
    grid_m, grid_k, grid_n = 132, 10, 4
    tile_m = tile_n = 128
    tile_k = elems_128b
    m, k, n = grid_m * tile_m, grid_k * tile_k, grid_n * tile_n
    def kernel(a_ref, b_ref, o_ref, acc_ref):
      # Make sure tiling does not alter the shape of references
      assert a_ref.shape == (tile_m, tile_k)
      assert b_ref.shape == (tile_k, tile_n)
      assert o_ref.shape == acc_ref.shape == (tile_m, tile_n)
      plgpu.wgmma(acc_ref, a_ref, b_ref)
      is_last_step = pl.program_id(2) == grid_k - 1
      @pl.when(is_last_step)
      def _epilogue():
        o_ref[...] = acc_ref[...].astype(dtype)
      plgpu.wgmma_wait(1)  # We don't await the last WGMMA, hence delay_release=1

    key1, key2 = jax.random.split(jax.random.key(42), 2)
    a = jax.random.uniform(key1, shape=(m, k), dtype=dtype)
    b = jax.random.uniform(key2, shape=(k, n), dtype=dtype)

    res = pl.pallas_call(
        kernel,
        in_specs=[
            plgpu.GPUBlockSpec(
                (tile_m, tile_k),
                lambda m, n, k: (m, k),
                transforms=(
                    plgpu.TilingTransform((64, elems_128b)),
                    plgpu.SwizzleTransform(128),
                ),
            ),
            plgpu.GPUBlockSpec(
                (tile_k, tile_n),
                lambda m, n, k: (k, n),
                transforms=(
                    plgpu.TilingTransform((elems_128b, elems_128b)),
                    plgpu.SwizzleTransform(128),
                ),
            ),
        ],
        out_specs=plgpu.GPUBlockSpec(
            (tile_m, tile_n),
            lambda m, n, k: (m, n),
            transforms=(
                plgpu.TilingTransform((64, elems_128b)),
                plgpu.SwizzleTransform(128),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.float16),
        scratch_shapes=[plgpu.ACC((tile_m, tile_n), jnp.float32)],
        grid=(grid_m, grid_n, grid_k),
        compiler_params=plgpu.GPUCompilerParams(
            dimension_semantics=["parallel", "parallel", "sequential"],
            max_concurrent_steps=2,
            delay_release=1,
        ),
    )(a, b)
    np.testing.assert_allclose(res, a @ b, rtol=1e-3)

  def test_slicing(self):
    left = upper = slice(None, 64)
    right = lower = slice(64, None)
    # We rotate the four quadrants of the input clockwise.
    def rotate(src, dst):
      dst[upper, left] = src[lower, left]
      dst[upper, right] = src[upper, left]
      dst[lower, right] = src[upper, right]
      dst[lower, left] = src[lower, right]

    x = jnp.arange(128 * 128).astype(jnp.float16).reshape(128, 128)
    spec = plgpu.GPUBlockSpec(
        (128, 128),
        lambda: (0, 0),
        transforms=(
            plgpu.TilingTransform((64, 64)),
            plgpu.SwizzleTransform(128),
        ),
    )
    f = pl.pallas_call(rotate, out_shape=x, in_specs=[spec], out_specs=spec)
    expected = np.empty_like(x)
    rotate(x, expected)
    np.testing.assert_array_equal(f(x), expected)

  def test_layout_cast(self, shape=(256, 64)):
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
    )
    def kernel(o_ref):
      o_ref[...] = plgpu.layout_cast(jnp.full(shape, 42.0), plgpu.Layout.WGMMA)

    x = jnp.full(shape, 42.0)
    np.testing.assert_array_equal(kernel(), x)

  def test_profiler(self):
    def kernel(x_ref, o_ref):
      with jax.named_scope("add"):
        with jax.named_scope("load"):
          x = x_ref[...]
        o = x + x
      with jax.named_scope("store"):
        o_ref[...] = o
    with tempfile.TemporaryDirectory() as tmpdir:
      x = jnp.arange(256).astype(jnp.float32)
      y = pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct([256], jnp.float32),
          compiler_params=plgpu.GPUCompilerParams(
              profile_space=16, profile_dir=tmpdir
          ),
      )(x)
      jax.block_until_ready(y)
      jax.effects_barrier()
      [name] = os.listdir(tmpdir)
      with open(os.path.join(tmpdir, name), "r") as f:
        data = f.read()
        self.assertEqual(data.count('"name": "add"'), 2)
        self.assertEqual(data.count('"name": "load"'), 2)
        self.assertEqual(data.count('"name": "store"'), 2)
      np.testing.assert_array_equal(y, x + x)

  @parameterized.parameters(
      (jnp.float16, jnp.float16),  # Noop
      (jnp.int16, jnp.bfloat16),
      (jnp.int16, jnp.float16),
      (jnp.uint16, jnp.float16),
      (jnp.float32, jnp.int32),
      (jnp.float32, jnp.uint32),
      (jnp.uint32, jnp.int32),
      (jnp.int32, jnp.uint32),
  )
  def test_bitcast_convert_type(self, in_dtype, out_dtype):
    m, n = 16, 8
    out_shape = jax.ShapeDtypeStruct((m, n), out_dtype)
    grid = ()

    @functools.partial(pl.pallas_call, out_shape=out_shape, grid=grid)
    def convert(x_ref, y_ref):
      y_ref[...] = jax.lax.bitcast_convert_type(x_ref[...], out_shape)

    x = jnp.arange(m * n, dtype=in_dtype).reshape((m, n))
    y = convert(x)
    y_ref = jax.lax.bitcast_convert_type(x, out_dtype)
    np.testing.assert_array_equal(y, y_ref)


class PipelineTest(PallasTest):

  def test_manual(self):
    max_concurrent_steps = 2
    num_steps = 4

    def kernel(x_gmem, o_gmem):
      return pl.run_scoped(
          functools.partial(scoped_kernel, x_gmem, o_gmem),
          plgpu.SMEM((max_concurrent_steps, 32, 16), jnp.float32),
          plgpu.SMEM((max_concurrent_steps, 32, 16), jnp.float32),
          plgpu.Barrier(1, num_barriers=max_concurrent_steps),
      )

    def scoped_kernel(x_gmem, o_gmem, x_smem, o_smem, barrier):
      gmem_slice = pl.ds(pl.program_id(0) * 32, 32)

      def body(step, _):
        slot = step % max_concurrent_steps

        # Wait for the current GMEM->SMEM copy to complete.
        plgpu.barrier_wait(barrier.at[slot])
        # Wait for the previous output SMEM->GMEM copy to complete.
        plgpu.wait_smem_to_gmem(max_concurrent_steps - 1)

        o_smem.at[slot][...] = x_smem.at[slot][...] + 1.0

        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(
            o_smem.at[slot], o_gmem.at[gmem_slice, pl.ds(step * 16, 16)]
        )

        fetch_step = step + max_concurrent_steps
        fetch_slot = slot  # (x + y) % y == x % y
        jax.lax.cond(
            fetch_step < num_steps,
            lambda: plgpu.copy_gmem_to_smem(
                x_gmem.at[gmem_slice, pl.ds(fetch_step * 16, 16)],
                x_smem.at[fetch_slot],
                barrier.at[fetch_slot],
            ),
            lambda: None,
        )
        return ()

      # Initialize the pipeline.
      for slot in range(min(max_concurrent_steps, num_steps)):
        plgpu.copy_gmem_to_smem(
            x_gmem.at[gmem_slice, pl.ds(slot * 16, 16)],
            x_smem.at[slot],
            barrier.at[slot],
        )

      jax.lax.fori_loop(0, num_steps, body, ())

      # Finalize the pipeline.
      plgpu.wait_smem_to_gmem(0)

    x = jnp.arange(32 * 4 * 64).reshape(32 * 4, 64).astype(jnp.float32)
    kernel_fn = pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(4, 1),
    )
    np.testing.assert_array_equal(kernel_fn(x), x + 1.0)

  def test_emit(self):
    num_steps = 4

    def kernel(x_gmem, o_gmem):
      plgpu.emit_pipeline(
          kernel_body,
          in_specs=[pl.BlockSpec((32, 16), lambda i: (0, i))],
          out_specs=[pl.BlockSpec((32, 16), lambda i: (0, i))],
          grid=(num_steps,),
          max_concurrent_steps=2,
      )(x_gmem, o_gmem)

    def kernel_body(x_smem, o_smem):
      o_smem[...] = x_smem[...] + 1.0

    x = jnp.arange(32 * num_steps * 16)
    x = x.reshape(-1, num_steps * 16).astype(jnp.float32)
    kernel_fn = pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    np.testing.assert_array_equal(kernel_fn(x), x + 1.0)

  def test_nested_emit(self):
    num_steps = 4

    def kernel(x_gmem, o_gmem):
      plgpu.emit_pipeline(
          nested_kernel,
          in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
          out_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
          grid=(),
      )(x_gmem, o_gmem)

    def nested_kernel(x_gmem, o_gmem):
      plgpu.emit_pipeline(
          nested_kernel_body,
          in_specs=[pl.BlockSpec((32, 16), lambda i: (0, i))],
          out_specs=[pl.BlockSpec((32, 16), lambda i: (0, i))],
          grid=(num_steps,),
          max_concurrent_steps=2,
      )(x_gmem, o_gmem)

    def nested_kernel_body(x_smem, o_smem):
      o_smem[...] = x_smem[...] + 1.0

    x = jnp.arange(32 * num_steps * 16)
    x = x.reshape(-1, num_steps * 16).astype(jnp.float32)
    kernel_fn = pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    np.testing.assert_array_equal(kernel_fn(x), x + 1.0)

  def test_emit_with_grid_invariant_output(self):
    num_steps = 4

    def kernel(x_gmem, o_gmem):
      plgpu.emit_pipeline(
          kernel_body,
          in_specs=[pl.BlockSpec((32, 16), lambda i: (0, i))],
          out_specs=[pl.BlockSpec((32, 16), lambda i: (0, 0))],
          grid=(num_steps,),
          max_concurrent_steps=2,
      )(x_gmem, o_gmem)

    def kernel_body(x_smem, o_smem):
      o_smem[...] = x_smem[...] + 1.0

    x = jnp.arange(32 * num_steps * 16)
    x = x.reshape(-1, num_steps * 16).astype(jnp.float32)
    kernel_fn = pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    y = jnp.empty_like(x)
    for i in range(num_steps):
      i_slice = slice(16 * i, 16 * (i + 1))
      y = y.at[:, :16].set(x[:, i_slice] + 1)
    # We only compare the elements in the first 16 columns, because the rest
    # are never written to.
    np.testing.assert_array_equal(kernel_fn(x)[:, :16], y[:, :16])

  def test_emit_with_parallel_grid(self):
    num_steps1 = 4
    num_steps2 = 5

    def kernel(x_gmem, o_gmem):
      pid = pl.program_id(0)
      plgpu.emit_pipeline(
          kernel_body,
          in_specs=[pl.BlockSpec((32, 16), lambda i: (pid, i))],
          out_specs=[pl.BlockSpec((32, 16), lambda i: (pid, i))],
          grid=(num_steps2,),
          max_concurrent_steps=2,
      )(x_gmem, o_gmem)

    def kernel_body(x_smem, o_smem):
      o_smem[...] = x_smem[...] + 1.0

    x = jnp.arange(num_steps1 * 32 * num_steps2 * 16)
    x = x.reshape(-1, num_steps2 * 16).astype(jnp.float32)
    kernel_fn = pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(num_steps1,),
    )
    y = x + 1.0
    np.testing.assert_array_equal(kernel_fn(x), y)

  def test_emit_with_2d_grid(self):
    num_steps1 = 4
    num_steps2 = 5

    def kernel(x_gmem, o_gmem):
      plgpu.emit_pipeline(
          kernel_body,
          in_specs=[pl.BlockSpec((32, 16, 8), lambda i, j: (0, i, j))],
          out_specs=[pl.BlockSpec((32, 16, 8), lambda i, j: (0, i, j))],
          grid=(num_steps1, num_steps2),
          max_concurrent_steps=2,
      )(x_gmem, o_gmem)

    def kernel_body(x_smem, o_smem):
      o_smem[...] = x_smem[...] + 1.0

    x = jnp.arange(32 * num_steps1 * 16 * num_steps2 * 8)
    x = x.reshape(-1, num_steps1 * 16, num_steps2 * 8).astype(jnp.float32)
    kernel_fn = pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    np.testing.assert_array_equal(kernel_fn(x), x + 1.0)


class CoreMapTest(PallasTest):

  def test_multiple_wg(self):
    mesh = plgpu.GPUMesh(num_threads=2, axis_names=("y",))

    @jax.jit
    def f():
      @pl.run_state
      def inner(y_ref):
        @pl.core_map(mesh)
        def kernel():
          wg_idx = jax.lax.axis_index("y")
          y_ref[wg_idx] = jnp.broadcast_to(wg_idx, (128,))
      y_init = jnp.zeros((2, 128), np.int32)
      return inner(y_init)
    np.testing.assert_array_equal(
        f(), np.repeat(np.arange(2), 128).reshape(2, 128)
    )

  def test_multiple_wg_with_grid(self):
    mesh = plgpu.GPUMesh(grid=(2, 2), num_threads=2, axis_names=("x", "y", "wg"))

    @jax.jit
    def f():
      @pl.run_state
      def inner(y_ref):
        @pl.core_map(mesh)
        def kernel():
          xy_idx = jax.lax.axis_index(("x", "y"))
          yx_idx = jax.lax.axis_index(("y", "x"))
          wg_idx = jax.lax.axis_index("wg")
          num_wgs = jax.lax.psum(1, "wg")
          y_ref[xy_idx, wg_idx] = jnp.broadcast_to(
              yx_idx * num_wgs + wg_idx, (128,)
          )
      y_init = jnp.zeros((4, 2, 128), np.int32)
      return inner(y_init)
    np.testing.assert_array_equal(
        f(), np.repeat([0, 1, 4, 5, 2, 3, 6, 7], 128).reshape(4, 2, 128)
    )

  def test_cross_wg_barrier(self):
    mesh = plgpu.GPUMesh(num_threads=2, axis_names=("wg",))

    @jax.jit
    def f():
      @pl.run_state
      def inner(y_ref):
        @pl.core_map(mesh)
        def kernel():
          def scoped(barrier):
            plgpu.barrier_arrive(barrier)
            plgpu.barrier_wait(barrier)
            wg_idx = jax.lax.axis_index("wg")
            y_ref[wg_idx] = jnp.broadcast_to(wg_idx, (128,))
          # Each warpgroup is a single logical thread!
          pl.run_scoped(scoped, plgpu.Barrier(num_arrivals=2))
      y_init = jnp.zeros((2, 128), np.int32)
      return inner(y_init)
    np.testing.assert_array_equal(f(), np.repeat([0, 1], 128).reshape(2, 128))


if __name__ == "__main__":
  absltest.main()
