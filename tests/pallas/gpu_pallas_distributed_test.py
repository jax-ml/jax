# Copyright 2025 The JAX Authors.
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

"""Tests for distributed pallas GPU operations."""

import functools
import os

from absl.testing import parameterized
import jax
from jax import lax
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
from jax.experimental import multihost_utils
from jax.experimental import pallas as pl
from jax.experimental import shard_map
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.pallas.ops.gpu.reduce_scatter_mgpu import reduce_scatter
import jax.numpy as jnp
import numpy as np


P = jax.sharding.PartitionSpec
partial = functools.partial


class TestCase(jt_multiprocess.MultiProcessTest):

  def setUp(self):
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("Only works on GPU with capability >= sm90")
    if not mgpu.supports_cross_device_collectives():
      self.skipTest("NVSHMEM library unavailable.")
    if jax.process_count() == 1:
      self.skipTest("Test requires multiple processes.")
    if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "") == "platform":
      self.skipTest("NVSHMEM doesn't work with the platform allocator.")
    super().setUp()


class PallasCallRemoteDMATest(TestCase):

  def test_remote_dma_basic(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.
    def kernel(x_ref, y_ref, ready_sem, recv_sem):
      other_dev_id = 1 - lax.axis_index('x')
      y_ref[...] = x_ref[...]
      pl.semaphore_signal(ready_sem, device_id=other_dev_id)
      pl.semaphore_wait(ready_sem)
      neighbor_ptr = plgpu.remote_ref(y_ref, other_dev_id)
      neighbor_ptr[...] = x_ref[...]
      pl.semaphore_signal(recv_sem, device_id=other_dev_id)
      pl.semaphore_wait(recv_sem)

    x = jnp.arange(2 * 8 * 128.0, dtype=jnp.float32).reshape((2 * 8, 128))
    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
          out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
          scratch_shapes=[
              plgpu.SemaphoreType.REGULAR,
              plgpu.SemaphoreType.REGULAR,
          ],
      )(x)

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y = jax.jit(
        shard_map.shard_map(
            body, mesh, in_specs=P('x'), out_specs=P('x'), check_rep=False,
        )
    )(x)

    expected = x[8:] if jax.process_index() == 0 else x[:8]
    np.testing.assert_allclose(y.addressable_shards[0].data, expected)

  @parameterized.parameters(('x',), ('y',))
  def test_remote_dma_2d_mesh(self, axis):
    if jax.process_count() < 4:
      self.skipTest('Test requires at least 4 devices (and processes).')
    if jax.process_index() > 4:
      return  # Only 4 processes needed.
    def kernel(x_ref, y_ref, recv_sem):
      other_dev_id = {axis: 1 - lax.axis_index(axis)}
      other_y_ref = plgpu.remote_ref(y_ref, other_dev_id)
      other_y_ref[...] = x_ref[...]
      pl.semaphore_signal(recv_sem, device_id=other_dev_id)
      pl.semaphore_wait(recv_sem)

    x = jnp.arange(2 * 8 * 128.0, dtype=jnp.float32).reshape((2 * 8, 128))
    def body(x):
      return pl.pallas_call(
          kernel,
          in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
          out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
          scratch_shapes=[plgpu.SemaphoreType.REGULAR],
      )(x)

    devices = jax.devices()[:4]
    mesh = jax.sharding.Mesh(np.asarray(devices).reshape(2, 2), ['x', 'y'])
    y = jax.jit(
        shard_map.shard_map(
            body, mesh, in_specs=P(axis), out_specs=P(axis), check_rep=False,
        )
    )(x)

    expected = x[8:] if jax.process_index() == 0 else x[:8]
    np.testing.assert_allclose(y.addressable_shards[0].data, expected)

  def test_wait_twice(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.

    def kernel(y_ref, sem):
      other_dev_id = 1 - lax.axis_index('x')
      pl.semaphore_signal(sem, 2, device_id=other_dev_id)
      pl.semaphore_wait(sem)
      pl.semaphore_wait(sem)
      y_ref[...] = jnp.ones_like(y_ref)

    kernel_call = pl.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
    )

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y = jax.jit(
        shard_map.shard_map(
            kernel_call, mesh, in_specs=(), out_specs=P(None), check_rep=False,
        )
    )()
    np.testing.assert_allclose(y, jnp.ones_like(y))

  def test_wait_nodec(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.

    def kernel(y_ref, sem):
      other_dev_id = 1 - lax.axis_index('x')
      pl.semaphore_signal(sem, 2, device_id=other_dev_id)
      pl.semaphore_wait(sem, decrement=False)
      pl.semaphore_wait(sem, 2, decrement=False)
      pl.semaphore_wait(sem, 2)
      y_ref[...] = jnp.ones_like(y_ref)

    kernel_call = pl.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
    )

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y = jax.jit(
        shard_map.shard_map(
            kernel_call, mesh, in_specs=(), out_specs=P(None), check_rep=False,
        )
    )()
    np.testing.assert_allclose(y, jnp.ones_like(y))

  def test_signal_parallel(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.

    def kernel(y_ref, sem, sem2):
      other_dev_id = 1 - lax.axis_index('x')
      plgpu.semaphore_signal_parallel(
          plgpu.SemaphoreSignal(sem, device_id=other_dev_id),
          plgpu.SemaphoreSignal(sem2, device_id=other_dev_id),
      )
      pl.semaphore_wait(sem)
      pl.semaphore_wait(sem2)
      y_ref[...] = jnp.ones_like(y_ref)

    kernel_call = pl.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR] * 2,
    )

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ['x'])
    y = jax.jit(
        shard_map.shard_map(
            kernel_call, mesh, in_specs=(), out_specs=P(None), check_rep=False,
        )
    )()
    np.testing.assert_allclose(y, jnp.ones_like(y))

  def test_permuted_mesh(self):
    def kernel(y_ref, sem):
      other_dev_id = 1 - lax.axis_index('x')
      pl.semaphore_signal(sem, 1, device_id=other_dev_id)
      pl.semaphore_wait(sem)

    kernel_call = pl.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
    )
    mesh = jax.sharding.Mesh(jax.devices()[::-1], ['x'])  # Reverse the devices.
    f = jax.jit(
        shard_map.shard_map(
            kernel_call, mesh, in_specs=(), out_specs=P(None), check_rep=False,
        )
    )
    msg = (
        'Mosaic GPU only supports meshes with device ordering that follows'
        ' row-major device ids.'
    )
    with self.assertRaisesRegex(NotImplementedError, msg):
      f()


class PallasCallMultimemTest(TestCase):

  def _get_reduction_impl(self, reduction):
    match reduction:
      case "add":
        return jnp.add
      case "min":
        return jnp.minimum
      case "max":
        return jnp.maximum
      case "and":
        return jnp.bitwise_and
      case "or":
        return jnp.bitwise_or
      case "xor":
        return jnp.bitwise_xor
      case _:
        raise ValueError(reduction)

  def test_multimem_store(self):
    if jax.process_index() > 2:
      return  # Only 2 processes needed.

    def kernel(y_ref, sem):
      @pl.when(lax.axis_index('x') == 0)
      def _store():
        output = plgpu.layout_cast(lax.broadcasted_iota(jnp.int32, (128, 128), 1), plgpu.Layout.WGMMA)
        plgpu.multimem_store(output, y_ref, 'x')
      other_dev_id = 1 - lax.axis_index('x')
      pl.semaphore_signal(sem, 1, device_id=other_dev_id)
      pl.semaphore_wait(sem)

    kernel_call = pl.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((128, 128), jnp.int32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
    )
    mesh = jax.sharding.Mesh(jax.devices(), ['x'])
    y = jax.jit(
        shard_map.shard_map(
            kernel_call, mesh, in_specs=(), out_specs=P("x"), check_rep=False,
        )
    )()
    y = multihost_utils.process_allgather(y, tiled=True)
    ref = lax.broadcasted_iota(jnp.int32, (128, 128), 1)
    np.testing.assert_array_equal(y, np.concat([ref, ref], axis=0))

  @parameterized.parameters(
      (jnp.int32, 1, "add"),
      (jnp.int32, 1, "min"),
      (jnp.int32, 1, "max"),
      (jnp.int32, 1, "and"),
      (jnp.int32, 1, "or"),
      (jnp.int32, 1, "xor"),
      (jnp.float32, 1, "add"),
      (jnp.float32, 2, "add", True),
      (jnp.float32, 4, "add"),
      (jnp.float16, 2, "add"),
      (jnp.float16, 2, "min"),
      (jnp.float16, 4, "max"),
      (jnp.float16, 8, "add", True),
      (jnp.bfloat16, 2, "max"),
      (jnp.bfloat16, 8, "add"),
      (jnp.float8_e5m2, 4, "add"),
      (jnp.float8_e5m2, 8, "min"),
      (jnp.float8_e5m2, 16, "max", True),
      (jnp.float8_e4m3fn, 4, "min", True),
      (jnp.float8_e4m3fn, 8, "max"),
      (jnp.float8_e4m3fn, 16, "add"),
  )
  def test_multimem_load_reduce(self, dtype, vector_length, reduction, tiled_layout=False):
    if dtype in (
        jnp.float8_e5m2,
        jnp.float8_e4m3fn,
    ) and not jtu.is_cuda_compute_capability_at_least("10.0"):
      self.skipTest("Only works on GPU with capability >= sm100")
    if jax.process_index() > 2:
      return  # Only 2 processes needed.
    devices = jax.devices()[:2]

    def kernel(x_ref, y_ref, _, sem_ref):
      if tiled_layout:
        layout = plgpu.Layout.TILED(
            plgpu.Tiling(
                (
                    (64, 2 * vector_length),
                    (16, 2 * vector_length),
                    (vector_length,),
                )
            ),
            warp_dims=(-5,),
            lane_dims=(-3, -2),
            vector_dim=-1,
        )
      else:
        layout = plgpu.Layout.WG_STRIDED((64, 32), vec_size=vector_length)
      y_ref[...] = plgpu.layout_cast(
          plgpu.multimem_load_reduce(
              x_ref.at[16:-16], collective_axes="x", reduction_op=reduction,
          ),
          layout
      )
      my_device = lax.axis_index("x")
      other_device = 1 - my_device
      pl.semaphore_signal(sem_ref, 1, device_id=other_device)
      pl.semaphore_wait(sem_ref)

    # The rounding we see in low precision types seems to be different from
    # what JAX/XLA use.
    match jnp.dtype(dtype).itemsize:
      case 4:
        bound = 800000
      case 2:
        bound = 128
      case 1:
        bound = 4
      case _:
        raise ValueError(f"Unsupported dtype: {dtype}")
    x_local = jax.random.randint(
        jax.random.key(1234), (128 + 64, 32), dtype=jnp.int32, minval=-bound, maxval=bound,
    ).astype(dtype)
    mesh = jax.sharding.Mesh(devices, ("x",))
    x_shard = jax.ShapeDtypeStruct((64 + 32, 32), dtype)
    y_shape = jax.ShapeDtypeStruct((64, 32), dtype)
    y, _ = jax.jit(
        shard_map.shard_map(
            pl.pallas_call(
                kernel,
                in_specs=[pl.BlockSpec(memory_space=plgpu.GMEM)],
                out_specs=[
                    pl.BlockSpec(memory_space=plgpu.SMEM),
                    pl.BlockSpec(memory_space=plgpu.GMEM),
                ],
                out_shape=(y_shape, x_shard),
                scratch_shapes=[plgpu.SemaphoreType.REGULAR],
                # TODO(b/448323639): Without aliasing XLA doesn't actually
                # insert the copy that puts the operand in symmetric memory,
                # which causes the kernel to crash.
                input_output_aliases={0: 1},
            ),
            mesh=mesh,
            in_specs=P("x"),
            out_specs=P("x"),  # Not really, but lets us test.
            check_rep=False,
        )
    )(x_local)
    y = multihost_utils.process_allgather(y, tiled=True)
    np_reduction = self._get_reduction_impl(reduction)
    np.testing.assert_array_equal(
        y.astype(jnp.float32),
        np.tile(np_reduction(x_local[16:64+16], x_local[64+48:128+48]), (2, 1)),
    )

  @parameterized.parameters(
      (jnp.float32, "add", 1),
      (jnp.float16, "add", 2),
      (jnp.bfloat16, "add", 2),
      (jnp.float16, "min", 4),
      (jnp.float16, "max", 8),
  )
  def test_reduce_scatter(self, dtype, reduction, vec_size):
    if jax.process_index() > 2:
      return

    devices = jax.devices()[:2]
    mesh = jax.sharding.Mesh(devices, ["x"])
    x = jax.random.uniform(
        jax.random.key(42), (128, 64), dtype=dtype, minval=-1.0, maxval=1.0
    )

    def body(x):
      return reduce_scatter(
          x, axis_name="x", reduction=reduction, vec_size=vec_size
      )

    y = jax.jit(
        shard_map.shard_map(
            body, mesh, in_specs=P("x"), out_specs=P("x"), check_rep=False
        )
    )(x)

    y = multihost_utils.process_allgather(y, tiled=True)
    np_reduction = self._get_reduction_impl(reduction)
    expected = np_reduction(x[:64], x[64:])
    tol = 1e-5 if reduction == "add" else 0
    np.testing.assert_allclose(y, expected, rtol=tol, atol=tol)


if __name__ == '__main__':
  # This test doesn't work with the platform allocator, so we override it
  # if it's ran alone. If it's part of a larger test suite and the platform
  # allocator is used, setUp will skip the test.
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'
  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'default'
  jt_multiprocess.main()
