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

import os

from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src import test_multiprocess as jt_multiprocess
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import memref
from jax.experimental.mosaic.gpu import dialect as mgpu_dialect  # pylint: disable=g-importing-member
from jax.experimental import shard
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
try:
  import jax._src.lib.mosaic_gpu  # noqa: F401
  HAS_MOSAIC_GPU = True
except ImportError:
  HAS_MOSAIC_GPU = False
else:
  import jax.experimental.mosaic.gpu as mgpu


# ruff: noqa: F405
# pylint: disable=g-complex-comprehension
P = jax.sharding.PartitionSpec


class TestCase(parameterized.TestCase):

  def setUp(self):
    if not HAS_MOSAIC_GPU:
      self.skipTest("jaxlib built without Mosaic GPU")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("Only works on GPU with capability >= sm90")
    if not mgpu.supports_cross_device_collectives():
      self.skipTest("NVSHMEM library unavailable.")
    if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "") == "platform":
      self.skipTest("NVSHMEM doesn't work with the platform allocator.")
    if jax.process_count() == 1:
      self.skipTest("Test requires multiple processes.")
    if jax.device_count() != jax.process_count():
      self.skipTest("Need 1 device per process")
    super().setUp()
    self.prng = np.random.default_rng(1234)
    self.context = mlir.make_ir_context()
    if mgpu_dialect is not None:
      mgpu_dialect.register_dialect(self.context)
    self.enter_context(config.traceback_filtering("off"))
    self.enter_context(self.context)
    self.enter_context(ir.Location.unknown())


class ProfilerTest(TestCase):

  def test_get_device_id(self):
    index = ir.IndexType.get()
    def kernel(ctx, dst, _):
      device_id = ctx.device_id()
      memref.store(device_id, dst, [arith.constant(index, 0)])
    mesh = jax.make_mesh(
        (jax.device_count(),), ("x",), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    with jax.sharding.use_mesh(mesh):
      out_shape = jax.ShapeDtypeStruct((1,), jnp.int32)
      y = jax.jit(
          jax.shard_map(
              mgpu.as_gpu_kernel(
                  kernel, (1, 1, 1), (128, 1, 1), (), out_shape, ()
              ),
              out_specs=P("x"),
              check_vma=False,
          )
      )()
      y_np = multihost_utils.process_allgather(y, tiled=True)
      np.testing.assert_array_equal(y_np, np.arange(jax.device_count()))

  def test_remote_async_copy(self):
    i32 = ir.IntegerType.get_signless(32)
    def kernel(ctx, src, dst, scratch):
      tmp, barrier = scratch
      other_device = arith.subi(arith.constant(i32, 1), ctx.device_id())
      ctx.async_copy(src_ref=src, dst_ref=tmp, barrier=barrier)
      barrier.wait()
      ctx.async_copy(src_ref=tmp, dst_ref=dst, gmem_peer_id=other_device)
      ctx.await_async_copy(0)
    mesh = jax.make_mesh(
        (2,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    with jax.sharding.use_mesh(mesh):
      x_np = np.arange(64 * 64, dtype=jnp.float32).reshape(64, 64)
      x = shard.reshard(x_np, P("x"))
      y = jax.jit(
          jax.shard_map(
              lambda x: mgpu.as_gpu_kernel(
                  kernel, (1, 1, 1), (128, 1, 1), x, x, (x, mgpu.TMABarrier())
              )(x),
              out_specs=P("x"),
              check_vma=False,
          )
      )(x)
      y_np = multihost_utils.process_allgather(y, tiled=True)
      np.testing.assert_array_equal(
          y_np, np.concatenate(np.split(x_np, 2)[::-1], axis=0)
      )

  def test_remote_semaphore(self):
    i32 = ir.IntegerType.get_signless(32)
    def kernel(ctx, sem, _):
      my_device = ctx.device_id()
      other_device = arith.subi(arith.constant(i32, 1), my_device)
      my_sem = mgpu.SemaphoreRef(mgpu.utils.memref_ptr(sem))
      other_dst = ctx.to_remote(sem, other_device)
      other_sem = mgpu.SemaphoreRef(mgpu.utils.memref_ptr(other_dst))
      # We signal and wait a different amount on each device to make sure we're
      # really communicating here.
      other_sem.signal(arith.addi(arith.constant(i32, 1), other_device))
      @mgpu.fori(arith.addi(arith.constant(i32, 1), my_device), None)
      def wait_loop(i, _):
        my_sem.wait(1)

    mesh = jax.make_mesh(
        (2,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    with jax.sharding.use_mesh(mesh):
      sem = shard.reshard(jnp.zeros((1,), dtype=jnp.int32), P())
      out_sem = jax.jit(
          jax.shard_map(
              mgpu.as_gpu_kernel(
                  kernel, (1, 1, 1), (128, 1, 1), (), (), (), inout_shape=sem
              ),
              out_specs=P("x"),
              check_vma=False,
          )
      )(sem)
      out_sems = multihost_utils.process_allgather(out_sem, tiled=True)
      np.testing.assert_array_equal(out_sems, np.zeros_like(out_sems))


if __name__ == "__main__":
  # This test doesn't work with the platform allocator, so we override it
  # if it's ran alone. If it's part of a larger test suite and the platform
  # allocator is used, setUp will skip the test.
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'
  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'default'
  jt_multiprocess.main()
