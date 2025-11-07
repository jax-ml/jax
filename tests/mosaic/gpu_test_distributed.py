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
from jax._src.lib.mlir.dialects import vector
from jax.experimental.mosaic.gpu import dialect as mgpu_dialect  # pylint: disable=g-importing-member
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import jax.experimental.mosaic.gpu as mgpu
import jax.experimental.mosaic.gpu.fragmented_array as fa


# ruff: noqa: F405
# pylint: disable=g-complex-comprehension
P = jax.sharding.PartitionSpec


class TestCase(parameterized.TestCase):

  def setUp(self):
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
    with jax.set_mesh(mesh):
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
    def kernel(ctx, src, sem, dst, scratch):
      tmp, barrier = scratch
      other_device = arith.subi(arith.constant(i32, 1), ctx.device_id())
      ctx.async_copy(src_ref=src, dst_ref=tmp, barrier=barrier)
      barrier.wait()
      ctx.async_copy(src_ref=tmp, dst_ref=dst, gmem_peer_id=other_device)
      ctx.await_async_copy(0)
      other_sem = mgpu.SemaphoreRef(
          mgpu.utils.memref_ptr(ctx.to_remote(sem, other_device))
      )
      other_sem.signal(1)
      my_sem = mgpu.SemaphoreRef(mgpu.utils.memref_ptr(sem))
      my_sem.wait(1)

    mesh = jax.make_mesh(
        (2,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    with jax.set_mesh(mesh):
      x_np = np.arange(64 * 64, dtype=jnp.float32).reshape(64, 64)
      x = jax.sharding.reshard(x_np, P("x"))
      sem = jax.sharding.reshard(jnp.zeros((1,), dtype=jnp.int32), P())
      y, _ = jax.jit(
          jax.shard_map(
              lambda x, sem: mgpu.as_gpu_kernel(
                  kernel, (1, 1, 1), (128, 1, 1), x, x, (x, mgpu.TMABarrier()), inout_shape=sem
              )(x, sem),
              in_specs=(P("x"), P(None)),
              out_specs=[P("x"), P(None)],
              check_vma=False,
          )
      )(x, sem)
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
    with jax.set_mesh(mesh):
      sem = jax.sharding.reshard(jnp.zeros((1,), dtype=jnp.int32), P())
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

  @parameterized.parameters(1, 2, 4)
  def test_multimem_basic(self, vector_length):
    i32 = ir.IntegerType.get_signless(32)
    index = ir.IndexType.get()
    def kernel(ctx, sem, out, _):
      my_device = ctx.device_id()
      other_device = arith.subi(arith.constant(i32, 1), my_device)
      my_sem = mgpu.SemaphoreRef(mgpu.utils.memref_ptr(sem))
      other_dst = ctx.to_remote(sem, other_device)
      other_sem = mgpu.SemaphoreRef(mgpu.utils.memref_ptr(other_dst))
      with mgpu.when(arith.cmpi(arith.CmpIPredicate.eq, my_device, arith.constant(i32, 0))):
        c = arith.constant(i32, 1)
        vc = vector.broadcast(ir.VectorType.get((vector_length,), i32), c)
        multicast_ref = ctx.to_remote_multicast(out)
        multicast_ref.store(vc, [arith.constant(index, 0)])
      other_sem.signal(arith.constant(i32, 1))
      my_sem.wait(1)

    mesh = jax.make_mesh(
        (2,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    with jax.set_mesh(mesh):
      sem = jax.sharding.reshard(jnp.zeros((1,), dtype=jnp.int32), P())
      out_shape = jax.ShapeDtypeStruct((vector_length,), jnp.int32)
      out, out_sem = jax.jit(
          jax.shard_map(
              mgpu.as_gpu_kernel(
                  kernel, (1, 1, 1), (128, 1, 1), (), out_shape, (), inout_shape=sem
              ),
              out_specs=P("x"),
              check_vma=False,
          )
      )(sem)
      out_sems = multihost_utils.process_allgather(out_sem, tiled=True)
      np.testing.assert_array_equal(out_sems, np.zeros_like(out_sems))
      out = multihost_utils.process_allgather(out, tiled=True)
      np.testing.assert_array_equal(out, np.ones_like(out))

  def test_multimem_store_registers(self):
    i32 = ir.IntegerType.get_signless(32)
    def kernel(ctx, inp, sem, out, _):
      my_device = ctx.device_id()
      other_device = arith.subi(arith.constant(i32, 1), my_device)
      my_sem = mgpu.SemaphoreRef(mgpu.utils.memref_ptr(sem))
      other_dst = ctx.to_remote(sem, other_device)
      other_sem = mgpu.SemaphoreRef(mgpu.utils.memref_ptr(other_dst))
      with mgpu.when(arith.cmpi(arith.CmpIPredicate.eq, my_device, arith.constant(i32, 0))):
        arr = mgpu.FragmentedArray.load_strided(inp, is_signed=True)
        arr.store_untiled(ctx.to_remote_multicast(out), optimized=False)
      other_sem.signal(arith.constant(i32, 1))
      my_sem.wait(1)

    mesh = jax.make_mesh(
        (2,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    with jax.set_mesh(mesh):
      sem = jax.sharding.reshard(jnp.zeros((1,), dtype=jnp.int32), P())
      x = jax.sharding.reshard(jnp.arange(2048, dtype=jnp.int32).reshape(64, 32), P())
      y, out_sem = jax.jit(
          jax.shard_map(
              mgpu.as_gpu_kernel(
                  kernel, (1, 1, 1), (128, 1, 1), x, x, (), inout_shape=sem
              ),
              out_specs=P("x"),
              check_vma=False,
          )
      )(x, sem)
      out_sems = multihost_utils.process_allgather(out_sem, tiled=True)
      np.testing.assert_array_equal(out_sems, np.zeros_like(out_sems))
      y = multihost_utils.process_allgather(y, tiled=True).reshape(2, *x.shape)
      np.testing.assert_array_equal(y, jnp.stack([x, x]))

  def test_multimem_store_tma(self):
    i32 = ir.IntegerType.get_signless(32)
    def kernel(ctx, inp, sem, out, scratch):
      my_device = ctx.device_id()
      other_device = arith.subi(arith.constant(i32, 1), my_device)
      my_sem = mgpu.SemaphoreRef(mgpu.utils.memref_ptr(sem))
      other_dst = ctx.to_remote(sem, other_device)
      other_sem = mgpu.SemaphoreRef(mgpu.utils.memref_ptr(other_dst))
      with mgpu.when(arith.cmpi(arith.CmpIPredicate.eq, my_device, arith.constant(i32, 0))):
        arr = mgpu.FragmentedArray.load_strided(inp, is_signed=True)
        arr.store_untiled(scratch)
        mgpu.commit_shared()
        ctx.async_copy(
            src_ref=scratch, dst_ref=out, gmem_peer_id=mgpu.GLOBAL_BROADCAST
        )
        ctx.await_async_copy(0)
      other_sem.signal(arith.constant(i32, 1))
      my_sem.wait(1)

    mesh = jax.make_mesh(
        (2,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    with jax.set_mesh(mesh):
      sem = jax.sharding.reshard(jnp.zeros((1,), dtype=jnp.int32), P())
      x = jax.sharding.reshard(jnp.arange(2048, dtype=jnp.int32).reshape(64, 32), P())
      y, out_sem = jax.jit(
          jax.shard_map(
              mgpu.as_gpu_kernel(
                  kernel, (1, 1, 1), (128, 1, 1), x, x, x, inout_shape=sem
              ),
              out_specs=P("x"),
              check_vma=False,
          )
      )(x, sem)
      out_sems = multihost_utils.process_allgather(out_sem, tiled=True)
      np.testing.assert_array_equal(out_sems, np.zeros_like(out_sems))
      y = multihost_utils.process_allgather(y, tiled=True).reshape(2, *x.shape)
      np.testing.assert_array_equal(y, jnp.stack([x, x]))

  @parameterized.parameters(
      (jnp.int32, 1, "add"),
      (jnp.int32, 1, "min"),
      (jnp.int32, 1, "max"),
      (jnp.int32, 1, "and"),
      (jnp.int32, 1, "or"),
      (jnp.int32, 1, "xor"),
      (jnp.float32, 1, "add"),
      (jnp.float32, 2, "add"),
      (jnp.float32, 4, "add"),
      (jnp.float16, 2, "add"),
      (jnp.float16, 2, "min"),
      (jnp.float16, 4, "max"),
      (jnp.float16, 8, "add"),
      (jnp.bfloat16, 2, "max"),
      (jnp.bfloat16, 8, "add"),
      (jnp.float8_e5m2, 4, "add"),
      (jnp.float8_e5m2, 8, "min"),
      (jnp.float8_e5m2, 16, "max"),
      (jnp.float8_e4m3fn, 4, "min"),
      (jnp.float8_e4m3fn, 8, "max"),
      (jnp.float8_e4m3fn, 16, "add"),
  )
  def test_multimem_load_reduce(self, dtype, vector_length, reduction):
    if dtype in (
        jnp.float8_e5m2,
        jnp.float8_e4m3fn,
    ) and not jtu.is_cuda_compute_capability_at_least("10.0"):
      self.skipTest("Only works on GPU with capability >= sm100")
    i32 = ir.IntegerType.get_signless(32)
    def kernel(ctx, inp, sem, out, _):
      my_device = ctx.device_id()
      other_device = arith.subi(arith.constant(i32, 1), my_device)
      my_sem = mgpu.SemaphoreRef(mgpu.utils.memref_ptr(sem))
      other_dst = ctx.to_remote(sem, other_device)
      other_sem = mgpu.SemaphoreRef(mgpu.utils.memref_ptr(other_dst))
      layout = fa.WGStridedFragLayout((64, 32), vec_size=vector_length)
      arr = mgpu.FragmentedArray.load_reduce_untiled(
          ctx.to_remote_multicast(inp),
          layout=layout,
          is_signed=True if jnp.issubdtype(dtype, jnp.integer) else None,
          reduction=reduction,
      )
      arr.store_untiled(out, optimized=False)
      other_sem.signal(arith.constant(i32, 1))
      my_sem.wait(1)

    mesh = jax.make_mesh(
        (2,), ("x",), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    with jax.set_mesh(mesh):
      sem = jax.sharding.reshard(jnp.zeros((1,), dtype=jnp.int32), P())
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
          jax.random.key(1234), (128, 32), dtype=jnp.int32, minval=-bound, maxval=bound
      ).astype(dtype)
      x = jax.sharding.reshard(x_local, P("x"))
      x_shard = jax.ShapeDtypeStruct((64, 32), dtype)
      # TODO(b/448323639): We don't need x to be inout here, but without aliasing
      # XLA doesn't actually insert the copy that puts the operand in symmetric
      # memory, which causes the kernel to crash.
      y, _, out_sem = jax.jit(
          jax.shard_map(
              mgpu.as_gpu_kernel(
                  kernel, (1, 1, 1), (128, 1, 1), (), x_shard, (), inout_shape=(x_shard, sem)
              ),
              out_specs=P("x"),
              check_vma=False,
          )
      )(x, sem)
      out_sems = multihost_utils.process_allgather(out_sem, tiled=True)
      np.testing.assert_array_equal(out_sems, np.zeros_like(out_sems))
      y = multihost_utils.process_allgather(y, tiled=True)
      match reduction:
        case "add":
          np_reduction = jnp.add
        case "min":
          np_reduction = jnp.minimum
        case "max":
          np_reduction = jnp.maximum
        case "and":
          np_reduction = jnp.bitwise_and
        case "or":
          np_reduction = jnp.bitwise_or
        case "xor":
          np_reduction = jnp.bitwise_xor
        case _:
          raise ValueError(reduction)
      np.testing.assert_array_equal(
          y.astype(jnp.float32), np.tile(np_reduction(x_local[:64], x_local[64:]), (2, 1))
      )


if __name__ == "__main__":
  # This test doesn't work with the platform allocator, so we override it
  # if it's ran alone. If it's part of a larger test suite and the platform
  # allocator is used, setUp will skip the test.
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.01'
  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'default'
  jt_multiprocess.main()
