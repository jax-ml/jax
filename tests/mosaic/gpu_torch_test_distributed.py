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
import unittest

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
import jax.numpy as jnp
import numpy as np
import jax.experimental.mosaic.gpu as mgpu
try:
  import torch
  import torch.distributed as dist
  import torch.distributed._symmetric_memory as symm_mem
except ImportError:
  torch = None


# ruff: noqa: F405
# pylint: disable=g-complex-comprehension


class TorchTest(parameterized.TestCase):

  def setUpClass():
    torch.cuda.set_device("cuda:0")
    torch.set_default_device("cuda")
    if torch is None:
      raise unittest.SkipTest("Test requires torch")
    if not torch.cuda.is_available():
      raise unittest.SkipTest("Test requires torch with CUDA support")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      raise unittest.SkipTest("Only works on GPU with capability >= sm90")
    device_count = torch.cuda.device_count()
    for d1 in range(device_count - 1):
      for d2 in range(d1 + 1, device_count):
        if not torch.cuda.can_device_access_peer(d1, d2):
          raise unittest.SkipTest("Test requires p2p access")
    if jax.process_count() == 1:
      raise unittest.SkipTest("Test requires multiple processes.")
    if jax.device_count() != jax.process_count():
      raise unittest.SkipTest("Need 1 device per process")

    os.environ["RANK"] = str(jax.process_index())
    os.environ["WORLD_SIZE"] = str(jax.process_count())
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5728"
    dist.init_process_group("nccl")
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    assert dist.is_initialized()
    assert symm_mem.is_nvshmem_available()
    symm_mem.set_backend("NVSHMEM")
    symm_mem.empty(1)  # Just to initialize NVSHMEM

  def setUp(self):
    self.prng = np.random.default_rng(1234)
    self.context = mlir.make_ir_context()
    if mgpu_dialect is not None:
      mgpu_dialect.register_dialect(self.context)
    self.enter_context(config.traceback_filtering("off"))
    self.enter_context(self.context)
    self.enter_context(ir.Location.unknown())

  def test_get_device_id(self):
    index = ir.IndexType.get()
    def kernel_body(ctx, dst, _):
      device_id = ctx.device_id()
      memref.store(device_id, dst, [arith.constant(index, 0)])

    out_shape = jax.ShapeDtypeStruct((1,), jnp.int32)
    kernel = mgpu.as_torch_gpu_kernel(
      kernel_body, (1, 1, 1), (128, 1, 1), (), out_shape, ()
    )
    gathered = torch.empty((2,), dtype=torch.int32)
    dist.all_gather_into_tensor(gathered, kernel())
    self.assertEqual(gathered.tolist(), list(range(jax.process_count())))

  def test_remote_semaphore(self):
    if dist.get_world_size() != 2:
      self.skipTest("Test assumes 2 devices")

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

    sem_shape = jax.ShapeDtypeStruct((1,), jnp.int32)
    kernel = mgpu.as_torch_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), (), (), (), inout_shape=sem_shape
    )
    gathered = torch.empty((2,), dtype=torch.int32)
    sem = symm_mem.empty((1,), dtype=torch.int32)
    sem_symm = symm_mem.rendezvous(sem, dist.group.WORLD)
    (sem_again,) = kernel(sem)
    self.assertEqual(sem_again.data_ptr(), sem.data_ptr())
    dist.all_gather_into_tensor(gathered, sem)
    self.assertEqual(gathered.tolist(), [0, 0])


if __name__ == "__main__":
  jt_multiprocess.main()
