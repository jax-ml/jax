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
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
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

  def test_torch(self):
    def kernel(y_ref, sem):
      plgpu.semaphore_signal_multicast(sem, collective_axes='x')
      # Wait for the multicast signal (each device gets signaled by all devices)
      pl.semaphore_wait(sem, 2)  # Wait for signals from both devices
      y_ref[...] = jnp.ones_like(y_ref)

    kernel_jax = pl.pallas_call(
        kernel,
        out_specs=pl.BlockSpec(memory_space=plgpu.GMEM),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[plgpu.SemaphoreType.REGULAR],
    )
    abstract_mesh = jax.sharding.AbstractMesh((2,), ("x",))
    plgpu.as_torch_kernel(kernel_jax, mesh=abstract_mesh)()
    return
    jax.jit(
        jax.shard_map(
            kernel_jax,
            mesh=abstract_mesh,
            in_specs=(),
            out_specs=jax.P(),
            check_vma=False,
        )
    ).trace().lower(
        lowering_platforms=("gpu",)
    )  # doesn't crash



if __name__ == "__main__":
  jt_multiprocess.main()

