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

import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import gpu
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import dialect as mgpu_dialect  # pylint: disable=g-importing-member
from jax.experimental.mosaic.gpu.utils import *  # noqa: F403
import jax.numpy as jnp
import numpy as np

try:
  import torch
except ImportError:
  torch = None


# ruff: noqa: F405
# pylint: disable=g-complex-comprehension
config.parse_flags_with_absl()


class TorchTest(parameterized.TestCase):

  def setUp(self):
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("Only works on GPU with capability >= sm90")
    super().setUp()
    self.prng = np.random.default_rng(1234)
    self.context = mlir.make_ir_context()
    mgpu_dialect.register_dialect(self.context)
    self.enter_context(config.traceback_filtering("off"))
    self.enter_context(self.context)
    self.enter_context(ir.Location.unknown())
    if torch is None:
      raise unittest.SkipTest("Test requires PyTorch")

  def test_basic(self):
    def kernel(ctx, i_gmem, o_gmem, _):
      x = mgpu.FragmentedArray.load_strided(i_gmem)
      (x + x).store_untiled(o_gmem)

    ty = jax.ShapeDtypeStruct((128, 128), jnp.float32)
    x = torch.randn((128, 128), dtype=torch.float, device='cuda')
    f = mgpu.as_torch_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), ty, ty, ())
    y = f(x)
    np.testing.assert_allclose(y.cpu(), x.cpu() * 2)
    del y  # Make sure the destructor runs successfully.

  def test_inout(self):
    def kernel(ctx, src, inout, dst, smem):
      val = memref.load(inout, [])
      gpu.barrier()
      new_val = arith.constant(ir.IntegerType.get_signless(32), 42)
      memref.store(new_val, inout, [])
      x = mgpu.FragmentedArray.load_strided(src, is_signed=True)
      (x + val).store_untiled(dst)
    x = torch.arange(128, dtype=torch.int32, device='cuda')
    y = torch.tensor(2.0, dtype=torch.int32, device='cuda')
    x_ty = jax.ShapeDtypeStruct((128,), jnp.int32)
    y_ty = jax.ShapeDtypeStruct((), jnp.int32)
    kernel = mgpu.as_torch_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x_ty, x_ty, (), inout_shape=y_ty,
    )
    xo, yo = kernel(x, y)
    np.testing.assert_array_equal(xo.cpu(), x.cpu() + 2.0)
    np.testing.assert_array_equal(yo.cpu(), torch.tensor(42, dtype=torch.int32))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
