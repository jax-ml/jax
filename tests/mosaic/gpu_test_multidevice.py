# Copyright 2024 The JAX Authors. All Rights Reserved.
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

from absl.testing import absltest, parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax.experimental.mosaic.gpu import dialect as mgpu_dialect  # pylint: disable=g-importing-member
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
config.parse_flags_with_absl()


class TestCase(parameterized.TestCase):

  def setUp(self):
    if not HAS_MOSAIC_GPU:
      self.skipTest("jaxlib built without Mosaic GPU")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("Only works on GPU with capability >= sm90")
    super().setUp()
    self.prng = np.random.default_rng(1234)
    self.context = mlir.make_ir_context()
    if mgpu_dialect is not None:
      mgpu_dialect.register_dialect(self.context)
    self.enter_context(config.traceback_filtering("off"))
    self.enter_context(self.context)
    self.enter_context(ir.Location.unknown())


class ProfilerTest(TestCase):

  def test_multigpu(self):
    if len(jax.devices()) < 2:
      self.skipTest("Need at least 2 devices")
    def kernel(ctx, src, dst, _):
      mgpu.FragmentedArray.load_strided(src).store_untiled(dst)
    x = np.arange(64 * 64, dtype=jnp.float32).reshape(64, 64)
    f = jax.jit(mgpu.as_gpu_kernel(
        kernel, (1, 1, 1), (128, 1, 1), x, x, ()
    ))
    # Make sure we can invoke the same program on different devices.
    for xd in (jax.device_put(x, d) for d in jax.devices()[:2]):
      jax.block_until_ready(f(xd))


if __name__ == "__main__":
  absltest.main(argv=["python"], testLoader=jtu.JaxTestLoader())
