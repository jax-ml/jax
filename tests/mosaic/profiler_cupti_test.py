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
"""Tests for Mosaic GPU CUPTI-based profiler."""

from absl.testing import absltest, parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp
try:
  import jax._src.lib.mosaic_gpu  # noqa: F401
  HAS_MOSAIC_GPU = True
except ImportError:
  HAS_MOSAIC_GPU = False
else:
  from jax.experimental.mosaic.gpu import profiler


# ruff: noqa: F405
# pylint: disable=g-complex-comprehension
config.parse_flags_with_absl()

class ProfilerCuptiTest(parameterized.TestCase):

  def setUp(self):
    if not HAS_MOSAIC_GPU:
      self.skipTest("jaxlib built without Mosaic GPU")
    if (not jtu.test_device_matches(["cuda"])):
      self.skipTest("Only works on NVIDIA GPUs")
    super().setUp()
    self.x = jnp.arange(1024 * 1024)
    self.f = lambda x: 2*x

  def test_measure_cupti_explicit(self):
    _, runtime_ms = profiler.measure(self.f, mode="cupti")(self.x)
    self.assertIsInstance(runtime_ms, float)

  def test_measure_per_kernel(self):
    _, runtimes_ms = profiler.measure(self.f, mode="cupti", aggregate=False)(self.x)
    for item in runtimes_ms:
      self.assertIsInstance(item, tuple)
      self.assertEqual(len(item), 2)
      name, runtime_ms = item
      self.assertIsInstance(name, str)
      self.assertIsInstance(runtime_ms, float)

  def test_measure_cupti_repeated(self):
    f_profiled = profiler.measure(self.f, mode="cupti")
    n = 3
    timings = [f_profiled(self.x)[1] for _ in range(n)]
    for item in timings:
      self.assertIsInstance(item, float)

  def test_measure_repeated_interleaved(self):
    # test that kernels run outside of measure() are not captured
    _, timings = profiler.measure(self.f, mode="cupti", aggregate=False)(self.x)
    self.assertEqual(len(timings), 1)
    self.f(self.x)
    _, timings = profiler.measure(self.f, mode="cupti", aggregate=False)(self.x)
    self.assertEqual(len(timings), 1)

  def test_measure_double_subscription(self):
    # This needs to run in a separate process, otherwise it affects the
    # outcomes of other tests since CUPTI state is global.
    self.skipTest("Must run in a separate process from other profiler tests")
    # Initialize profiler manually, which subscribes to CUPTI. There can only
    # be one CUPTI subscriber at a time.
    jax._src.lib.mosaic_gpu._mosaic_gpu_ext._cupti_init()
    with self.assertRaisesRegex(RuntimeError,
      "Attempted to subscribe to CUPTI while another subscriber, "
      "such as Nsight Systems or Nsight Compute, is active."):
      profiler.measure(self.f, aggregate=False)(self.x)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
