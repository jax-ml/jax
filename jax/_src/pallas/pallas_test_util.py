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
"""Pallas test utilities."""
import sys

from jax._src import test_util as jtu
from jax._src.pallas import pallas_call
from jax.experimental import pallas as pl

use_mosaic_gpu = pallas_call._PALLAS_USE_MOSAIC_GPU.value


@jtu.with_config(jax_traceback_filtering="off")
class PallasTest(jtu.JaxTestCase):
  INTERPRET: bool = False

  def setUp(self):
    if not jtu.test_device_matches(['cpu']) and self.INTERPRET:
      self.skipTest('Only run interpret tests on CPU.')
    if not self.INTERPRET:
      # Running on accelerator
      if jtu.test_device_matches(["cpu"]):
        self.skipTest("On CPU the test works only in interpret mode")
      if (jtu.test_device_matches(["cuda"]) and
          not jtu.is_cuda_compute_capability_at_least("8.0")):
        self.skipTest("Only works on GPU with capability >= sm80")
      if (jtu.test_device_matches(["cuda"]) and use_mosaic_gpu and
          not jtu.is_cuda_compute_capability_at_least("9.0")):
        self.skipTest("Mosaic GPU requires capability >= sm90")
      if sys.platform == "win32":
        self.skipTest("Only works on non-Windows platforms")
    super().setUp()

  def pallas_call(self, *args, **kwargs):
    return pl.pallas_call(*args, **kwargs, interpret=self.INTERPRET)


class PallasTPUTest(PallasTest):
  """A test case that only runs on TPUs or in interpret mode on CPU."""

  def setUp(self):
    if not jtu.test_device_matches(['tpu']) and not self.INTERPRET:
      self.skipTest('Test requires TPUs')
    super().setUp()
