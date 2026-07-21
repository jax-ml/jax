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
import warnings

from jax._src import config
from jax._src import test_util as jtu
from jax._src.pallas import pallas_call as pl_lib


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
      if (jtu.test_device_matches(["cuda"]) and
          config.jax_pallas_use_mosaic_gpu.value and
          not jtu.is_cuda_compute_capability_at_least("9.0")):
        self.skipTest("Mosaic GPU requires capability >= sm90")
      if sys.platform == "win32":
        self.skipTest("Only works on non-Windows platforms")
    super().setUp()

    if jtu.test_device_matches(["gpu"]):
      self.enter_context(warnings.catch_warnings())
      warnings.filterwarnings(
          "ignore",
          category=DeprecationWarning,
          message="The Pallas Triton backend is deprecated",
      )

  def pallas_call(self, *args, **kwargs):
    return pl_lib.pallas_call(*args, **kwargs, interpret=self.INTERPRET)


class PallasTPUTest(PallasTest):
  """A test case that only runs on TPUs or in interpret mode on CPU."""

  def setUp(self):
    if not jtu.test_device_matches(['tpu']) and not self.INTERPRET:
      self.skipTest('Test requires TPUs')
    super().setUp()
