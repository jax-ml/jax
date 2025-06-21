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

import unittest

from absl.testing import absltest
import jax
from jax._src import test_util as jtu

try:
  import portpicker
except ImportError:
  portpicker = None

jax.config.parse_flags_with_absl()


@unittest.skipIf(not portpicker, "Test requires portpicker")
class DistributedInitializeTest(jtu.JaxTestCase):

  @jtu.skip_under_pytest(
      """Side effects from jax.distributed.initialize conflict with other tests
      in the same process. pytest runs multiple tests in the same process."""
  )
  def test_is_distributed_initialized(self):
    port = portpicker.pick_unused_port()  # type: ignore
    self.assertFalse(jax.distributed.is_initialized())
    jax.distributed.initialize(f"localhost:{port}", 1, 0)
    self.assertTrue(jax.distributed.is_initialized())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
