# Copyright 2024 The JAX Authors.
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

import contextlib
from absl.testing import absltest
import jax
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()

# Run all tests with 8 CPU devices.
_exit_stack = contextlib.ExitStack()

def setUpModule():
  _exit_stack.enter_context(jtu.set_host_platform_device_count(8))

def tearDownModule():
  _exit_stack.close()

class FooTest(jtu.JaxTestCase):
  def test_device_count_is_8(self):
      self.assertEqual(8, jax.device_count())

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
