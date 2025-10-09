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

"""Thread map test for TPU-specific interpret mode."""

import threading

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as mosaic_interpret


jax.config.parse_flags_with_absl()
jax.config.update('jax_threefry_partitionable', True)


# TODO(jburnim): Figure out how to safely run different instance of TPU
# interpret mode in parallel, and then remove this decorator.
@jtu.thread_unsafe_test_class()
class InterpretThreadMapTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()

    if not jtu.test_device_matches(['cpu']):
      self.skipTest('CPU-only test')

    self.num_devices = jax.device_count()
    if self.num_devices > 1:
      # Workaround for https://github.com/jax-ml/jax/issues/25671
      self.skipTest(f'requires 1 device, found {self.num_devices}')

  def test_thread_map(self):
    barrier = threading.Barrier(8)
    lock = threading.Lock()
    concurrent_calls = [0]
    max_concurrent_calls = [0]

    def _barrier():
      with lock:
        concurrent_calls[0] += 1
        max_concurrent_calls[0] = max(
            max_concurrent_calls[0], concurrent_calls[0])
      barrier.wait()
      with lock:
        concurrent_calls[0] -= 1

    def f(core_index):
      del core_index
      jax.experimental.io_callback(_barrier, (), ordered=True)

    mosaic_interpret._thread_map(f, 8)
    self.assertEqual(max_concurrent_calls[0], 8)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
