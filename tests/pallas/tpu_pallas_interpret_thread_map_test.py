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
import time

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.lib import jaxlib_extension_version
from jax._src.pallas.mosaic.interpret.thread_map import thread_map
import jax.numpy as jnp

jax.config.parse_flags_with_absl()


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

    def f(core_index, token):
      del core_index
      jax.experimental.io_callback(_barrier, (), ordered=True)
      return token

    thread_map(f, 8, jnp.int32(0))
    self.assertEqual(max_concurrent_calls[0], 8)
    # `thread_map` returns only after all threads have completed, so the final
    # value of `concurrent_calls` should be zero.
    self.assertEqual(concurrent_calls[0], 0)

  def test_threads_do_not_wait_for_computations_queued_behind_callback(self):
    if jaxlib_extension_version < 505:
      self.skipTest('Requires jaxlib_extension_version >= 505')

    # thread_map runs in a host callback of `program`. The CPU client lets 32
    # computations per device be in flight, and computations queued behind
    # `program` wait for it, so the threads must not wait for them.
    def f(core_index, token):
      # Uses `core_index`, so that the threads' computations have a live input
      # on the CPU, like interpret mode's.
      jax.experimental.io_callback(lambda _: None, (), core_index, ordered=True)
      return token

    program = jax.jit(lambda token: thread_map(f, 2, token))
    inc = jax.jit(lambda x: x + 1)

    @jax.jit
    def slow(a):
      for _ in range(16):
        a = jnp.tanh(a @ a)
      return a

    a = jnp.full((1024, 1024), 1e-4, jnp.float32)
    token = jnp.int32(0)
    queued = []

    def dispatch():
      # Compiles first: this thread's configs differ from the test's.
      jax.block_until_ready((slow(a), inc(program(token))))
      slow(a)  # Keeps the device busy, so that `program` waits in the queue.
      out = program(token)
      queued.extend(inc(out) for _ in range(100))

    # A daemon thread dispatches, so that the test can report a deadlock.
    threading.Thread(target=dispatch, daemon=True).start()
    deadline = time.monotonic() + 60
    while len(queued) < 100 or not all(x.is_ready() for x in queued):
      if time.monotonic() > deadline:
        self.fail('Computations did not finish within 60 s.')
      time.sleep(0.01)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
