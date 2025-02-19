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

import subprocess
import sys
import threading
import unittest

from absl.testing import absltest, parameterized

import jax
from jax._src import distributed
from jax._src import test_util as jtu

try:
  import portpicker
except ImportError:
  portpicker = None

jax.config.parse_flags_with_absl()


@unittest.skipIf(not portpicker, "Test requires portpicker")
class DistributedTest(jtu.JaxTestCase):
  # TODO(phawkins): Enable after https://github.com/jax-ml/jax/issues/11222
  # is fixed.
  @unittest.SkipTest
  def testInitializeAndShutdown(self):
    if not jtu.test_device_matches(["gpu"]):
      self.skipTest("Test only works with GPUs.")
    # Tests the public APIs. Since they use global state, we cannot use
    # concurrency to simulate multiple tasks.
    port = portpicker.pick_unused_port()
    jax.distributed.initialize(
      coordinator_address=f"localhost:{port}", num_processes=1, process_id=0
    )
    jax.distributed.shutdown()

  @parameterized.parameters([1, 2, 4])
  def testConcurrentInitializeAndShutdown(self, n):
    if not jtu.test_device_matches(["gpu"]):
      self.skipTest("Test only works with GPUs.")
    port = portpicker.pick_unused_port()

    def task(i):
      # We can't call the public APIs directly because they use global state.
      state = distributed.State()
      state.initialize(
        coordinator_address=f"localhost:{port}", num_processes=n, process_id=i
      )
      state.shutdown()

    threads = [threading.Thread(target=task, args=(i,)) for i in range(n)]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

  def test_is_distributed_initialized(self):
    # Run in subprocess to isolate side effects from jax.distributed.initialize which conflict with other
    # tests. Unfortunately this can't be avoided by calling jax.distributed.shutdown, as the XLA backend
    # will be warmed up, which yields a RuntimeError on subsequent calls to initialize.
    port = portpicker.pick_unused_port() # type: ignore
    cmd = f"""import jax;
    assert not jax.distributed.is_initialized();
    jax.distributed.initialize('localhost:{port}', 1, 0);
    assert jax.distributed.is_initialized();
    """.replace("\n", ' ')

    result = subprocess.run([sys.executable, "-c", cmd], capture_output=True)
    self.assertEqual(
      result.returncode, 0, msg=f"Test failed with:\n{result.stdout}\n{result.stderr}"
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
