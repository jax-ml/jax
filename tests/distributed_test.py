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

import subprocess
import sys
import unittest

from absl.testing import absltest

from jax._src import test_util as jtu

try:
  import portpicker
except ImportError:
  portpicker = None


@unittest.skipIf(not portpicker, "Test requires portpicker")
class DistributedInitializedTest(jtu.JaxTestCase):
  @jtu.sample_product(num_processes=[1, 2], run_initialize=[True, False])
  def test_is_distributed_initialized_multi_process(self, num_processes, run_initialize):
    port = portpicker.pick_unused_port()

    subprocesses = []
    for process_id in range(num_processes):
      pycmd = (
        "import jax; "
        + (
          f"jax.distributed.initialize('localhost:{port}', {num_processes}, {process_id}); "
          if run_initialize
          else ""
        )
        + 'print(jax.distributed.is_initialized(), end="")'
      )
      args = [sys.executable, "-c", pycmd]
      process = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
      )
      subprocesses.append(process)

    try:
      for process in subprocesses:
        out, _ = process.communicate()
        self.assertEqual(process.returncode, 0)
        self.assertEqual(out, str(run_initialize))
    finally:
      for process in subprocesses:
        process.kill()


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
