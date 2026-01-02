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

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.lib import utils


# Note: This test modifies global logging library configuration knobs.
# We isolate it from other tests by running it as a separate test target.
@jtu.skip_under_pytest("Test must run in an isolated process")
class AbslCppLoggingTest(jtu.JaxTestCase):

  def test_vlogging(self):
    utils.absl_set_min_log_level(0)  # INFO
    with jtu.capture_stderr() as stderr:
      jax.jit(lambda x: x + 1)(1)
    self.assertNotIn("hlo_pass_pipeline.cc", stderr())
    with jtu.capture_stderr() as stderr:
      utils.absl_set_vlog_level("hlo_pass_pipeline", 1)
      jax.jit(lambda x: x + 2)(1)
    self.assertIn("hlo_pass_pipeline.cc", stderr())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
