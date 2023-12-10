# Copyright 2023 The JAX Authors.
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
import io
import logging


import jax
from jax import config
import jax._src.test_util as jtu

# Note: importing absltest causes an extra absl root log handler to be
# registered, which causes extra debug log messages. We don't expect users to
# import absl logging, so it should only affect this test. We need to use
# absltest.main and config.parse_flags_with_absl() in order for jax_test flag
# parsing to work correctly with bazel (otherwise we could avoid importing
# absltest/absl logging altogether).
from absl.testing import absltest
config.parse_flags_with_absl()


@contextlib.contextmanager
def capture_jax_logs():
  log_output = io.StringIO()
  handler = logging.StreamHandler(log_output)
  logger = logging.getLogger("jax")

  logger.addHandler(handler)
  try:
    yield log_output
  finally:
    logger.removeHandler(handler)


class LoggingTest(jtu.JaxTestCase):

  def test_debug_logging(self):
    # Warmup so we don't get "No GPU/TPU" warning later.
    jax.jit(lambda x: x + 1)(1)

    # Nothing logged by default (except warning messages, which we don't expect
    # here).
    with capture_jax_logs() as log_output:
      jax.jit(lambda x: x + 1)(1)
    self.assertEmpty(log_output.getvalue())

    # Turn on all debug logging.
    config.update("jax_debug_log_modules", "jax")
    with capture_jax_logs() as log_output:
      jax.jit(lambda x: x + 1)(1)
    self.assertIn("Finished tracing + transforming", log_output.getvalue())
    self.assertIn("Compiling <lambda>", log_output.getvalue())

    # Turn off all debug logging.
    config.update("jax_debug_log_modules", None)
    with capture_jax_logs() as log_output:
      jax.jit(lambda x: x + 1)(1)
    self.assertEmpty(log_output.getvalue())

    # Turn on one module.
    config.update("jax_debug_log_modules", "jax._src.dispatch")
    with capture_jax_logs() as log_output:
      jax.jit(lambda x: x + 1)(1)
    self.assertIn("Finished tracing + transforming", log_output.getvalue())
    self.assertNotIn("Compiling <lambda>", log_output.getvalue())

    # Turn everything off again.
    config.update("jax_debug_log_modules", None)
    with capture_jax_logs() as log_output:
      jax.jit(lambda x: x + 1)(1)
    self.assertEmpty(log_output.getvalue())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
