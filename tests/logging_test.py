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
import os
import platform
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest

import jax
import jax._src.test_util as jtu
from jax._src import xla_bridge
from jax._src.logging_config import _default_TF_CPP_MIN_LOG_LEVEL

# Note: importing absltest causes an extra absl root log handler to be
# registered, which causes extra debug log messages. We don't expect users to
# import absl logging, so it should only affect this test. We need to use
# absltest.main and config.parse_flags_with_absl() in order for jax_test flag
# parsing to work correctly with bazel (otherwise we could avoid importing
# absltest/absl logging altogether).
from absl.testing import absltest
jax.config.parse_flags_with_absl()


@contextlib.contextmanager
def jax_debug_log_modules(value):
  # jax_debug_log_modules doesn't have a context manager, because it's
  # not thread-safe. But since tests are always single-threaded, we
  # can define one here.
  original_value = jax.config.jax_debug_log_modules
  jax.config.update("jax_debug_log_modules", value)
  try:
    yield
  finally:
    jax.config.update("jax_debug_log_modules", original_value)

@contextlib.contextmanager
def jax_logging_level(value):
  # jax_logging_level doesn't have a context manager, because it's
  # not thread-safe. But since tests are always single-threaded, we
  # can define one here.
  original_value = jax.config.jax_logging_level
  jax.config.update("jax_logging_level", value)
  try:
    yield
  finally:
    jax.config.update("jax_logging_level", original_value)


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


# Saves and runs script from the file in order to fix the problem with
# `tsl::Env::Default()->GetExecutablePath()` not working properly with
# command flag.
def _run(program, env_var = {}):
  # strip the leading whitespace from the program script
  program = re.sub(r"^\s+", "", program, flags=re.MULTILINE)

  with tempfile.NamedTemporaryFile(
          mode="w+", encoding="utf-8", suffix=".py", dir=os.getcwd()
  ) as f:
    f.write(textwrap.dedent(program))
    f.flush()
    python = sys.executable
    assert "python" in python
    if env_var:
      env_var.update(os.environ)
    else:
      env_var = os.environ

    # Make sure C++ logging is at default level for the test process.
    p = subprocess.run([python, f.name], env=env_var, capture_output=True, text=True)

    return type("", (object,), { "stdout": p.stdout, "stderr": p.stderr })


class LoggingTest(jtu.JaxTestCase):

  @unittest.skipIf(platform.system() == "Windows",
                   "Subprocess test doesn't work on Windows")
  def test_no_log_spam(self):
    if jtu.is_cloud_tpu() and xla_bridge._backends:
      raise self.skipTest(
          "test requires fresh process on Cloud TPU because only one process "
          "can use the TPU at a time")
    if sys.executable is None:
      raise self.skipTest("test requires access to python binary")

    o = _run("""
        import jax
        jax.device_count()
        f = jax.jit(lambda x: x + 1)
        f(1)
        f(2)
        jax.numpy.add(1, 1)
    """)

    lines = o.stdout.split("\n")
    lines.extend(o.stderr.split("\n"))
    allowlist = [
        (
            "An NVIDIA GPU may be present on this machine, but a"
            " CUDA-enabled jaxlib is not installed. Falling back to cpu."
        ),
    ]
    lines = [l for l in lines if l in allowlist]
    self.assertEmpty(lines)

  def test_debug_logging(self):
    # Warmup so we don't get "No GPU/TPU" warning later.
    jax.jit(lambda x: x + 1)(1)

    # Nothing logged by default (except warning messages, which we don't expect
    # here).
    with capture_jax_logs() as log_output:
      jax.jit(lambda x: x + 1)(1)
    self.assertEmpty(log_output.getvalue())

    # Turn on all debug logging.
    with jax_debug_log_modules("jax"):
      with capture_jax_logs() as log_output:
        jax.jit(lambda x: x + 1)(1)
      self.assertIn("Finished tracing + transforming", log_output.getvalue())
      self.assertIn("Compiling <lambda>", log_output.getvalue())

    # Turn off all debug logging.
    with jax_debug_log_modules(""):
      with capture_jax_logs() as log_output:
        jax.jit(lambda x: x + 1)(1)
      self.assertEmpty(log_output.getvalue())

    # Turn on one module.
    with jax_debug_log_modules("jax._src.dispatch"):
      with capture_jax_logs() as log_output:
        jax.jit(lambda x: x + 1)(1)
      self.assertIn("Finished tracing + transforming", log_output.getvalue())
      self.assertNotIn("Compiling <lambda>", log_output.getvalue())

    # Turn everything off again.
    with jax_debug_log_modules(""):
      with capture_jax_logs() as log_output:
        jax.jit(lambda x: x + 1)(1)
      self.assertEmpty(log_output.getvalue())

  @jtu.skip_on_devices("tpu")
  @unittest.skipIf(platform.system() == "Windows",
                   "Subprocess test doesn't work on Windows")
  def test_subprocess_stderr_info_logging(self):
    if sys.executable is None:
      raise self.skipTest("test requires access to python binary")

    o = _run("""
        import jax  # this prints INFO logging from backend imports
        jax.jit(lambda x: x)(1)  # this prints logs to DEBUG (from compilation)
    """, { "JAX_LOGGING_LEVEL": "INFO" })

    log_output = o.stderr
    info_lines = log_output.split("\n")
    self.assertGreater(len(info_lines), 0)
    self.assertIn("INFO", log_output)
    self.assertNotIn("DEBUG", log_output)

  @jtu.skip_on_devices("tpu")
  @unittest.skipIf(platform.system() == "Windows",
                   "Subprocess test doesn't work on Windows")
  def test_subprocess_stderr_debug_logging(self):
    if sys.executable is None:
      raise self.skipTest("test requires access to python binary")

    program = """
    import jax  # this prints INFO logging from backend imports
    jax.jit(lambda x: x)(1)  # this prints logs to DEBUG (from compilation)
    """

    o = _run(program, { "JAX_LOGGING_LEVEL": "DEBUG" })

    log_output = o.stderr
    self.assertIn("INFO", log_output)
    self.assertIn("DEBUG", log_output)

    o = _run(program, { "JAX_DEBUG_LOG_MODULES": "jax" })
    log_output = o.stderr
    self.assertIn("DEBUG", log_output)

  @jtu.skip_on_devices("tpu")
  @unittest.skipIf(platform.system() == "Windows",
                   "Subprocess test doesn't work on Windows")
  def test_subprocess_toggling_logging_level(self):
    if sys.executable is None:
      raise self.skipTest("test requires access to python binary")

    _separator = "---------------------------"
    o = _run(f"""
    import sys
    import jax  # this prints INFO logging from backend imports
    jax.jit(lambda x: x)(1)  # this prints logs to DEBUG (from compilation)
    jax.config.update("jax_logging_level", None)
    sys.stderr.write("{_separator}")
    jax.jit(lambda x: x)(1)  # should not log anything now
    """, {"JAX_LOGGING_LEVEL": "DEBUG"})
    log_output = o.stderr
    m = re.search(_separator, log_output)
    self.assertTrue(m is not None)
    log_output_verbose = log_output[:m.start()]
    log_output_silent = log_output[m.end():]

    self.assertIn("Finished tracing + transforming <lambda> for pjit",
                  log_output_verbose)
    self.assertEqual(log_output_silent, "")

  @jtu.skip_on_devices("tpu")
  @unittest.skipIf(platform.system() == "Windows",
                   "Subprocess test doesn't work on Windows")
  def test_subprocess_double_logging_absent(self):
    if sys.executable is None:
      raise self.skipTest("test requires access to python binary")

    o = _run("""
    import jax  # this prints INFO logging from backend imports
    jax.config.update("jax_debug_log_modules", "jax._src.compiler,jax._src.dispatch")
    jax.jit(lambda x: x)(1)  # this prints logs to DEBUG (from compilation)
    """, { "JAX_LOGGING_LEVEL": "DEBUG" })

    log_output = o.stderr
    self.assertNotEmpty(log_output)
    log_lines = log_output.strip().split("\n")
    # only one tracing line should be printed, if there's more than one
    # then logs are printing duplicated
    self.assertLen([line for line in log_lines
                    if "Finished tracing + transforming" in line], 1)

  @jtu.skip_on_devices("tpu")
  @unittest.skipIf(platform.system() == "Windows",
                   "Subprocess test doesn't work on Windows")
  def test_subprocess_cpp_logging_level(self):
    if sys.executable is None:
      raise self.skipTest("test requires access to python binary")

    program = """
    import sys
    import jax  # this prints INFO logging from backend imports
    jax.distributed.initialize("127.0.0.1:12345", num_processes=1, process_id=0)
    """

    o = _run(program, { "JAX_LOGGING_LEVEL": "DEBUG" })
    self.assertIn("Initializing CoordinationService", o.stderr)

    o = _run(program, { "JAX_LOGGING_LEVEL": "INFO" })
    self.assertIn("Initializing CoordinationService", o.stderr)

    # verbose logging: WARNING, None
    o = _run(program, { "JAX_LOGGING_LEVEL": "WARNING" })
    self.assertNotIn("Initializing CoordinationService", o.stderr)

    o = _run(program)
    if int(_default_TF_CPP_MIN_LOG_LEVEL) >= 1:
      self.assertNotIn("Initializing CoordinationService", o.stderr)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
