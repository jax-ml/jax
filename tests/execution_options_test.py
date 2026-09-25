# Copyright 2026 The JAX Authors.
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

"""Tests for the config state backing per-execution options."""

from concurrent import futures
import unittest

from absl.testing import absltest

import jax
from jax._src import execution_options
from jax._src.lib import jaxlib_extension_version


@unittest.skipIf(
    jaxlib_extension_version < 500,
    "Requires jaxlib_extension_version >= 500",
)
class ExecutionOptionsTest(absltest.TestCase):

  def test_nested_context_restores_state(self):
    state = execution_options.execution_options_context_manager
    original = state.get_local()
    self.assertIsNone(state.value)
    with jax.execution_options(custom_options={"count": 7, "mode": "fast"}):
      outer = state.value
      with jax.execution_options():
        self.assertIs(state.value, outer)
      with self.assertRaisesRegex(RuntimeError, "test error"):
        with jax.execution_options(custom_options={"count": 8}):
          self.assertEqual(
              state.value, {"count": 8, "mode": "fast"})
          raise RuntimeError("test error")
      self.assertIs(state.value, outer)
    self.assertIs(state.get_local(), original)

  def test_bytes_options_rejected(self):
    f = jax.jit(lambda x: x + 1)
    f(1)  # Populate the JIT cache before testing dispatch with invalid options.
    compiled = f.lower(1).compile()
    for execute in (f, compiled):
      for blob in (b"fast", b"\xff\xfe", b""):
        with self.subTest(execute=execute, blob=blob):
          with jax.execution_options(custom_options={"blob": blob}):
            with jax.execution_options(custom_options={"n": 1}):
              with self.assertRaisesRegex(
                  TypeError, "Unsupported custom option blob"):
                execute(1)

  def test_thread_isolation(self):
    state = execution_options.execution_options_context_manager

    def worker():
      self.assertIsNone(state.value)
      with jax.execution_options(custom_options={"thread": "worker"}):
        self.assertEqual(state.value, {"thread": "worker"})
      self.assertIsNone(state.value)

    with jax.execution_options(custom_options={"thread": "main"}):
      with futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(worker).result()
        pool.submit(worker).result()
      self.assertEqual(state.value, {"thread": "main"})
    self.assertIsNone(state.value)


if __name__ == "__main__":
  absltest.main()
