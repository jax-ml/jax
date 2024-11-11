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
"""Tests for garbage allocation guard."""

import gc
import io
from unittest import mock

from absl.testing import absltest
import jax
from jax._src import config
import jax._src.test_util as jtu
import jax.numpy as jnp

jax.config.parse_flags_with_absl()


# Helper class used to create a reference cycle.
class GarbageCollectionGuardTestNodeHelper:

  def __init__(self, data):
    self.data = data
    self.next = None


def _create_array_cycle():
  """Creates a reference cycle of two jax.Arrays."""
  n1 = GarbageCollectionGuardTestNodeHelper(jax.jit(lambda: jnp.ones( (2, 2)))())
  n2 = GarbageCollectionGuardTestNodeHelper(jax.jit(lambda: jnp.zeros((2, 2)))())
  n1.next = n2
  n2.next = n1


class GarbageCollectionGuardTest(jtu.JaxTestCase):

  def test_gced_array_is_not_logged_by_default(self):
    # Create a reference cycle of two jax.Arrays.
    _create_array_cycle()

    # Use mock_stderr to be able to inspect stderr.
    mock_stderr = io.StringIO()
    with mock.patch("sys.stderr", mock_stderr):
      # Trigger a garbage collection, which will garbage collect the arrays
      # in the cycle.
      gc.collect()
    # Check that no error message is logged because
    # `array_garbage_collection_guard` defaults to `allow`.
    self.assertNotIn(
        "`jax.Array` was deleted by the Python garbage collector",
        mock_stderr.getvalue(),
    )

  def test_gced_array_is_logged(self):
    # Use mock_stderr to be able to inspect stderr.
    mock_stderr = io.StringIO()

    with config.array_garbage_collection_guard("log"):
      # Create a reference cycle of two jax.Arrays.
      _create_array_cycle()
      with mock.patch("sys.stderr", mock_stderr):
        gc.collect()

    # Verify that an error message is logged because two jax.Arrays were garbage
    # collected.
    self.assertIn(
        "`jax.Array` was deleted by the Python garbage collector",
        mock_stderr.getvalue(),
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
