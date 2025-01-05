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
import weakref

from absl.testing import absltest
import jax
from jax._src import config
import jax._src.test_util as jtu
import jax.numpy as jnp

jax.config.parse_flags_with_absl()


def _create_array_cycle():
  """Creates a reference cycle of two jax.Arrays."""
  n1 = jnp.ones((2, 2))
  n2 = jnp.zeros((2, 2))
  n1.next = n2
  n2.next = n1
  return weakref.ref(n1)


class GarbageCollectionGuardTest(jtu.JaxTestCase):

  def test_gced_array_is_not_logged_by_default(self):
    # Create a reference cycle of two jax.Arrays.
    ref = _create_array_cycle()
    with jtu.capture_stderr() as stderr:
      self.assertIsNotNone(ref())  # Cycle still alive.
      gc.collect()
      self.assertIsNone(ref())  # Cycle collected.
    # Check that no error message is logged because
    # `array_garbage_collection_guard` defaults to `allow`.
    self.assertNotIn(
        "`jax.Array` was deleted by the Python garbage collector", stderr(),
    )

  def test_gced_array_is_logged(self):
    with config.array_garbage_collection_guard("log"):
      with jtu.capture_stderr() as stderr:
        # Create a reference cycle of two jax.Arrays.
        ref = _create_array_cycle()
        self.assertIsNotNone(ref())  # Cycle still alive.
        gc.collect()
        self.assertIsNone(ref())  # Cycle collected.
    # Verify that an error message is logged because two jax.Arrays were garbage
    # collected.
    self.assertIn(
        "`jax.Array` was deleted by the Python garbage collector", stderr()
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
