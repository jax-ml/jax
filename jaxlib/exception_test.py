# Copyright 2026 The JAX Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from jax.jaxlib import _jax as jaxlib
from jax.jaxlib import test_exceptions

from absl.testing import absltest


class ExceptionTest(absltest.TestCase):

  def test_status_exception(self):
    with self.assertRaises(jaxlib.JaxRuntimeError) as mgr:
      test_exceptions.returns_status()

      # absl::StatusCode::kInvalidArgument
      self.assertEqual(mgr.exception.error_code(), 3)
      self.assertEqual(mgr.exception.error_message(), "we are testing a status")

  def test_non_status_exception(self):
    with self.assertRaises(jaxlib.JaxRuntimeError) as mgr:
      test_exceptions.throws_string()

      # absl::StatusCode::kUnknown
      self.assertEqual(mgr.exception.error_code(), 2)
      self.assertEqual(mgr.exception.error_message(), "we are testing a string")


if __name__ == "__main__":
  absltest.main()
