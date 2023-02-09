# Copyright 2022 The JAX Authors.
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

import os
import sys
import warnings

from absl.testing import absltest
from jax._src import test_util as jtu


class DeprecationTest(absltest.TestCase):

  def testDeprecation(self):
    # This manipulation of sys.path exists to make this test work in Google's
    # Hermetic Python environment: it ensures the module is resolvable.
    saved_path = sys.path[0]
    try:
      sys.path[0] = os.path.dirname(__file__)
      import deprecation_module as m
    finally:
      sys.path[0] = saved_path

    with warnings.catch_warnings():
      warnings.simplefilter("error")
      self.assertEqual(m.x, 42)

    with self.assertWarns(DeprecationWarning, msg="Please use x"):
      self.assertEqual(m.y, 101)

    with self.assertRaises(AttributeError, msg="Please do not use z"):
      _ = m.z

    with self.assertRaisesRegex(AttributeError,
                                "module .* has no attribute 'w'"):
      _ = m.w


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
