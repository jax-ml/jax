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

import warnings

from absl.testing import absltest

from jax._src import config
from jax._src import test_warning_util
from jax._src import test_util as jtu


config.parse_flags_with_absl()

class WarningsUtilTest(jtu.JaxTestCase):

  @test_warning_util.raise_on_warnings()
  def test_warning_raises(self):
    with self.assertRaises(UserWarning, msg="hello"):
      warnings.warn("hello", category=UserWarning)

    with self.assertRaises(DeprecationWarning, msg="hello"):
      warnings.warn("hello", category=DeprecationWarning)

  @test_warning_util.raise_on_warnings()
  def test_ignore_warning(self):
    with test_warning_util.ignore_warning(message="h.*o"):
      warnings.warn("hello", category=UserWarning)

    with self.assertRaises(UserWarning, msg="hello"):
      with test_warning_util.ignore_warning(message="h.*o"):
        warnings.warn("goodbye", category=UserWarning)

    with test_warning_util.ignore_warning(category=UserWarning):
      warnings.warn("hello", category=UserWarning)

    with self.assertRaises(UserWarning, msg="hello"):
      with test_warning_util.ignore_warning(category=DeprecationWarning):
        warnings.warn("goodbye", category=UserWarning)

  def test_record_warning(self):
    with test_warning_util.record_warnings() as w:
      warnings.warn("hello", category=UserWarning)
      warnings.warn("goodbye", category=DeprecationWarning)
    self.assertLen(w, 2)
    self.assertIs(w[0].category, UserWarning)
    self.assertIn("hello", str(w[0].message))
    self.assertIs(w[1].category, DeprecationWarning)
    self.assertIn("goodbye", str(w[1].message))

  def test_record_warning_nested(self):
    with test_warning_util.record_warnings() as w:
      warnings.warn("aa", category=UserWarning)
      with test_warning_util.record_warnings() as v:
        warnings.warn("bb", category=UserWarning)
      warnings.warn("cc", category=DeprecationWarning)
    self.assertLen(w, 2)
    self.assertIs(w[0].category, UserWarning)
    self.assertIn("aa", str(w[0].message))
    self.assertIs(w[1].category, DeprecationWarning)
    self.assertIn("cc", str(w[1].message))
    self.assertLen(v, 1)
    self.assertIs(v[0].category, UserWarning)
    self.assertIn("bb", str(v[0].message))


  def test_raises_warning(self):
    with self.assertRaises(UserWarning, msg="hello"):
      with test_warning_util.ignore_warning():
        with test_warning_util.raise_on_warnings():
          warnings.warn("hello", category=UserWarning)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
