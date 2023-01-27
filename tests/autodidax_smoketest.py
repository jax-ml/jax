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

import importlib
import os

from absl.testing import absltest


class AutodidaxTest(absltest.TestCase):
  """Test that importing autodidax does not raise any errors."""

  def test_autodidax(self):
    try:
      # This works in bazel test environments.
      from jax.oss.docs import autodidax  # pylint: disable=unused-import
    except ImportError:
      # This works outside bazel, when run from the JAX source tree.
      autodidax_file = os.path.join(
          os.path.dirname(__file__), '..', 'docs', 'autodidax.py'
      )
      if not os.path.exists(autodidax_file):
        self.skipTest('Cannot locate autodidax.py')
      spec = importlib.util.spec_from_file_location('autodidax', autodidax_file)
      autodidax_module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(autodidax_module)


if __name__ == '__main__':
  absltest.main()
