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

import sys

from absl.testing import absltest
from jax._src import test_util as jtu
from jax._src.internal_test_util import lazy_loader_module as l


class LazyLoaderTest(absltest.TestCase):

  def testLazyLoader(self):
    self.assertEmpty([m for m in sys.modules if "lazy_test_submodule" in m])

    self.assertEqual(["lazy_test_submodule"], l.__all__)
    self.assertEqual(["lazy_test_submodule"], dir(l))

    # The submodule should be imported only after it is accessed.
    self.assertEmpty([m for m in sys.modules if "lazy_test_submodule" in m])
    self.assertEqual(42, l.lazy_test_submodule.a_function())
    self.assertLen([m for m in sys.modules if "lazy_test_submodule" in m], 1)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
