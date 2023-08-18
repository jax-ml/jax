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

import unittest

from absl.testing import absltest

from jax._src.lib import check_jaxlib_version


class JaxVersionTest(unittest.TestCase):

  def testVersions(self):
    check_jaxlib_version(jax_version="1.2.3", jaxlib_version="1.2.3",
                         minimum_jaxlib_version="1.2.3")

    check_jaxlib_version(jax_version="1.2.3.4", jaxlib_version="1.2.3",
                         minimum_jaxlib_version="1.2.3")

    check_jaxlib_version(jax_version="2.5.dev234", jaxlib_version="1.2.3",
                         minimum_jaxlib_version="1.2.3")

    with self.assertRaisesRegex(RuntimeError, ".*jax requires version >=.*"):
      check_jaxlib_version(jax_version="1.2.3", jaxlib_version="1.0",
                           minimum_jaxlib_version="1.2.3")

    with self.assertRaisesRegex(RuntimeError, ".*jax requires version >=.*"):
      check_jaxlib_version(jax_version="1.2.3", jaxlib_version="1.0",
                           minimum_jaxlib_version="1.0.1")

    with self.assertRaisesRegex(RuntimeError,
                                ".incompatible with jax version.*"):
      check_jaxlib_version(jax_version="1.2.3", jaxlib_version="1.2.4",
                           minimum_jaxlib_version="1.0.5")

    with self.assertRaisesRegex(RuntimeError,
                                ".incompatible with jax version.*"):
      check_jaxlib_version(jax_version="0.4.14.dev20230818",
                           jaxlib_version="0.4.14.dev20230819",
                           minimum_jaxlib_version="0.4.14")


if __name__ == "__main__":
  absltest.main()
