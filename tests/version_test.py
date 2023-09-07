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

import datetime
import unittest

from absl.testing import absltest

import jax
from jax._src.lib import check_jaxlib_version
from jax._src import test_util as jtu


class JaxVersionTest(unittest.TestCase):

  def testBuildVersion(self):
    base_version = jax.version._version

    if jax.version._release_version is not None:
      version = jax.version._get_version_for_build()
      self.assertEqual(version, jax.version._release_version)
    else:
      with jtu.set_env(JAX_RELEASE=None, JAXLIB_RELEASE=None,
                      JAX_NIGHTLY=None, JAXLIB_NIGHTLY=None):
        version = jax.version._get_version_for_build()
        # TODO(jakevdp): confirm that this includes a date string & commit hash?
        self.assertTrue(version.startswith(f"{base_version}.dev"))

      with jtu.set_env(JAX_RELEASE=None, JAXLIB_RELEASE=None,
                      JAX_NIGHTLY="1", JAXLIB_NIGHTLY=None):
        version = jax.version._get_version_for_build()
        datestring = datetime.date.today().strftime("%Y%m%d")
        self.assertEqual(version, f"{base_version}.dev{datestring}")

      with jtu.set_env(JAX_RELEASE=None, JAXLIB_RELEASE=None,
                      JAX_NIGHTLY=None, JAXLIB_NIGHTLY="1"):
        version = jax.version._get_version_for_build()
        datestring = datetime.date.today().strftime("%Y%m%d")
        self.assertEqual(version, f"{base_version}.dev{datestring}")

      with jtu.set_env(JAX_RELEASE="1", JAXLIB_RELEASE=None,
                      JAX_NIGHTLY=None, JAXLIB_NIGHTLY=None):
        version = jax.version._get_version_for_build()
        self.assertEqual(version, base_version)

      with jtu.set_env(JAX_RELEASE=None, JAXLIB_RELEASE="1",
                      JAX_NIGHTLY=None, JAXLIB_NIGHTLY=None):
        version = jax.version._get_version_for_build()
        self.assertEqual(version, base_version)

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
