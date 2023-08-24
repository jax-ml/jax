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

from absl.testing import absltest

import jax.extend as jex

from jax._src import prng
from jax._src import test_util as jtu

from jax import config
config.parse_flags_with_absl()

class ExtendTest(jtu.JaxTestCase):
  def test_symbols(self):
    # Assume these are tested in random_test.py, only check equivalence
    self.assertIs(jex.random.PRNGImpl, prng.PRNGImpl)
    self.assertIs(jex.random.seed_with_impl, prng.seed_with_impl)
    self.assertIs(jex.random.threefry2x32_p, prng.threefry2x32_p)
    self.assertIs(jex.random.threefry_2x32, prng.threefry_2x32)
    self.assertIs(jex.random.threefry_prng_impl, prng.threefry_prng_impl)
    self.assertIs(jex.random.rbg_prng_impl, prng.rbg_prng_impl)
    self.assertIs(jex.random.unsafe_rbg_prng_impl, prng.unsafe_rbg_prng_impl)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
