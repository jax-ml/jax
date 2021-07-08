# Copyright 2021 Google LLC
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

from jax import test_util as jtu
import jax.numpy as jnp

from jax.config import config
config.parse_flags_with_absl()


class TestUtilTest(jtu.JaxTestCase):

  @jtu.disable_implicit_rank_promotion
  def testDisableImplicitRankPromotion(self):
    x = jnp.zeros([2])
    y = jnp.zeros([2])

    with self.assertRaises(ValueError):
      x[None, :] + y


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
