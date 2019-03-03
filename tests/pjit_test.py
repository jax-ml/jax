# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as np
from jax import test_util as jtu
from jax import lax
from jax.api import pjit, pmap, vmap, jvp, grad, make_jaxpr, linearize
from jax.lax import psum
from jax.lib import xla_bridge

from jax.config import config
config.parse_flags_with_absl()


class PjitTest(jtu.JaxTestCase):

  @jtu.skip_on_devices("gpu", "tpu")
  def testNestedWithClosure(self):
    assert xla_bridge.get_replica_count() == 1  # OSS CPU testing only
    x = onp.arange(3, dtype=onp.float32).reshape(1, 1, 3)

    @partial(pjit, axis_name='i')
    def test_fun(x):
      y = np.sum(np.sin(x))

      @partial(pjit, axis_name='j')
      def g(z):
        return 3. * np.exp(np.sin(x).sum() * np.cos(y) * np.tan(z))

      return grad(lambda w: np.sum(g(w)))(x)

    @vmap
    def baseline_fun(x):
      y = np.sum(np.sin(x))

      @vmap
      def g(z):
        return 3. * np.exp(np.sin(x).sum() * np.cos(y) * np.tan(z))

      return grad(lambda w: np.sum(g(w)))(x)

    ans = grad(lambda x: np.sum(test_fun(x)))(x)
    expected = grad(lambda x: np.sum(baseline_fun(x)))(x)
    self.assertAllClose(ans, expected, check_dtypes=True)


if __name__ == '__main__':
  absltest.main()
