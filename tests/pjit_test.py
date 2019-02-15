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

import numpy as onp
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as np
from jax import test_util as jtu
from jax.api import pjit, pmap, jvp, grad
from jax.lax import psum

from jax.config import config
config.parse_flags_with_absl()


class PmapTest(jtu.JaxTestCase):

  # @jtu.skip_on_devices("gpu")
  # def testBasic(self):
  #   f = lambda x: x - psum(x, 'i')
  #   x = onp.arange(8., dtype=onp.float32).reshape(4, 2)
  #   f = pjit(f, axis_name='i', in_axes=0, out_axes=0, mesh_axis=0)
  #   ans = f(x)
  #   expected = x - x.sum(0)
  #   self.assertAllClose(ans, expected, check_dtypes=False)

  # @jtu.skip_on_devices("gpu")
  # def testTupleOutput(self):
  #   f = lambda x: (x - psum(x, 'i'),)
  #   x = onp.arange(8., dtype=onp.float32).reshape(4, 2)
  #   f = pjit(f, axis_name='i', in_axes=0, out_axes=0, mesh_axis=0)
  #   ans = f(x)
  #   expected = (x - x.sum(0),)
  #   self.assertAllClose(ans, expected, check_dtypes=False)

  # @jtu.skip_on_devices("gpu")
  # def testTupleInput(self):
  #   f = lambda x: x[0] - psum(x[0], 'i')
  #   x = onp.arange(8., dtype=onp.float32).reshape(4, 2)
  #   f = pjit(f, axis_name='i', in_axes=0, out_axes=0, mesh_axis=0)
  #   ans = f((x,))
  #   expected = x - x.sum(0)
  #   self.assertAllClose(ans, expected, check_dtypes=False)

  # @jtu.skip_on_devices("gpu")
  # def testNested(self):
  #   def f(x, y):
  #     return psum(psum(x, 'i'), 'j')
  #   f = pjit(f, 'i')
  #   f = pjit(f, 'j', out_axes=1)

  #   x = onp.ones((3, 4), onp.float32)
  #   ans = f(x, x)
  #   expected = 12 * onp.ones((4, 3), onp.float32)
  #   self.assertAllClose(ans, expected, check_dtypes=True)

  # @jtu.skip_on_devices("gpu")
  # def testForwardModeAutodiff(self):
  #   def f(x):
  #     return np.cos(x - psum(np.sin(x), 'i'))

  #   x = np.ones(4)
  #   expected = jvp(pmap(f, 'i'), (x,), (x,))

  #   g = pjit(f, axis_name='i')
  #   ans = jvp(g, (x,), (x,))

  #   self.assertAllClose(ans, expected, check_dtypes=False)

  # @jtu.skip_on_devices("gpu")
  # def testReverseModeAutodiff(self):
  #   def f(x):
  #     return x - psum(x, 'i')

  #   x = np.ones(4)
  #   expected1 = grad(lambda x: np.sum(pmap(f, 'i')(x)))(x)
  #   expected2 = grad(lambda x: np.sum(x - np.sum(x)))(x)

  #   g = pjit(f, axis_name='i')
  #   ans = grad(lambda x: np.sum(g(x)))(x)

  #   self.assertAllClose(ans, expected1, check_dtypes=False)
  #   self.assertAllClose(ans, expected2, check_dtypes=False)

  @jtu.skip_on_devices("gpu")
  def testStuff(self):
    def f(x):
      return np.sin(np.sin(np.sin(x)))

    x = np.ones(4)

    g = pjit(f, axis_name='i')
    ans = grad(lambda x: np.sum(g(x)))(x)


if __name__ == '__main__':
  absltest.main()
