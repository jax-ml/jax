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
from jax import lax
from jax.api import pmap, papply, jit, make_jaxpr
from jax.linear_util import wrap_init
from jax.interpreters.parallel import psum, scatter_like, chunk

from jax.config import config
config.parse_flags_with_absl()


# class PmapTest(jtu.JaxTestCase):

#   def testConstantFunction(self):
#     f = lambda x: 3
#     ans = pmap(f, axis_name='i')(onp.ones(4))
#     expected = 3 * onp.ones(4)
#     self.assertAllClose(ans, expected, check_dtypes=False)

#   def testReduceSum(self):
#     f = lambda x: psum(x, 'i')
#     ans = pmap(f, axis_name='i')(onp.ones(4))
#     expected = 4 * onp.ones(4)
#     self.assertAllClose(ans, expected, check_dtypes=False)

#   def testLogSoftmax(self):
#     f = lambda x: x - np.log(psum(np.exp(x), 'i'))
#     x = onp.log(onp.arange(1., 10., dtype=onp.float32))
#     ans = pmap(f, axis_name='i')(x)
#     expected = x - onp.log(onp.sum(onp.exp(x)))
#     self.assertAllClose(ans, expected, check_dtypes=False)

#   # def testPmapOfJit(self):
#   #   f = jit(lambda x: psum(x, 'i'))
#   #   x = onp.arange(12.)
#   #   ans = pmap(f, axis_name='i')(x)
#   #   expected = onp.sum(x)
#   #   self.assertAllClose(ans, expected, check_dtypes=False)


# class PapplyTest(jtu.JaxTestCase):

#   def testIdentity(self):
#     pfun, axis_name = papply(lambda x: x)
#     ans = pfun(onp.arange(3))
#     expected = onp.arange(3)
#     self.assertAllClose(ans, expected, check_dtypes=False)

#   def testMap(self):
#     pfun, axis_name = papply(np.sin)
#     ans = pfun(onp.arange(3.))
#     expected = onp.sin(onp.arange(3.))
#     self.assertAllClose(ans, expected, check_dtypes=False)

#   def testSum(self):
#     pfun, axis_name = papply(np.sum)

#     jaxpr = make_jaxpr(pfun)(onp.zeros(5))
#     expected_jaxpr = make_jaxpr(lambda x: psum(x, axis_name))(onp.zeros(5))
#     assert repr(jaxpr) == repr(expected_jaxpr)

#     ans = pmap(pfun, axis_name)(onp.arange(3.))
#     expected = onp.sum(onp.arange(3.))
#     self.assertAllClose(ans, expected, check_dtypes=False)

#   def testLogSoftmax(self):

#     def fun(x):
#       return x - np.log(np.sum(np.exp(x)))

#     pfun, axis_name = papply(fun)

#     jaxpr = make_jaxpr(pfun)(onp.zeros(5))
#     expected_jaxpr = make_jaxpr(lambda x: x - np.log(psum(np.exp(x), axis_name))
#                                 )(onp.zeros(5))
#     assert repr(jaxpr) == repr(expected_jaxpr)

#     ans = pmap(pfun, axis_name)(onp.arange(1., 5.))
#     expected = fun(onp.arange(1., 5.))
#     self.assertAllClose(ans, expected, check_dtypes=False)

#   def testAdd(self):
#     x = onp.array([[1, 2, 3], [4, 5, 6]])
#     expected = x + x

#     pfun, axis_name = papply(np.add)
#     ans = pmap(pfun, axis_name)(x, x)
#     self.assertAllClose(ans, expected, check_dtypes=True)

# #   def testAddDifferentSharding(self):
# #     x = onp.array([[1, 2, 3], [4, 5, 6]])
# #     expected = x + x

# #     pfun, axis_name = papply(np.add, (0, 1))
# #     ans = pmap(pfun, axis_name)(x, x)
# #     self.assertAllClose(ans, expected, check_dtypes=True)

# #   def testScatterLike(self):
# #     def fun(y):
# #       x_scattered = scatter_like(x, y)
# #       return lax.add(x_scattered, y)  # TODO replace with x + y

# #     x = onp.array([[1, 2], [3, 4]])
# #     expected = x + x

# #     pfun, axis_name = papply(fun)
# #     ans = pmap(pfun, axis_name)(x)
# #     self.assertAllClose(ans, expected, check_dtypes=True)

#   def testAddBroadcasting(self):

#     def fun(x):
#       return x + 3

#     x = onp.array([[1, 2], [3, 4]])
#     expected = x + 3

#     pfun, axis_name = papply(fun)
#     ans = pmap(pfun, axis_name)(x)
#     self.assertAllClose(ans, expected, check_dtypes=True)


class ChunkTest(jtu.JaxTestCase):

  def testChunkingSum(self):
    f = lambda x: psum(x, 'i')

    x = 3 * onp.ones((4, 2))
    ans = pmap(lambda x: chunk(wrap_init(f), 2, 'i', (x,), (0,), 0), 'i')(x)
    expected = 24

    self.assertAllClose(ans, expected, check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
