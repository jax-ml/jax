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

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import numpy.random as npr
import six
from unittest import SkipTest

from jax import api
from jax import test_util as jtu
from jax import numpy as np

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS
npr.seed(0)


class MultiBackendTest(jtu.JaxTestCase):
  """Tests jit targeting to different backends."""

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": "_backend={}".format(backend),
         "backend": backend,
        }
        for backend in ['cpu', 'gpu', 'tpu', None]
        ))
  def testMultiBackend(self, backend):
    if backend not in ('cpu', jtu.device_under_test(), None):
      raise SkipTest()
    @partial(api.jit, device=backend)
    def fun(x, y):
        return np.matmul(x, y)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    z_host = onp.matmul(x, y)
    z = fun(x, y)
    self.assertAllClose(z, z_host, check_dtypes=True, rtol=1e-2)
    correct_platform = backend if backend else jtu.device_under_test()
    self.assertEqual(z.device_buffer.platform(), correct_platform)

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": "_ordering={}".format(ordering),
         "ordering": ordering,}
        for ordering in [('cpu', None), ('gpu', None), ('tpu', None), (None, None)]))
  def testMultiBackendNestedJit(self, ordering):
    outer, inner = ordering
    if outer not in ('cpu', jtu.device_under_test(), None):
      raise SkipTest()
    @partial(api.jit, device=outer)
    def fun(x, y):
        @partial(api.jit, device=inner)
        def infun(x, y):
            return np.matmul(x, y)
        return infun(x, y) + np.ones_like(x)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    z_host = onp.matmul(x, y) + onp.ones_like(x)
    z = fun(x, y)
    self.assertAllClose(z, z_host, check_dtypes=True, rtol=1e-2)
    correct_platform = outer if outer else jtu.device_under_test()
    self.assertEqual(z.device_buffer.platform(), correct_platform)

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": "_ordering={}".format(ordering),
         "ordering": ordering,}
        for ordering in [
            ('cpu', 'gpu'), ('gpu', 'cpu'),
            ('cpu', 'tpu'), ('tpu', 'cpu'),
            (None, 'cpu'), (None, 'gpu'), (None, 'tpu'),
        ]))
  def testMultiBackendNestedJitConflict(self, ordering):
    outer, inner = ordering
    if outer not in ('cpu', jtu.device_under_test(), None):
      raise SkipTest()
    if inner not in ('cpu', jtu.device_under_test(), None):
      raise SkipTest()
    @partial(api.jit, device=outer)
    def fun(x, y):
        @partial(api.jit, device=inner)
        def infun(x, y):
            return np.matmul(x, y)
        return infun(x, y) + np.ones_like(x)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    self.assertRaises(ValueError, lambda: fun(x, y))

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": "_backend={}".format(backend),
         "backend": backend,}
        for backend in ['cpu', 'gpu', 'tpu']
        ))
  def testGpuMultiBackendOpByOpReturn(self, backend):
    if backend not in ('cpu', jtu.device_under_test()):
      raise SkipTest()
    @partial(api.jit, device=backend)
    def fun(x, y):
        return np.matmul(x, y)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    z = fun(x, y)
    w = np.sin(z)
    self.assertEqual(z.device_buffer.platform(), backend)
    self.assertEqual(w.device_buffer.platform(), backend)

  @jtu.skip_on_devices("cpu")  # test can only fail with non-cpu backends
  def testJitCpu(self):
    @partial(api.jit, device='cpu')
    def get_arr(scale):
      return scale + np.ones((2, 2))

    x = get_arr(0.1)

    a = x / x.shape[0]
    b = x + np.ones_like(x)
    c = x + np.eye(2)

    self.assertEqual(a.device_buffer.device(), api.devices('cpu')[0])
    self.assertEqual(b.device_buffer.device(), api.devices('cpu')[0])
    self.assertEqual(c.device_buffer.device(), api.devices('cpu')[0])


if __name__ == "__main__":
  absltest.main()
