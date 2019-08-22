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
        for backend in ['cpu', 'gpu']
        ))
  @jtu.skip_on_devices('cpu', 'tpu')
  def testGpuMultiBackend(self, backend):
    @partial(api.jit, backend=backend)
    def fun(x, y):
        return np.matmul(x, y)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    z_host = onp.matmul(x, y)
    z = fun(x, y)
    self.assertAllClose(z, z_host, check_dtypes=True)
    self.assertEqual(z.device_buffer.platform(), backend)

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": "_backend={}".format(backend),
         "backend": backend,
        }
        for backend in ['cpu', 'tpu']
        ))
  @jtu.skip_on_devices('cpu', 'gpu')
  def testTpuMultiBackend(self, backend):
    @partial(api.jit, backend=backend)
    def fun(x, y):
        return np.matmul(x, y)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    z_host = onp.matmul(x, y)
    z = fun(x, y)
    self.assertAllClose(z, z_host, check_dtypes=True)
    self.assertEqual(z.device_buffer.platform(), backend)

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": "_ordering={}".format(ordering),
         "ordering": ordering,}
        for ordering in [('cpu', None), ('tpu', None)]))
  @jtu.skip_on_devices('cpu', 'gpu')
  def testTpuMultiBackendNestedJit(self, ordering):
    outer, inner = ordering
    @partial(api.jit, backend=outer)
    def fun(x, y):
        @partial(api.jit, backend=inner)
        def infun(x, y):
            return np.matmul(x, y)
        return infun(x, y) + np.ones_like(x)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    z_host = onp.matmul(x, y) + onp.ones_like(x)
    z = fun(x, y)
    self.assertAllClose(z, z_host, check_dtypes=True)
    self.assertEqual(z.device_buffer.platform(), outer)

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": "_ordering={}".format(ordering),
         "ordering": ordering,}
        for ordering in [('cpu', None), ('gpu', None)]))
  @jtu.skip_on_devices('cpu', 'tpu')
  def testGpuMultiBackendNestedJit(self, ordering):
    outer, inner = ordering
    @partial(api.jit, backend=outer)
    def fun(x, y):
        @partial(api.jit, backend=inner)
        def infun(x, y):
            return np.matmul(x, y)
        return infun(x, y) + np.ones_like(x)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    z_host = onp.matmul(x, y) + onp.ones_like(x)
    z = fun(x, y)
    self.assertAllClose(z, z_host, check_dtypes=True)
    self.assertEqual(z.device_buffer.platform(), outer)

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": "_ordering={}".format(ordering),
         "ordering": ordering,}
        for ordering in [
            ('cpu', 'tpu'), ('tpu', 'cpu'), (None, 'cpu'), (None, 'tpu'),
        ]))
  @jtu.skip_on_devices('cpu', 'gpu')
  def testTpuMultiBackendNestedJitConflict(self, ordering):
    outer, inner = ordering
    @partial(api.jit, backend=outer)
    def fun(x, y):
        @partial(api.jit, backend=inner)
        def infun(x, y):
            return np.matmul(x, y)
        return infun(x, y) + np.ones_like(x)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    self.assertRaises(ValueError, lambda: fun(x, y))

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": "_ordering={}".format(ordering),
         "ordering": ordering,}
        for ordering in [
            ('cpu', 'gpu'), ('gpu', 'cpu'), (None, 'cpu'), (None, 'gpu'),
        ]))
  @jtu.skip_on_devices('cpu', 'tpu')
  def testGpuMultiBackendNestedJitConflict(self, ordering):
    outer, inner = ordering
    @partial(api.jit, backend=outer)
    def fun(x, y):
        @partial(api.jit, backend=inner)
        def infun(x, y):
            return np.matmul(x, y)
        return infun(x, y) + np.ones_like(x)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    self.assertRaises(ValueError, lambda: fun(x, y))

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": "_backend={}".format(backend),
         "backend": backend,}
        for backend in ['cpu', 'gpu']
        ))
  @jtu.skip_on_devices('cpu', 'tpu')
  def testGpuMultiBackendOpByOpReturn(self, backend):
    @partial(api.jit, backend=backend)
    def fun(x, y):
        return np.matmul(x, y)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    z = fun(x, y)
    w = np.sin(z)
    self.assertEqual(z.device_buffer.platform(), backend)
    self.assertEqual(w.device_buffer.platform(), 'gpu')

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": "_backend={}".format(backend),
         "backend": backend,}
        for backend in ['cpu', 'tpu']
        ))
  @jtu.skip_on_devices('cpu', 'gpu')
  def testTpuMultiBackendOpByOpReturn(self, backend):
    @partial(api.jit, backend=backend)
    def fun(x, y):
        return np.matmul(x, y)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    z = fun(x, y)
    w = np.sin(z)
    self.assertEqual(z.device_buffer.platform(), backend)
    self.assertEqual(w.device_buffer.platform(), 'tpu')


if __name__ == "__main__":
  absltest.main()
