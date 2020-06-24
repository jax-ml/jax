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


from functools import partial

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import numpy.random as npr
from unittest import SkipTest

from jax import api
from jax import test_util as jtu
from jax import numpy as jnp

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
      raise SkipTest("Backend is not CPU or the device under test")
    @partial(api.jit, backend=backend)
    def fun(x, y):
        return jnp.matmul(x, y)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    z_host = np.matmul(x, y)
    z = fun(x, y)
    self.assertAllClose(z, z_host, rtol=1e-2)
    correct_platform = backend if backend else jtu.device_under_test()
    self.assertEqual(z.device_buffer.platform(), correct_platform)

  @parameterized.named_parameters(jtu.cases_from_list(
        {"testcase_name": "_ordering={}".format(ordering),
         "ordering": ordering,}
        for ordering in [('cpu', None), ('gpu', None), ('tpu', None), (None, None)]))
  def testMultiBackendNestedJit(self, ordering):
    outer, inner = ordering
    if outer not in ('cpu', jtu.device_under_test(), None):
      raise SkipTest("Backend is not CPU or the device under test")
    @partial(api.jit, backend=outer)
    def fun(x, y):
        @partial(api.jit, backend=inner)
        def infun(x, y):
            return jnp.matmul(x, y)
        return infun(x, y) + jnp.ones_like(x)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    z_host = np.matmul(x, y) + np.ones_like(x)
    z = fun(x, y)
    self.assertAllClose(z, z_host, rtol=1e-2)
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
      raise SkipTest("Backend is not CPU or the device under test")
    if inner not in ('cpu', jtu.device_under_test(), None):
      raise SkipTest("Backend is not CPU or the device under test")
    @partial(api.jit, backend=outer)
    def fun(x, y):
        @partial(api.jit, backend=inner)
        def infun(x, y):
            return jnp.matmul(x, y)
        return infun(x, y) + jnp.ones_like(x)
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
      raise SkipTest("Backend is not CPU or the device under test")
    @partial(api.jit, backend=backend)
    def fun(x, y):
        return jnp.matmul(x, y)
    x = npr.uniform(size=(10,10))
    y = npr.uniform(size=(10,10))
    z = fun(x, y)
    w = jnp.sin(z)
    self.assertEqual(z.device_buffer.platform(), backend)
    self.assertEqual(w.device_buffer.platform(), backend)

  @jtu.skip_on_devices("cpu")  # test can only fail with non-cpu backends
  def testJitCpu(self):
    @partial(api.jit, backend='cpu')
    def get_arr(scale):
      return scale + jnp.ones((2, 2))

    x = get_arr(0.1)

    a = x / x.shape[0]
    b = x + jnp.ones_like(x)
    c = x + jnp.eye(2)

    self.assertEqual(a.device_buffer.device(), api.devices('cpu')[0])
    self.assertEqual(b.device_buffer.device(), api.devices('cpu')[0])
    self.assertEqual(c.device_buffer.device(), api.devices('cpu')[0])

  @jtu.skip_on_devices("cpu")  # test can only fail with non-cpu backends
  def test_closed_over_values_device_placement(self):
    # see https://github.com/google/jax/issues/1431
    def f(): return jnp.add(3., 4.)
    self.assertNotEqual(api.jit(f)().device_buffer.device(),
                        api.devices('cpu')[0])
    self.assertEqual(api.jit(f, backend='cpu')().device_buffer.device(),
                     api.devices('cpu')[0])

  @jtu.skip_on_devices("cpu")  # test only makes sense on non-cpu backends
  def test_jit_on_nondefault_backend(self):
    cpus = api.devices("cpu")
    self.assertNotEmpty(cpus)

    # Since we are not on CPU, some other backend will be the default
    default_dev = api.devices()[0]
    self.assertNotEqual(default_dev.platform, "cpu")

    data_on_cpu = api.device_put(1, device=cpus[0])
    self.assertEqual(data_on_cpu.device_buffer.device(), cpus[0])

    def my_sin(x): return jnp.sin(x)
    # jit without any device spec follows the data
    result1 = api.jit(my_sin)(2)
    self.assertEqual(result1.device_buffer.device(), default_dev)
    result2 = api.jit(my_sin)(data_on_cpu)
    self.assertEqual(result2.device_buffer.device(), cpus[0])

    # jit with `device` spec places the data on the specified device
    result3 = api.jit(my_sin, device=cpus[0])(2)
    self.assertEqual(result3.device_buffer.device(), cpus[0])

    # jit with `backend` spec places the data on the specified backend
    result4 = api.jit(my_sin, backend="cpu")(2)
    self.assertEqual(result4.device_buffer.device(), cpus[0])

  @jtu.skip_on_devices("cpu")  # test only makes sense on non-cpu backends
  def test_indexing(self):
    # https://github.com/google/jax/issues/2905
    cpus = api.devices("cpu")

    x = api.device_put(np.ones(2), cpus[0])
    y = x[0]
    self.assertEqual(y.device_buffer.device(), cpus[0])

  @jtu.skip_on_devices("cpu")  # test only makes sense on non-cpu backends
  def test_sum(self):
    # https://github.com/google/jax/issues/2905
    cpus = api.devices("cpu")

    x = api.device_put(np.ones(2), cpus[0])
    y = x.sum()
    self.assertEqual(y.device_buffer.device(), cpus[0])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
