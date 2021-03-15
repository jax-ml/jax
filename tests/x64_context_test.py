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


import concurrent.futures
import time

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import api
from jax import lax
from jax import partial
from jax import random
from jax.config import config
from jax.config import FLAGS
from jax.experimental import enable_x64, disable_x64
import jax.numpy as jnp
import jax.test_util as jtu

config.parse_flags_with_absl()


def _maybe_jit(jit_type, func, *args, **kwargs):
  if jit_type == "python":
    return api._python_jit(func, *args, **kwargs)
  elif jit_type == "cpp":
    return api._cpp_jit(func, *args, **kwargs)
  elif jit_type is None:
    return func
  else:
    raise ValueError(f"Unrecognized jit_type={jit_type!r}")


class X64ContextTests(jtu.JaxTestCase):
  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_jit={}".format(jit), "jit": jit}
      for jit in ["python", "cpp", None]))
  def test_make_array(self, jit):
    if jit == "cpp" and not config.omnistaging_enabled:
      self.skipTest("cpp_jit requires omnistaging")
    func = _maybe_jit(jit, lambda: jnp.arange(10.0))
    dtype_start = func().dtype
    with enable_x64():
      self.assertEqual(func().dtype, "float64")
    with disable_x64():
      self.assertEqual(func().dtype, "float32")
    self.assertEqual(func().dtype, dtype_start)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name": "_jit={}_f_{}".format(jit, f.__name__),
          "jit": jit,
          "enable_or_disable": f
      } for jit in ["python", "cpp", None] for f in [enable_x64, disable_x64]))
  def test_correctly_capture_default(self, jit, enable_or_disable):
    if jit == "cpp" and not config.omnistaging_enabled:
      self.skipTest("cpp_jit requires omnistaging")

    # The fact we defined a jitted function with a block with a different value
    # of `config.enable_x64` has no impact on the output.
    with enable_or_disable():
      func = _maybe_jit(jit, lambda: jnp.arange(10.0))
      func()

    expected_dtype = "float64" if config.read("jax_enable_x64") else "float32"
    self.assertEqual(func().dtype, expected_dtype)

    with enable_x64():
      self.assertEqual(func().dtype, "float64")
    with disable_x64():
      self.assertEqual(func().dtype, "float32")

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_jit={}".format(jit), "jit": jit}
      for jit in ["python", "cpp", None]))
  def test_near_singular_inverse(self, jit):
    if jtu.device_under_test() == "tpu":
      self.skipTest("64-bit inverse not available on TPU")
    if jit == "cpp" and not config.omnistaging_enabled:
      self.skipTest("cpp_jit requires omnistaging")
    @partial(_maybe_jit, jit, static_argnums=1)
    def near_singular_inverse(key, N, eps):
      X = random.uniform(key, (N, N))
      X = X.at[-1].mul(eps)
      return jnp.linalg.inv(X)

    key = random.PRNGKey(1701)
    eps = 1E-40
    N = 5

    with enable_x64():
      result_64 = near_singular_inverse(key, N, eps)
      self.assertTrue(jnp.all(jnp.isfinite(result_64)))

    with disable_x64():
      result_32 = near_singular_inverse(key, N, eps)
      self.assertTrue(jnp.all(~jnp.isfinite(result_32)))

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_jit={}".format(jit), "jit": jit}
      for jit in ["python", "cpp", None]))
  def test_while_loop(self, jit):
    if jit == "cpp" and not config.omnistaging_enabled:
      self.skipTest("cpp_jit requires omnistaging")
    @partial(_maybe_jit, jit)
    def count_to(N):
      return lax.while_loop(lambda x: x < N, lambda x: x + 1.0, 0.0)

    with enable_x64():
      self.assertArraysEqual(count_to(10), jnp.float64(10), check_dtypes=True)

    with disable_x64():
      self.assertArraysEqual(count_to(10), jnp.float32(10), check_dtypes=True)

  def test_thread_safety(self):
    def func_x32():
      with disable_x64():
        time.sleep(0.1)
        return jnp.arange(10).dtype

    def func_x64():
      with enable_x64():
        time.sleep(0.1)
        return jnp.arange(10).dtype

    with concurrent.futures.ThreadPoolExecutor() as executor:
      x32 = executor.submit(func_x32)
      x64 = executor.submit(func_x64)
      self.assertEqual(x64.result(), jnp.int64)
      self.assertEqual(x32.result(), jnp.int32)

  def test_jit_cache(self):
    if jtu.device_under_test() == "tpu":
      self.skipTest("64-bit random not available on TPU")
    if jax.lib._xla_extension_version < 4 and FLAGS.experimental_cpp_jit:
      self.skipTest(
          "Known failure due to https://github.com/google/jax/issues/5532")

    f = partial(random.uniform, random.PRNGKey(0), (1,), 'float64', -1, 1)
    with disable_x64():
      for _ in range(2):
        f()
    with enable_x64():
      for _ in range(2):
        f()

  def test_convert_element_type(self):
    # Regression test for part of https://github.com/google/jax/issues/5982
    with enable_x64():
      x = jnp.int64(1)
    self.assertEqual(x.dtype, jnp.int64)

    y = x.astype(jnp.int32)
    self.assertEqual(y.dtype, jnp.int32)

    z = api.jit(lambda x: x.astype(jnp.int32))(x)
    self.assertEqual(z.dtype, jnp.int32)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
