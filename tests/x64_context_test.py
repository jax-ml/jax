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

from absl.testing import parameterized

from jax import api, config, lax, partial, random
from jax.experimental import enable_x64, disable_x64
import jax.numpy as jnp
import jax.test_util as jtu

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
      self.assertEqual(func().dtype, 'float64')
    with disable_x64():
      self.assertEqual(func().dtype, 'float32')
    self.assertEqual(func().dtype, dtype_start)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_jit={}".format(jit), "jit": jit}
      for jit in ["python", "cpp", None]))
  def test_near_singular_inverse(self, jit):
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
