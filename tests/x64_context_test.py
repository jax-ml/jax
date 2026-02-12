# Copyright 2021 The JAX Authors.
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
from functools import partial
import time

from absl.testing import absltest
import numpy as np

import jax
from jax import custom_jvp, custom_vjp, grad, jvp
from jax import lax
from jax import random
import jax.numpy as jnp
from jax._src import config
import jax._src.test_util as jtu


jax.config.parse_flags_with_absl()

# TODO(jakevdp): remove this check for JAX v0.9.0 and test jax.enable_x64 directly.
with jtu.ignore_warning(message=".* is deprecated", category=DeprecationWarning):
  if hasattr(jax.experimental, "enable_x64"):
    from jax.experimental import enable_x64, disable_x64
  else:
    enable_x64 = jax.enable_x64
    disable_x64 = lambda: jax.enable_x64(False)


class X64ContextTests(jtu.JaxTestCase):
  @jtu.sample_product(jit=jtu.JIT_IMPLEMENTATION)
  def test_make_array(self, jit):
    func = jit(lambda: jnp.array(np.float64(0)))
    dtype_start = func().dtype
    with enable_x64():
      self.assertEqual(func().dtype, "float64")
    with disable_x64():
      self.assertEqual(func().dtype, "float32")
    self.assertEqual(func().dtype, dtype_start)

  @jtu.sample_product(
    jit=jtu.JIT_IMPLEMENTATION,
    enable_or_disable=[enable_x64, disable_x64],
  )
  def test_correctly_capture_default(self, jit, enable_or_disable):
    # The fact we defined a jitted function with a block with a different value
    # of `jax.config.enable_x64` has no impact on the output.
    with enable_or_disable():
      func = jit(lambda: jnp.array(np.float64(0)))
      func()

    expected_dtype = "float64" if jax.config._read("jax_enable_x64") else "float32"
    self.assertEqual(func().dtype, expected_dtype)

    with enable_x64():
      self.assertEqual(func().dtype, "float64")
    with disable_x64():
      self.assertEqual(func().dtype, "float32")

  @jtu.sample_product(jit=jtu.JIT_IMPLEMENTATION)
  @jtu.run_on_devices("cpu")  # Test presumes CPU precision
  def test_near_singular_inverse(self, jit):
    rng = jtu.rand_default(self.rng())

    @partial(jit, static_argnums=1)
    def near_singular_inverse(N=5, eps=1E-40):
      X = rng((N, N), dtype='float64')
      X = jnp.asarray(X)
      X = X.at[-1].mul(eps)
      return jnp.linalg.inv(X)

    with enable_x64():
      result_64 = near_singular_inverse()
      self.assertTrue(jnp.all(jnp.isfinite(result_64)))

    with disable_x64():
      result_32 = near_singular_inverse()
      self.assertTrue(jnp.all(~jnp.isfinite(result_32)))

  @jtu.sample_product(jit=jtu.JIT_IMPLEMENTATION)
  def test_while_loop(self, jit):
    @jit
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
        return jnp.array(np.int64(0)).dtype

    def func_x64():
      with enable_x64():
        time.sleep(0.1)
        return jnp.array(np.int64(0)).dtype

    with concurrent.futures.ThreadPoolExecutor() as executor:
      x32 = executor.submit(func_x32)
      x64 = executor.submit(func_x64)
      self.assertEqual(x64.result(), jnp.int64)
      self.assertEqual(x32.result(), jnp.int32)

  @jax.legacy_prng_key('allow')
  @jax.debug_key_reuse(False)
  @jtu.ignore_warning(category=UserWarning,
                      message="Explicitly requested dtype float64 is not available")
  def test_jit_cache(self):
    if jtu.test_device_matches(["tpu"]):
      self.skipTest("64-bit random not available on TPU")

    if config.explicit_x64_dtypes.value == config.ExplicitX64Mode.ERROR:
      self.skipTest("Test uses float64 which is not available")

    f = partial(random.uniform, random.PRNGKey(0), (1,), 'float64', -1, 1)
    with disable_x64():
      for _ in range(2):
        f()
    with enable_x64():
      for _ in range(2):
        f()

  def test_convert_element_type(self):
    # Regression test for part of https://github.com/jax-ml/jax/issues/5982
    with enable_x64():
      x = jnp.int64(1)
    self.assertEqual(x.dtype, jnp.int64)

    y = x.astype(jnp.int32)
    self.assertEqual(y.dtype, jnp.int32)

    z = jax.jit(lambda x: x.astype(jnp.int32))(x)
    self.assertEqual(z.dtype, jnp.int32)

  def test_python_scalar(self):
    @jax.jit
    def f(a):
      with enable_x64():
        return 2 + a
    self.assertEqual(f(1).dtype, jnp.int64)

  def test_grad(self):
    def fun(x):
      with enable_x64(True):
        return jnp.sin(x)
    self.assertEqual(
        jax.grad(fun)(0.5).dtype,
        jnp.float64 if jax.config.x64_enabled else jnp.float32,
    )

  def test_sin(self):
    def fun(x):
      with enable_x64(True):
        x = jnp.asarray(x, dtype=jnp.float64)
      return lax.sin(x)

    self.assertEqual(fun(0.5).dtype, jnp.float64)

  def test_mul(self):
    def fun(x, y):
      with enable_x64(True):
        x = jnp.asarray(x, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)
      return lax.mul(x, y)

    self.assertEqual(fun(0.5, 1.5).dtype, jnp.float64)

  @jtu.sample_product(disable_jit=[True, False])
  def test_scan_with_contextmanager(self, disable_jit):
    with jax.disable_jit(disable_jit):
      def f(a):
        def body(carry, _):
          with enable_x64():
            y = (carry + a).astype(jnp.int64)
            assert y.dtype == jnp.int64
            z = y.astype(jnp.int32)
          return carry, (z, y)
        return lax.scan(body, jnp.int32(2), jnp.arange(4))
      carry_out, ys_out = f(3)
      self.assertEqual(carry_out.dtype, jnp.int32)
      self.assertEqual(ys_out[0].dtype, jnp.int32)
      self.assertEqual(ys_out[1].dtype, jnp.int64)

  def test_custom_jvp(self):

    @custom_jvp
    def f(x):
      return x ** 2.

    @f.defjvp
    def f_jvp(xs, ts):
      x, = xs
      t, = ts
      self.assertTrue(jax.config.x64_enabled)
      return f(x), t * jnp.sin(x)

    def g(x):
      with enable_x64():
        x = jnp.array(x, jnp.float64)
        return f(x)

    self.assertEqual(g(5.).dtype, jnp.float64)
    out_primal, out_tangent = jvp(g, (5.,), (1.,))
    self.assertEqual(out_primal.dtype, jnp.float64)
    self.assertEqual(out_tangent.dtype, jnp.float64)

    with enable_x64(False):
      self.assertEqual(g(5.).dtype, jnp.float64)
      self.assertEqual(grad(g)(5.).dtype, jnp.float32)
      self.assertEqual(grad(grad(g))(5.).dtype, jnp.float32)
      self.assertEqual(grad(grad(grad(g)))(5.).dtype, jnp.float32)

    with enable_x64(True):
      self.assertEqual(g(5.).dtype, jnp.float64)
      self.assertEqual(grad(g)(5.).dtype, jnp.float64)
      self.assertEqual(grad(grad(g))(5.).dtype, jnp.float64)
      self.assertEqual(grad(grad(grad(g)))(5.).dtype, jnp.float64)

  def test_custom_vjp(self):

    @custom_vjp
    def f(x):
      return x ** 2.

    def f_fwd(x):
      return f(x), jnp.sin(x)

    def f_bwd(res, t):
      return (res * t,)
    f.defvjp(f_fwd, f_bwd)

    def g(x):
      with enable_x64():
        x = jnp.array(x, jnp.float64)
        return f(x)

    with enable_x64(False):
      self.assertEqual(g(5.).dtype, jnp.float64)
      self.assertEqual(grad(g)(5.).dtype, jnp.float32)
      self.assertEqual(grad(grad(g))(5.).dtype, jnp.float32)
      self.assertEqual(grad(grad(grad(g)))(5.).dtype, jnp.float32)

    with enable_x64(True):
      self.assertEqual(g(5.).dtype, jnp.float64)
      self.assertEqual(grad(g)(5.).dtype, jnp.float64)
      self.assertEqual(grad(grad(g))(5.).dtype, jnp.float64)
      self.assertEqual(grad(grad(grad(g)))(5.).dtype, jnp.float64)

  @config.explicit_x64_dtypes("error")
  def test_promotion(self):
    with config.explicit_x64_dtypes("allow"):
      x = jnp.array(1, jnp.int64)
    # This should not raise an error. Users are free to use x64 types outside of
    # an x64 context.
    y = x + 1
    self.assertEqual(y.dtype, jnp.int64)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
