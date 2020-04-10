# Copyright 2020 Google LLC
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


from functools import reduce

from absl.testing import absltest
import numpy as onp

from jax import test_util as jtu
import jax.numpy as np
from jax import random
from jax import jacfwd, jit
from jax.experimental import stax
from jax.experimental.jet import jet, fact, zero_series
from jax.tree_util import tree_map
from jax import lax

from jax.config import config
config.parse_flags_with_absl()

def jvp_taylor(fun, primals, series):
  # Computes the Taylor series the slow way, with nested jvp.
  order, = set(map(len, series))
  def composition(eps):
    taylor_terms = [sum([eps ** (i+1) * terms[i] / fact(i + 1)
                         for i in range(len(terms))]) for terms in series]
    nudged_args = [x + t for x, t in zip(primals, taylor_terms)]
    return fun(*nudged_args)
  primal_out = fun(*primals)
  terms_out = [repeated(jacfwd, i+1)(composition)(0.) for i in range(order)]
  return primal_out, terms_out

def repeated(f, n):
  def rfun(p):
    return reduce(lambda x, _: f(x), range(n), p)
  return rfun

def transform(lims, x):
  return x * (lims[1] - lims[0]) + lims[0]

class JetTest(jtu.JaxTestCase):

  def check_jet(self, fun, primals, series, atol=1e-5, rtol=1e-5,
                check_dtypes=True):
    y, terms = jet(fun, primals, series)
    expected_y, expected_terms = jvp_taylor(fun, primals, series)
    self.assertAllClose(y, expected_y, atol=atol, rtol=rtol,
                        check_dtypes=check_dtypes)

    # TODO(duvenaud): Lower zero_series to actual zeros automatically.
    if terms == zero_series:
      terms = tree_map(np.zeros_like, expected_terms)

    self.assertAllClose(terms, expected_terms, atol=atol, rtol=rtol,
                        check_dtypes=check_dtypes)

  @jtu.skip_on_devices("tpu")
  def test_dot(self):
    M, K, N = 2, 3, 4
    order = 3
    rng = onp.random.RandomState(0)
    x1 = rng.randn(M, K)
    x2 = rng.randn(K, N)
    primals = (x1, x2)
    terms_in1 = [rng.randn(*x1.shape) for _ in range(order)]
    terms_in2 = [rng.randn(*x2.shape) for _ in range(order)]
    series_in = (terms_in1, terms_in2)
    self.check_jet(np.dot, primals, series_in)

  @jtu.skip_on_devices("tpu")
  def test_conv(self):
    order = 3
    input_shape = (1, 5, 5, 1)
    key = random.PRNGKey(0)
    # TODO(duvenaud): Check all types of padding
    init_fun, apply_fun = stax.Conv(3, (2, 2), padding='VALID')
    _, (W, b) = init_fun(key, input_shape)

    rng = onp.random.RandomState(0)

    x = rng.randn(*input_shape).astype("float32")
    primals = (W, b, x)

    series_in1 = [rng.randn(*W.shape).astype("float32") for _ in range(order)]
    series_in2 = [rng.randn(*b.shape).astype("float32") for _ in range(order)]
    series_in3 = [rng.randn(*x.shape).astype("float32") for _ in range(order)]

    series_in = (series_in1, series_in2, series_in3)

    def f(W, b, x):
      return apply_fun((W, b), x)

    self.check_jet(f, primals, series_in, check_dtypes=False)

  def unary_check(self, fun, lims=[-2, 2], order=3):
    dims = 2, 3
    rng = onp.random.RandomState(0)
    primal_in = transform(lims, rng.rand(*dims))
    terms_in = [rng.randn(*dims) for _ in range(order)]
    self.check_jet(fun, (primal_in,), (terms_in,), atol=1e-4, rtol=1e-4)

  def binary_check(self, fun, lims=[-2, 2], order=3):
    dims = 2, 3
    rng = onp.random.RandomState(0)
    primal_in = (transform(lims, rng.rand(*dims)),
                 transform(lims, rng.rand(*dims)))
    series_in = ([rng.randn(*dims) for _ in range(order)],
                 [rng.randn(*dims) for _ in range(order)])
    self.check_jet(fun, primal_in, series_in, atol=1e-4, rtol=1e-4)

  @jtu.skip_on_devices("tpu")
  def test_exp(self):        self.unary_check(np.exp)
  @jtu.skip_on_devices("tpu")
  def test_neg(self):        self.unary_check(np.negative)
  @jtu.skip_on_devices("tpu")
  def test_floor(self):      self.unary_check(np.floor)
  @jtu.skip_on_devices("tpu")
  def test_ceil(self):       self.unary_check(np.ceil)
  @jtu.skip_on_devices("tpu")
  def test_round(self):      self.unary_check(np.round)
  @jtu.skip_on_devices("tpu")
  def test_sign(self):       self.unary_check(np.sign)
  @jtu.skip_on_devices("tpu")
  def test_log(self):        self.unary_check(np.log, lims=[0.8, 4.0])
  @jtu.skip_on_devices("tpu")
  def test_gather(self):     self.unary_check(lambda x: x[1:])
  @jtu.skip_on_devices("tpu")
  def test_reduce_max(self): self.unary_check(lambda x: x.max(axis=1))
  @jtu.skip_on_devices("tpu")
  def test_reduce_min(self): self.unary_check(lambda x: x.min(axis=1))
  @jtu.skip_on_devices("tpu")
  def test_all_max(self):    self.unary_check(np.max)
  @jtu.skip_on_devices("tpu")
  def test_all_min(self):    self.unary_check(np.min)
  @jtu.skip_on_devices("tpu")
  def test_stopgrad(self):   self.unary_check(lax.stop_gradient)
  @jtu.skip_on_devices("tpu")
  def test_abs(self):        self.unary_check(np.abs)
  @jtu.skip_on_devices("tpu")
  def test_fft(self):        self.unary_check(np.fft.fft)
  @jtu.skip_on_devices("tpu")
  def test_log1p(self):      self.unary_check(np.log1p, lims=[0, 4.])
  @jtu.skip_on_devices("tpu")
  def test_expm1(self):      self.unary_check(np.expm1)

  @jtu.skip_on_devices("tpu")
  def test_div(self):   self.binary_check(lambda x, y: x / y, lims=[0.8, 4.0])
  @jtu.skip_on_devices("tpu")
  def test_sub(self):   self.binary_check(lambda x, y: x - y)
  @jtu.skip_on_devices("tpu")
  def test_add(self):   self.binary_check(lambda x, y: x + y)
  @jtu.skip_on_devices("tpu")
  def test_mul(self):   self.binary_check(lambda x, y: x * y)
  @jtu.skip_on_devices("tpu")
  def test_le(self):    self.binary_check(lambda x, y: x <= y)
  @jtu.skip_on_devices("tpu")
  def test_gt(self):    self.binary_check(lambda x, y: x > y)
  @jtu.skip_on_devices("tpu")
  def test_lt(self):    self.binary_check(lambda x, y: x < y)
  @jtu.skip_on_devices("tpu")
  def test_ge(self):    self.binary_check(lambda x, y: x >= y)
  @jtu.skip_on_devices("tpu")
  def test_eq(self):    self.binary_check(lambda x, y: x == y)
  @jtu.skip_on_devices("tpu")
  def test_ne(self):    self.binary_check(lambda x, y: x != y)
  @jtu.skip_on_devices("tpu")
  def test_and(self):   self.binary_check(lambda x, y: np.logical_and(x, y))
  @jtu.skip_on_devices("tpu")
  def test_or(self):    self.binary_check(lambda x, y: np.logical_or(x, y))
  @jtu.skip_on_devices("tpu")
  def test_xor(self):   self.binary_check(lambda x, y: np.logical_xor(x, y))

  def test_process_call(self):
    def f(x):
      return jit(lambda x: x * x)(x)
    self.unary_check(f)

  def test_post_process_call(self):
    def f(x):
      return jit(lambda y: x * y)(2.)

    self.unary_check(f)

  def test_select(self):
    M, K = 2, 3
    order = 3
    rng = onp.random.RandomState(0)
    b = rng.rand(M, K) < 0.5
    x = rng.randn(M, K)
    y = rng.randn(M, K)
    primals = (b, x, y)
    terms_b = [rng.randn(*b.shape) for _ in range(order)]
    terms_x = [rng.randn(*x.shape) for _ in range(order)]
    terms_y = [rng.randn(*y.shape) for _ in range(order)]
    series_in = (terms_b, terms_x, terms_y)
    self.check_jet(np.where, primals, series_in)


if __name__ == '__main__':
  absltest.main()
