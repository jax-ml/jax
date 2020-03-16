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


from functools import partial, reduce
import operator as op
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as onp

from jax import core
from jax import test_util as jtu

import jax.numpy as np
from jax import random
from jax import jacobian, jit
from jax.experimental import stax
from jax.experimental.jet import jet, fact

from jax.config import config
config.parse_flags_with_absl()


def jvp_taylor(fun, primals, series):
  order, = set(map(len, series))
  def composition(eps):
    taylor_terms = [sum([eps ** (i+1) * terms[i] / fact(i + 1)
                         for i in range(len(terms))]) for terms in series]
    nudged_args = [x + t for x, t in zip(primals, taylor_terms)]
    return fun(*nudged_args)
  primal_out = fun(*primals)
  terms_out = [repeated(jacobian, i+1)(composition)(0.) for i in range(order)]
  return primal_out, terms_out

def repeated(f, n):
  def rfun(p):
    return reduce(lambda x, _: f(x), range(n), p)
  return rfun

class JetTest(jtu.JaxTestCase):

  def check_jet(self, fun, primals, series, atol=1e-5, rtol=1e-5,
                check_dtypes=True):
    y, terms = jet(fun, primals, series)
    expected_y, expected_terms = jvp_taylor(fun, primals, series)
    self.assertAllClose(y, expected_y, atol=atol, rtol=rtol,
                        check_dtypes=check_dtypes)
    self.assertAllClose(terms, expected_terms, atol=atol, rtol=rtol,
                        check_dtypes=check_dtypes)

  @jtu.skip_on_devices("tpu")
  def test_exp(self):
    order, dim = 4, 3
    rng = onp.random.RandomState(0)
    primal_in = rng.randn(dim)
    terms_in = [rng.randn(dim) for _ in range(order)]
    self.check_jet(np.exp, (primal_in,), (terms_in,), atol=1e-4, rtol=1e-4)

  @jtu.skip_on_devices("tpu")
  def test_log(self):
    order, dim = 4, 3
    rng = onp.random.RandomState(0)
    primal_in = np.exp(rng.randn(dim))
    terms_in = [rng.randn(dim) for _ in range(order)]
    self.check_jet(np.log, (primal_in,), (terms_in,), atol=1e-4, rtol=1e-4)

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

  @jtu.skip_on_devices("tpu")
  def test_div(self):
    primals = 1., 5.
    order = 4
    rng = onp.random.RandomState(0)
    series_in = ([rng.randn() for _ in range(order)], [rng.randn() for _ in range(order)])
    self.check_jet(op.truediv, primals, series_in)

  @jtu.skip_on_devices("tpu")
  def test_sub(self):
    primals = 1., 5.
    order = 4
    rng = onp.random.RandomState(0)
    series_in = ([rng.randn() for _ in range(order)], [rng.randn() for _ in range(order)])
    self.check_jet(op.sub, primals, series_in)

  @jtu.skip_on_devices("tpu")
  def test_gather(self):
    order, dim = 4, 3
    rng = onp.random.RandomState(0)
    x = rng.randn(dim)
    terms_in = [rng.randn(dim) for _ in range(order)]
    self.check_jet(lambda x: x[1:], (x,), (terms_in,))

  @jtu.skip_on_devices("tpu")
  def test_reduce_max(self):
    dim1, dim2 = 3, 5
    order = 6
    rng = onp.random.RandomState(0)
    x = rng.randn(dim1, dim2)
    terms_in = [rng.randn(dim1, dim2) for _ in range(order)]
    self.check_jet(lambda x: x.max(axis=1), (x,), (terms_in,))


if __name__ == '__main__':
  absltest.main()
