import numpy.random as npr
from scipy.special import factorial as fact
from fdm import central_fdm

from jax import jvp, jet
import jax.numpy as np


def fdm_taylor(f, primals, series):
  def expansion(eps):
    tayterms = [
        sum([eps**(i + 1) * terms[i] / fact(i + 1) for i in range(len(terms))])
        for terms in series
    ]
    return f(*map(sum, zip(primals, tayterms)))

  n_derivs = []
  N = len(series[0]) + 1
  for i in range(1, N):
    d = central_fdm(order=(i + 4), deriv=i, condition=0)(expansion, 0.)
    n_derivs.append(d)
  return f(*primals), n_derivs


def fdm_test_jet(f, primals, series, atol=1e-3):
  y, terms = jet(f, primals, series)
  y_fdm, terms_fdm = fdm_taylor(f, primals, series)
  assert np.allclose(y, y_fdm)
  assert np.allclose(terms, terms_fdm, atol=atol)


def test_exp():
  raise NotImplementedError


def test_log():
  raise NotImplementedError


def test_tanh():
  raise NotImplementedError


def test_dot():
  D = 2
  N = 4
  x1 = npr.randn(D)
  x2 = npr.randn(D)
  primals = (x1,x2)
  terms_in = npr.randn(N, D)
  series_in = (terms_in,terms_in)
  fdm_test_jet(np.dot,primals,series_in,atol=1e-2)


def test_sin():
  N = 4
  x = npr.randn()
  terms_in = npr.randn(N)
  fdm_test_jet(np.sin, (x,),(terms_in,),atol=1e-2)


def test_cos():
  N = 4
  x = npr.randn()
  terms_in = npr.randn(N)
  fdm_test_jet(np.cos, (x,),(terms_in,),atol=1e-2)


## Test Combinations?
def test_sin_sin():
  N = 4
  x = npr.randn()
  terms_in = npr.randn(N)
  f = lambda x: np.sin(np.sin(x))
  fdm_test_jet(f, (x,),(terms_in,),atol=1e-2)


# def test_vector_sin():
#   D = 10
#   x = npr.randn(D)

#   def f(x): return np.sin(x)

#   vs = [x, npr.randn(D), np.zeros(D), np.zeros(D), np.zeros(D)]
#   terms = prop(make_derivs_sin, vs)
#   assert np.isclose(terms, jvps(f, x, vs[1], 4)).all()

# def test_vector_sin_sin():
#   D = 1
#   x = npr.randn(D)

#   def f(x): return np.sin(np.sin(x))

#   vs = [x, np.ones(D), np.zeros(D), np.zeros(D), np.zeros(D)]
#   terms1 = prop(make_derivs_sin, vs)
#   terms2 = prop(make_derivs_sin, terms1)
#   assert np.isclose(terms2, jvps(f, x, vs[1], 4)).all()
