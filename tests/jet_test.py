import numpy.random as npr
from scipy.special import factorial as fact
from fdm import central_fdm

from jax import jvp, jet
import jax.numpy as np


def fdm_taylor(f, primals, terms, n):
  def expansion(eps):
    return f(
        primals +
        sum([eps**(i + 1) * terms[i] / fact(i+1) for i in range(len(terms))]))

  n_derivs = []
  for i in range(n):
    d = central_fdm(order=(i+2)*2, deriv=i,condition=0)(expansion, 0.)
    n_derivs.append(d)
  return n_derivs


def test_exp():
  raise NotImplementedError


def test_log():
  raise NotImplementedError


def test_tanh():
  raise NotImplementedError


def test_sin():
  x = 4.0
  vs = (1., 0., 0., 0.)
  y, terms = jet(np.sin, (x, ), [vs])
  expected = jvps(np.sin, x, vs[0], 4)
  assert np.allclose(y, expected[0])
  assert all(map(np.allclose, terms, expected[1:]))


def test_cos():
  raise NotImplementedError


def test_cos():
  raise NotImplementedError


## Test Combinations?
def test_sin_sin():
  x = 4.0
  vs = (1., 0., 0., 0.)
  f = lambda x: np.sin(np.sin(x))
  y, terms = jet(f, (x, ), [vs])
  expected = jvps(f, x, vs[0], 4)
  assert np.allclose(y, expected[0])
  assert all(map(np.allclose, terms, expected[1:]))


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
