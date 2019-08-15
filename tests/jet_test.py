import numpy.random as npr
from scipy.special import factorial as fact

from jax import jvp, jet, grad
import jax.numpy as np


def repeated(f, n):
  def rfun(p):
    return reduce(lambda x, _: f(x), xrange(n), p)

  return rfun


def jvp_taylor(f, primals, series):
  def expansion(eps):
    tayterms = [
        sum([eps**(i + 1) * terms[i] / fact(i + 1) for i in range(len(terms))])
        for terms in series
    ]
    return f(*map(sum, zip(primals, tayterms)))

  n_derivs = []
  N = len(series[0]) + 1
  for i in range(1, N):
    d = repeated(grad, i)(expansion)(0.)
    n_derivs.append(d)
  return f(*primals), n_derivs


def jvp_test_jet(f, primals, series, atol=1e-5):
  y, terms = jet(f, primals, series)
  y_jvp, terms_jvp = jvp_taylor(f, primals, series)
  assert np.allclose(y, y_jvp)
  assert np.allclose(terms, terms_jvp, atol=atol)

#XXX
def test_exp():
  N = 4
  x = npr.randn()
  terms_in = list(npr.randn(N))
  jvp_test_jet(np.exp, (x, ), (terms_in, ), atol=1e-2)


def test_log():
  raise NotImplementedError


def test_tanh():
  raise NotImplementedError


def test_sin():
  N = 4
  x = npr.randn()
  terms_in = list(npr.randn(N))
  jvp_test_jet(np.sin, (x, ), (terms_in, ))


def test_cos():
  N = 4
  x = npr.randn()
  terms_in = list(npr.randn(N))
  jvp_test_jet(np.cos, (x, ), (terms_in, ), atol=1e-1)


def test_neg():
  N = 4
  x = npr.randn()
  terms_in = list(npr.randn(N))
  f = lambda x: -x
  jvp_test_jet(f, (x, ), (terms_in, ), atol=1e-2)


def test_dot():
  D = 2
  N = 4
  x1 = npr.randn(D)
  x2 = npr.randn(D)
  primals = (x1, x2)
  terms_in = list(npr.randn(N, D))
  series_in = (terms_in, terms_in)
  jvp_test_jet(np.dot, primals, series_in)


def test_mul():
  N = 4
  x1 = npr.randn()
  x2 = npr.randn()
  f = lambda a, b: a * b
  primals = (x1, x2)
  terms_in = list(npr.randn(N))
  series_in = ( terms_in, terms_in )
  jvp_test_jet(f, primals, series_in)


## Test Combinations?
def test_sin_sin():
  N = 4
  x = npr.randn()
  terms_in = npr.randn(N)
  f = lambda x: np.sin(np.sin(x))
  jvp_test_jet(f, (x, ), (terms_in, ), atol=1e-2)


def test_expansion():
  N = 4
  x = npr.randn()
  terms_in = [1., 0., 0., 0.]
  f = np.sin
  print jet(f, (x, ), (terms_in, ))
  return expand(f, (x, ), (terms_in, ))
  # return jet(f,(x,),(terms_in,))


def expand(f, primals, series):
  def expansion(eps):
    tayterms = [
        sum([eps**(i + 1) * terms[i] / fact(i + 1) for i in range(len(terms))])
        for terms in series
    ]
    return f(*map(sum, zip(primals, tayterms)))

  return grad(grad(grad(expansion)))(0.)


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
