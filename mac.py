import jax.numpy as np
from jax import jet, jvp

def f(x, y):
  return x + 2 * np.exp(y)

out = jet(f, (1., 2.), [(1., 0.), (1., 0.)])
print(out)

out = jvp(f, (1., 2.), (1., 1.))
print(out)


###

from functools import reduce
import numpy.random as npr
from jax import jacobian

from scipy.special import factorial as fact

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
    d = repeated(jacobian, i)(expansion)(0.)
    n_derivs.append(d)
  return f(*primals), n_derivs

def repeated(f, n):
  def rfun(p):
    return reduce(lambda x, _: f(x), range(n), p)
  return rfun

def jvp_test_jet(f, primals, series, atol=1e-5):
  y, terms = jet(f, primals, series)
  y_jvp, terms_jvp = jvp_taylor(f, primals, series)
  # import ipdb; ipdb.set_trace()
  assert np.allclose(y, y_jvp)
  assert np.allclose(terms, terms_jvp, atol=atol)

def test_exp():
  npr.seed(0)
  D = 3  # dimensionality
  N = 6  # differentiation order
  x = npr.randn(D)
  terms_in = list(npr.randn(N,D))
  jvp_test_jet(np.exp, (x,), (terms_in,), atol=1e-4)


def test_dot():
  D = 6
  N = 4
  x1 = npr.randn(D)
  x2 = npr.randn(D)
  primals = (x1, x2)
  terms_in1 = list(npr.randn(N,D))
  terms_in2 = list(npr.randn(N,D))
  series_in = (terms_in1, terms_in2)
  jvp_test_jet(np.dot, primals, series_in)



def test_mlp():
  sigm = lambda x: 1. / (1. + np.exp(-x))
  def mlp(M1,M2,x):
    return np.dot(sigm(np.dot(x,M1)),M2)
  f_mlp = lambda x: mlp(M1,M2,x)
  M1,M2 = (npr.randn(10,10), npr.randn(10,5))
  x= npr.randn(2,10)
  terms_in = [np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]
  jvp_test_jet(f_mlp,(x,),[terms_in])

test_exp()
test_dot()
test_mlp()  # TODO add div rule!
