from jax import jvp, jet
import jax.numpy as np

import numpy.random as npr


def jvps(f, x, v, n):
    def j_i(f):
        return lambda x: jvp(f,(x,),(v,))[1]
    funcs = [f]
    for i in range(n):
        funcs.append(j_i(funcs[i]))
    return [f(x) for f in funcs]


def test_sin():
  x = 4.0
  vs = (1., 0., 0., 0.)
  y, terms = jet(np.sin, (x,), [vs])
  expected = jvps(np.sin, x, vs[0], 4)
  assert np.allclose(y, expected[0])
  assert all(map(np.allclose, terms, expected[1:]))

def test_sin_sin():
  x = 4.0
  vs = (1., 0., 0., 0.)
  f = lambda x: np.sin(np.sin(x))
  y, terms = jet(f, (x,), [vs])
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
