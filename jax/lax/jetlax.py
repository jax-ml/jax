from functools import partial
from lax import *
import jax.numpy as np
from ..interpreters import fdb
from ..interpreters import xla
from scipy.special import factorial as fact
from scipy.special import binom


def deflinear(prim):
  fdb.jet_rules[prim] = partial(linear_jet, prim)

def linear_jet(prim, primals, order, **params):
  ans = prim.bind(*primals, **params)
  def fst(vs):
    vs, = vs
    return prim.bind(*vs, **params)
  def nth(vs):
    return np.zeros_like(ans)
  derivs = itertools.chain([fst], itertools.repeat(nth))
  return ans, list(itertools.islice(derivs, order))

deflinear(neg_p)
deflinear(slice_p)
deflinear(xla.device_put_p)
deflinear(reshape_p)
deflinear(concatenate_p)  # TODO


def make_derivs_sin(primals, order):
  x, = primals
  sin_x = sin(x)
  def derivs():
    cos_x = cos(x)
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * cos_x
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * -sin_x
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * -cos_x
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * sin_x
  derivs = list(itertools.islice(itertools.cycle(derivs()), order))
  return sin_x, derivs
fdb.jet_rules[sin_p] = make_derivs_sin

def make_derivs_cos(primals, order):
  x, = primals
  cos_x = cos(x)
  def derivs():
    sin_x = sin(x)
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * -sin_x
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * -cos_x
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * sin_x
    yield lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * cos_x
  derivs = list(itertools.islice(itertools.cycle(derivs()), order))
  return cos_x, derivs
fdb.jet_rules[cos_p] = make_derivs_cos

def make_derivs_sqrt(primals, order, **params):
  x, = primals
  out = np.sqrt(x)

  def derivs(n):
    if (n - 1) % 2 == 0:
      sign = 1.
    else:
      sign = -1.
    fact_term = fact(2 * (n - 1)) / float(fact(n - 1))
    x_power = (1. - 2. * n) / 2.
    coeff = sign * fact_term * 4 ** x_power
    return lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * coeff * (x ** x_power)

  return out, [derivs(n) for n in range(1, order + 1)]
fdb.jet_rules[sqrt_p] = make_derivs_sqrt

# modified from https://github.com/sympy/sympy/blob/master/sympy/functions/combinatorial/numbers.py
# the original uses a decorator to automatically memoize
def _stirling2(n, k):
  if n == k == 0:
    return 1
  if 0 in (n, k):
    return 0
  n1 = n - 1

  # some special values
  if k == n1:
    return binom(n, 2)
  elif k == 2:
    return 2 ** n1 - 1

  # general recurrence
  return k * _stirling2(n1, k) + _stirling2(n1, k - 1)

# based on Eqn 2.4 of https://arxiv.org/pdf/0903.0117.pdf
def make_derivs_tanh(primals, order, **params):
  x, = primals
  out = np.tanh(x)

  def derivs(n):
    if n % 2 == 0:
      sign = 1.
    else:
      sign = -1.
    coeff = sign * (out + 1) * (2 ** n)
    def term(m, k):
      return (fact(k) * _stirling2(m, k) * (out - 1) ** k) / (2 ** k)
    sum_terms = 0.
    for i in range(n + 1):
      sum_terms += term(n, i)
    return lambda vs: fdb.product(map(operator.itemgetter(0), vs)) * coeff * sum_terms

  return out, [derivs(n) for n in range(1, order + 1)]
fdb.jet_rules[tanh_p] = make_derivs_tanh

def make_derivs_exp(primals,order,**params):
  x, = primals
  out = np.exp(x)
  #TODO: make generator dependent on order...
  def nth(vs):
    return fdb.product(map(operator.itemgetter(0), vs)) * out
  derivs = itertools.chain(itertools.repeat(nth))
  return out, list(itertools.islice(derivs,order))
fdb.jet_rules[exp_p] = make_derivs_exp

def make_derivs_mul(primals, order):
  a, b = primals
  def fst(vs):
    (va, vb), = vs
    return va * b + a * vb
  def snd(vs):
    (v0a, v0b), (v1a, v1b) = vs
    return v0a * v1b + v1a * v0b
  def nth(vs):
    return np.zeros_like(a)
  derivs = itertools.chain([fst,snd], itertools.repeat(nth))
  return mul(a, b), list(itertools.islice(derivs, order))
fdb.jet_rules[mul_p] = make_derivs_mul

def make_derivs_dot(primals, order, **params):
  a, b = primals
  out = dot(a, b)
  def fst(vs):
    (va, vb), = vs
    return dot(va, b) + dot(a, vb)
  def snd(vs):
    (v0a, v0b), (v1a, v1b) = vs
    return dot(v0a, v1b) + dot(v1a, v0b)
  def nth(vs):
    return np.zeros_like(out)
  derivs = itertools.chain([fst, snd], itertools.repeat(nth))
  return out, list(itertools.islice(derivs, order))
fdb.jet_rules[dot_p] = make_derivs_dot



# from scipy.integrate import odeint as odeint_impl
#
# def odeint(f,z0,t, rtol=1e-7,atol=1e-9):
#   return odeint_p.bind(f,z0,t,rtol = rtol,atol=atol)
# odeint_p = Primitive('odeint')
# odeint_p.def_impl(odeint_impl)
# def make_derivs_odeint(primals,order):
#   def nth(vs):
#     return onp.zeros_like(a)
#   derivs = itertools.chain(itertools.repeat(nth))
#   return primals, list(itertools.islice(derivs, order))
# fdb.jet_rules[odeint_p] = make_derivs_odeint

def sol(f,z,t):
  sol_p.bind(f,z,t)

def sol_impl(f,z,t):
  return f(z,t)

sol_p = Primitive('sol')
sol_p.def_impl(sol_impl)
def make_derivs_sol(primals,order):
  f,z,t = primals
  def nth(vs):
    return np.zeros_like(z)
  derivs = itertools.chain(itertools.repeat(nth))
  import pdb; pdb.set_trace()
  return f(z,t), list(itertools.islice(derivs, order))
fdb.jet_rules[sol_p] = make_derivs_sol

