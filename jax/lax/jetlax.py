from lax import *
import jax.numpy as np
from ..interpreters import fdb

def make_derivs_neg(primals,order):
  ans = neg(*primals)
  def nth(vs):
    return vs[0]
  derivs = itertools.chain(itertools.repeat(nth))
  return ans, list(itertools.islice(derivs,order))
fdb.jet_rules[neg_p] = make_derivs_neg

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

def make_derivs_dot(primals, order):
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

