"""precision doubling arithmetic transform.

Following the approach of Dekker 1971
(http://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf).
"""
from typing import Any

from jax.util import curry
from jax import core, lax, grad
import jax.numpy as jnp
import jax.linear_util as lu


class DoublingTracer(core.Tracer):  
  def __init__(self, trace, hi, lo):
    self._trace = trace
    self.hi = hi
    self.lo = lo

  @property
  def aval(self):
    return core.raise_to_shaped(core.get_aval(self.hi))
  
  def full_lower(self):
    return self


class DoublingTrace(core.Trace):
  def pure(self, val: Any):
    return DoublingTracer(self, val, jnp.zeros(jnp.shape(val), jnp.result_type(val)))

  def lift(self, val: core.Tracer):
    return DoublingTracer(self, val, jnp.zeros(jnp.shape(val), jnp.result_type(val)))

  def sublift(self, val: DoublingTracer):
    return DoublingTracer(self, val.hi, val.lo)

  def process_primitive(self, primitive, tracers, params):
    func = doubling_rules.get(primitive, None)
    if func is None:
      raise NotImplementedError(f"primitive={primitive}")
    out = func(*((t.hi, t.lo) for t in tracers))
    return DoublingTracer(self, *out)


@lu.transformation
def doubling_transform(*args):
  with core.new_master(DoublingTrace) as master:
    trace = DoublingTrace(master, core.cur_sublevel())
    in_tracers = [DoublingTracer(trace, hi, lo) for hi, lo in args]
    outputs = yield in_tracers, {}
    out_tracers = list(map(trace.full_raise, outputs))
    out_pairs = [(x.hi, x.lo) for x in out_tracers]
  yield out_pairs


@curry
def doubledouble(f, *args):
  g = doubling_transform(lu.wrap_init(f))
  # TODO: flatten pytrees
  args = ((a, jnp.zeros(jnp.shape(a), jnp.result_type(a))) for a in args)
  out, = g.call_wrapped(*args)
  return out[0] + out[1]  


# Following routines come from Dekker 1971
doubling_rules = {}

_nmant = jnp.finfo(jnp.float64).nmant
_mul_const = (2 << (_nmant - _nmant // 2)) + 1

def _abs2(x):
  x, xx = x
  return jnp.where(
    lax.sign(x) == lax.sign(xx),
    (lax.abs(x), lax.abs(xx)),
    (lax.abs(x), -lax.abs(xx))
  )
doubling_rules[lax.abs_p] = _abs2

def _neg2(x):
  return (-x[0], -x[1])
doubling_rules[lax.neg_p] = _neg2

def _add2(x, y):
  (x, xx), (y, yy) = x, y
  r = x + y
  s = jnp.where(
    lax.abs(x) > lax.abs(y),
    x - r + y + yy + xx,
    y - r + x + xx + yy,
  )
  z = r + s
  zz = r - z + s
  return (z, zz)
doubling_rules[lax.add_p] = _add2

def _sub2(x, y):
  (x, xx), (y, yy) = x, y
  r = x - y
  s = jnp.where(
    lax.abs(x) > lax.abs(y),
    x - r - y - yy + xx,
    -y - r + x + xx - yy,
  )
  z = r + s
  zz = r - z + s
  return (z, zz)
doubling_rules[lax.sub_p] = _sub2

def _mul12(x, y):
  p = x * _mul_const
  hx = x - p + p
  tx = x - hx
  p = y * _mul_const
  hy = y - p + p
  ty = y - hy
  p = hx * hy
  q = hx * ty + tx * hy
  z = p + q
  zz = p - z + q + tx * ty
  return z, zz

def _mul2(x, y):
  (x, xx), (y, yy) = x, y
  c, cc = _mul12(x, y)
  cc = x * yy + xx * y + cc
  z = c + cc
  zz = c - z + cc
  return (z, zz)
doubling_rules[lax.mul_p] = _mul2

def _div2(x, y):
  (x, xx), (y, yy) = x, y
  c = x / y
  u, uu = _mul12(c, y)
  cc = (x - u - uu + xx - c * yy) / y
  z = c + cc
  zz = c - z + cc
  return z, zz
doubling_rules[lax.div_p] = _div2

def _sqrt2(x):
  x, xx = x
  c = jnp.sqrt(x)
  u, uu = mul12(c, c)
  cc = x - u - uu + xx * 0.5 / c
  y = c + cc
  yy = c - y + cc
  return y, yy
doubling_rules[lax.sqrt_p] = _sqrt2


if __name__ == '__main__':

  @doubledouble
  def f(x, y):
    out = x * y + 2
    return (out,)

  result = f(10., 5.)
  print(result)
