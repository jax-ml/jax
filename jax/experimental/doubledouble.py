"""precision doubling arithmetic transform.

Following the approach of Dekker 1971
(http://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf).
"""
from typing import Any, Sequence

from jax.util import curry
from jax import core, lax, grad
import jax.numpy as jnp
import jax.linear_util as lu


class DoublingTracer(core.Tracer):  
  def __init__(self, trace, head, tail):
    self._trace = trace
    # TODO(vanderplas): check head/tail have matching shapes & dtypes
    self.head = head
    self.tail = tail

  @property
  def aval(self):
    return core.raise_to_shaped(core.get_aval(self.head))
  
  def full_lower(self):
    return self


class DoublingTrace(core.Trace):
  def pure(self, val: Any):
    return DoublingTracer(self, val, jnp.zeros(jnp.shape(val), jnp.result_type(val)))

  def lift(self, val: core.Tracer):
    return DoublingTracer(self, val, jnp.zeros(jnp.shape(val), jnp.result_type(val)))

  def sublift(self, val: DoublingTracer):
    return DoublingTracer(self, val.head, val.tail)

  def process_primitive(self, primitive, tracers, params):
    func = doubling_rules.get(primitive, None)
    if func is None:
      raise NotImplementedError(f"primitive={primitive}")
    out = func(*((t.head, t.tail) for t in tracers))
    return DoublingTracer(self, *out)


@lu.transformation
def doubling_transform(*args):
  with core.new_master(DoublingTrace) as master:
    trace = DoublingTrace(master, core.cur_sublevel())
    in_tracers = [DoublingTracer(trace, hi, lo) for hi, lo in args]
    outputs = yield in_tracers, {}
    if isinstance(outputs, Sequence):
      out_tracers = map(trace.full_raise, outputs)
      result = [(x.head, x.tail) for x in out_tracers]
    else:
      out_tracer = trace.full_raise(outputs)
      result = (out_tracer.head, out_tracer.tail)
  yield result


@curry
def doubledouble(f, *args):
  g = doubling_transform(lu.wrap_init(f))
  # TODO: flatten pytrees
  args = ((a, jnp.zeros_like(a)) for a in args)
  out = g.call_wrapped(*args)
  if isinstance(out, list):
    return tuple(o[0] + o[1] for o in out)
  else:
    return out[0] + out[1]  


doubling_rules = {}

def _mul_const(dtype):
  _nmant = jnp.finfo(jnp.float64).nmant
  return jnp.array((2 << (_nmant - _nmant // 2)) + 1, dtype=dtype)

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
  dtype = jnp.result_type(x, y)
  K = _mul_const(dtype)
  p = x * K
  hx = x - p + p
  tx = x - hx
  p = y * K
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
    return abs(x + y) - abs(x)

  result = f(-1E20, -1.0)
  print(result)
