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

"""Precision doubling arithmetic transform

Following the approach of Dekker 1971
(http://csclub.uwaterloo.ca/~pbarfuss/dekker1971.pdf).
"""
from functools import wraps
import operator
from typing import Any, Callable, Dict, Sequence

from jax.tree_util import tree_flatten, tree_unflatten
from jax.api_util import flatten_fun_nokwargs
from jax import ad_util, core, lax, xla
from jax.util import unzip2, wrap_name
import jax.numpy as jnp
import jax.linear_util as lu

class _Zeros:
  def __repr__(self):
    return "_zeros"
_zero = _Zeros()

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
    if self.tail is None:
      return core.full_lower(self.head)
    else:
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
    out = func(*((t.head, t.tail) for t in tracers), **params)
    # TODO: handle primitive.multiple_results
    return DoublingTracer(self, *out)

  def process_call(self, call_primitive, f, tracers, params):
    assert call_primitive.multiple_results
    heads, tails = unzip2((t.head, t.tail) for t in tracers)
    nonzero_tails, in_tree_def = tree_flatten(tails)
    f_double, out_tree_def = screen_nones(doubling_subtrace(f, self.master),
                                          len(heads), in_tree_def)
    name = params.get('name', f.__name__)
    new_params = dict(params, name=wrap_name(name, 'doubledouble'),
                      donated_invars=(False,) * (len(heads) + len(nonzero_tails)))
    result = call_primitive.bind(f_double, *heads, *nonzero_tails, **new_params)
    heads_out, tails_out = tree_unflatten(out_tree_def(), result)
    return [DoublingTracer(self, h, t) for h, t in zip(heads_out, tails_out)]


@lu.transformation
def doubling_subtrace(master, heads, tails):
  trace = DoublingTrace(master, core.cur_sublevel())
  in_tracers = [DoublingTracer(trace, h, t) if t is not None else h
                for h, t in zip(heads, tails)]
  ans = yield in_tracers, {}
  out_tracers = map(trace.full_raise, ans)
  yield unzip2([(out_tracer.head, out_tracer.tail)
                for out_tracer in out_tracers])


@lu.transformation_with_aux
def screen_nones(num_heads, in_tree_def, *heads_and_tails):
  new_heads  = heads_and_tails[:num_heads]
  new_tails = heads_and_tails[num_heads:]
  new_tails = tree_unflatten(in_tree_def, new_tails)
  head_out, tail_out = yield (new_heads, new_tails), {}
  out_flat, tree_def = tree_flatten((head_out, tail_out))
  yield out_flat, tree_def


@lu.transformation
def doubling_transform(*args):
  with core.new_master(DoublingTrace) as master:
    trace = DoublingTrace(master, core.cur_sublevel())
    in_tracers = [DoublingTracer(trace, head, tail) for head, tail in args]
    outputs = yield in_tracers, {}
    if isinstance(outputs, Sequence):
      out_tracers = map(trace.full_raise, outputs)
      result = [(x.head, x.tail) for x in out_tracers]
    else:
      out_tracer = trace.full_raise(outputs)
      result = (out_tracer.head, out_tracer.tail)
  yield result


def doubledouble(f):
  @wraps(f)
  def wrapped(*args):
    args_flat, in_tree = tree_flatten(args)
    f_flat, out_tree = flatten_fun_nokwargs(lu.wrap_init(f), in_tree)
    arg_pairs = [(x, jnp.zeros_like(x)) for x in args_flat]
    out_pairs_flat = doubling_transform(f_flat).call_wrapped(*arg_pairs)
    out_flat = [head + tail for head, tail in out_pairs_flat]
    out = tree_unflatten(out_tree(), out_flat)
    return out
  return wrapped


doubling_rules: Dict[core.Primitive, Callable] = {}

def _mul_const(dtype):
  _nmant = jnp.finfo(dtype).nmant
  return jnp.array((2 << (_nmant - _nmant // 2)) + 1, dtype=dtype)

def _normalize(x, y):
  z = x + y
  zz = jnp.where(
    lax.abs(x) > lax.abs(y),
    x - z + y,
    y - z + x,
  )
  return z, zz

def _abs2(x):
  x, xx = x
  sign = jnp.where(lax.sign(x) == lax.sign(xx), 1, -1)
  return (lax.abs(x), sign * lax.abs(xx))
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
  c = lax.sqrt(x)
  u, uu = _mul12(c, c)
  cc = (x - u - uu + xx) * 0.5 / c
  y = c + cc
  yy = c - y + cc
  return y, yy
doubling_rules[lax.sqrt_p] = _sqrt2


def _def_inequality(prim, op):
  def transformed(x, y):
    z, zz = _sub2(x, y)
    return op(z + zz, 0), None
  doubling_rules[prim] = transformed

_def_inequality(lax.gt_p, operator.gt)
_def_inequality(lax.ge_p, operator.ge)
_def_inequality(lax.lt_p, operator.lt)
_def_inequality(lax.le_p, operator.le)
_def_inequality(lax.eq_p, operator.eq)
_def_inequality(lax.ne_p, operator.ne)

def _convert_element_type(operand, new_dtype, old_dtype):
  head, tail = operand
  head = lax.convert_element_type_p.bind(head, new_dtype=new_dtype, old_dtype=old_dtype)
  if tail is not None:
    tail = lax.convert_element_type_p.bind(tail, new_dtype=new_dtype, old_dtype=old_dtype)
  if jnp.issubdtype(new_dtype, jnp.floating):
    if tail is None:
      tail = jnp.zeros_like(head)
  elif tail is not None:
    head = head + tail
    tail = None
  return (head, tail)
doubling_rules[lax.convert_element_type_p] = _convert_element_type

def _add_jaxvals(xs, ys):
  # return ad_util.jaxval_adders[type(xs[0])](xs, ys)
  return _add2(xs, ys)
doubling_rules[ad_util.add_jaxvals_p] = _add_jaxvals

def _def_passthrough(prim, argnums=(0,)):
  def transformed(*args, **kwargs):
    return (
      prim.bind(*(arg[0] if i in argnums else arg for i, arg in enumerate(args)), **kwargs),
      prim.bind(*(arg[1] if i in argnums else arg for i, arg in enumerate(args)), **kwargs)
    )
  doubling_rules[prim] = transformed

_def_passthrough(lax.select_p, (0, 1, 2))
_def_passthrough(lax.broadcast_in_dim_p)
_def_passthrough(xla.device_put_p)
_def_passthrough(lax.tie_in_p, (0, 1))


class DoubleDouble:
  """DoubleDouble class with overloaded operators."""
  __slots__ = ["head", "tail"]

  def __init__(self, val, dtype=None):
    if isinstance(val, tuple):
      head, tail = val
    elif isinstance(val, str):
      raise NotImplementedError('string input')
    elif isinstance(val, int):
      dtype = jnp.dtype(dtype or 'float64').type
      head = jnp.array(val, dtype=dtype)
      tail = jnp.array(val - int(head), dtype=dtype)
    elif isinstance(val, DoubleDouble):
      head, tail = val.head, val.tail
    else:
      head, tail = val, jnp.zeros_like(val)
    dtype = dtype or jnp.result_type(head, tail)
    head = jnp.asarray(head, dtype=dtype)
    tail = jnp.asarray(tail, dtype=dtype)
    self.head, self.tail = _normalize(head, tail)

  def normalize(self):
    """Return a normalized copy of self."""
    return self._wrap(_normalize(self.head, self.tail))

  @property
  def dtype(self):
    return self.head.dtype

  def to_array(self, dtype=None):
    head, tail = self._tup
    if dtype is not None:
      head = head.astype(dtype)
      tail = tail.astype(dtype)
    return head + tail

  def __repr__(self):
    return f"{self.__class__.__name__}({self.head}, {self.tail})"

  @property
  def _tup(self):
    return self.head, self.tail

  def _wrap(self, other):
    return self.__class__(other, dtype=self.dtype)

  def __abs__(self):
    return self._wrap(_abs2(self._tup))

  def __neg__(self):
    return self._wrap(_neg2(self._tup))

  def __add__(self, other):
    return self._wrap(_add2(self._tup, self._wrap(other)._tup))

  def __sub__(self, other):
    return self._wrap(_sub2(self._tup, self._wrap(other)._tup))

  def __mul__(self, other):
    return self._wrap(_mul2(self._tup, self._wrap(other)._tup))

  def __truediv__(self, other):
    return self._wrap(_div2(self._tup, self._wrap(other)._tup))

  def __radd__(self, other):
    return self._wrap(_add2(self._wrap(other)._tup, self._tup))

  def __rsub__(self, other):
    return self._wrap(_sub2(self._wrap(other)._tup, self._tup))

  def __rmul__(self, other):
    return self._wrap(_mul2(self._wrap(other)._tup, self._tup))

  def __rtruediv__(self, other):
    return self._wrap(_div2(self._wrap(other)._tup, self._tup))

  def __lt__(self, other):
    return (self - other).to_array() < 0

  def __le__(self, other):
    return (self - other).to_array() <= 0

  def __gt__(self, other):
    return (self - other).to_array() > 0

  def __ge__(self, other):
    return (self - other).to_array() >= 0

  def __eq__(self, other):
    return (self - other).to_array() == 0

  def __ne__(self, other):
    return (self - other).to_array() != 0