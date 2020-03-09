# Copyright 2019 Google LLC
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


from contextlib import contextmanager
from collections import defaultdict, Counter, namedtuple
import functools
from functools import partial, wraps
import itertools as it
import operator as op
import string

import numpy as onp

from .. import abstract_arrays
from .. import core
from ..core import Trace, Tracer
from ..util import unzip2, safe_map, safe_zip, curry
from ..abstract_arrays import ShapedArray
from .. import linear_util as lu

map = safe_map
zip = safe_zip

def prod(xs):
  xs = list(xs)
  return functools.reduce(op.mul, xs) if xs else 1


### main transformation functions

ShapeEnvs = namedtuple("ShapeEnvs", ["logical", "padded"])
shape_envs = ShapeEnvs({}, {})  # TODO(mattjj): make this a stack for efficiency

@contextmanager
def extend_shape_envs(logical_env, padded_env):
  global shape_envs
  new_logical = dict(it.chain(shape_envs.logical.items(), logical_env.items()))
  new_padded = dict(it.chain(shape_envs.padded.items(), padded_env.items()))
  shape_envs, prev = ShapeEnvs(new_logical, new_padded), shape_envs
  yield
  shape_envs = prev

def is_polymorphic(shape):
  return any(map(lambda d: isinstance(d, Poly), shape))

def shape_as_value(expr):
  if type(expr) is tuple and is_polymorphic(expr):
    return tuple(eval_dim_expr(shape_envs.logical, d) if type(d) is Poly else d
                 for d in expr)
  else:
    return expr

def padded_shape_as_value(expr):
  if type(expr) is tuple and is_polymorphic(expr):
    return tuple(eval_dim_expr(shape_envs.padded, d) if type(d) is Poly else d
                 for d in expr)
  else:
    return expr


def mask_fun(fun, logical_env, padded_env, in_vals, shape_exprs):
  with core.new_master(MaskTrace) as master:
    fun, out_shapes = mask_subtrace(fun, master)
    with extend_shape_envs(logical_env, padded_env):
      out_vals = fun.call_wrapped(in_vals, shape_exprs)
    del master
  return out_vals, out_shapes()

@lu.transformation_with_aux
def mask_subtrace(master, in_vals, shape_exprs):
  trace = MaskTrace(master, core.cur_sublevel())
  in_tracers = [MaskTracer(trace, x, s).full_lower()
                for x, s in zip(in_vals, shape_exprs)]
  outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  out_vals, out_shapes = unzip2((t.val, t.shape_expr) for t in out_tracers)
  yield out_vals, out_shapes

def ensure_poly(p):
  if isinstance(p, Poly):
    return p

  return constant_poly(int(p))

class Poly(Counter):
  """Polynomial with integer coefficients,
  usable as element in a polymorphic shape.

  type Poly = Map Mon Int -- monomials to coeffs
  type Mon = Map Str Int
  """
  def __init__(self, coeffs):
    # Makes sure Polynomials are always in canonical form to simplify operators:
    coeffs = {mon: coeff for mon, coeff in coeffs.items() if coeff != 0}
    coeffs = {Mon(): 0} if len(coeffs) == 0 else coeffs
    super().__init__(coeffs)

  def __add__(self, other):
    coeffs = self.copy()

    for mon, coeff in ensure_poly(other).items():
      coeffs[mon] = coeffs.get(mon, 0) + coeff

    return Poly(coeffs)

  def __sub__(self, other):
    return self + -other

  def __neg__(self):
    return Poly({mon: -coeff for mon, coeff in self.items()})

  def __mul__(self, other):
    coeffs = dict()
    for (mon1, coeff1), (mon2, coeff2) \
            in it.product(self.items(), ensure_poly(other).items()):
      mon = Mon(mon1 + mon2)                        # add monomials' id degrees
      coeff = coeff1 * coeff2                       # multiply integer coeffs
      coeffs[mon] = coeffs.get(mon, 0) + coeff  # accumulate coeffs

    return Poly(coeffs)

  def __rmul__(self, other):
    return self * other

  def __radd__(self, other):
    return self + other

  def __rsub__(self, other):
    return self + -other

  def __floordiv__(self, divisor):
    q, _ = divmod(self, divisor)  # pytype: disable=wrong-arg-types
    return q

  def __mod__(self, divisor):
    _, r = divmod(self, divisor)  # pytype: disable=wrong-arg-types
    return r

  def __divmod__(self, divisor):
    if self.is_constant:
      q, r = divmod(int(self), divisor)

      return constant_poly(q), r

    def divided(count):
      q, r = divmod(count, divisor)
      if r != 0:
        raise ValueError('shapecheck currently only supports strides '
                         'that exactly divide the strided axis length.')
      return q

    return Poly(
      {k: coeff // divisor if k.degree == 0 else divided(coeff)
      for k, coeff in self.items()}), self[Mon()] % divisor

  def __hash__(self):
    return hash(super())

  def __eq__(self, other):
    return super().__eq__(ensure_poly(other))

  def __ne__(self, other):
    return not self == other

  def __ge__(self, other):
    other = ensure_poly(other)

    if other.is_constant and self.is_constant:
      return int(self) >= int(other)

    if other.is_constant and int(other) <= 1:
        # Assume polynomials > 0, allowing to use shape rules of binops, conv:
        return True

    if self.is_constant and int(self) <= 0:
      return False # See above.

    if self == other:
      return True

    raise ValueError('Polynomials comparison "{} >= {}" is inconclusive.'
                     .format(self, other))

  def __le__(self, other):
    return ensure_poly(other) >= self

  def __lt__(self, other):
    return not (self >= other)

  def __gt__(self, other):
    return not (ensure_poly(other) >= self)

  def __str__(self):
    return ' + '.join('{} {}'.format(v, k) if (v != 1 or k.degree == 0) else str(k)
                      for k, v in sorted(self.items())).strip()

  def __int__(self):
    assert self.is_constant

    return int(next(iter(self.values())))

  @property
  def is_constant(self):
    return len(self) == 1 and next(iter(self)).degree == 0

abstract_arrays._DIMENSION_TYPES.add(Poly)


class Mon(Counter):  # type Mon = Map Id Int -- ids to degrees
  def __hash__(self):
    return hash(tuple(self.items()))

  def __str__(self):
    return ' '.join('{}**{}'.format(k, v) if v != 1 else str(k)
                    for k, v in sorted(self.items()))

  def __lt__(self, other):
    # sort by total degree, then lexicographically on indets
    self_key = self.degree, tuple(sorted(self))
    other_key = other.degree, tuple(sorted(other))
    return self_key < other_key

  @property
  def degree(self):
    return sum(self.values())

def eval_shape_expr(env, expr):
  return tuple(eval_dim_expr(env, poly) for poly in expr)

def eval_dim_expr(env, poly):
  terms = [mul(coeff, prod([pow(env[id], deg) for id, deg in mon.items()]))
           for mon, coeff in poly.items()]
  return sum(terms) if len(terms) > 1 else terms[0]

def pow(x, deg):
  try:
    deg = int(deg)
  except:
    return x ** deg
  else:
    return 1 if deg == 0 else x if deg == 1 else x ** deg

def mul(coeff, mon):
  try:
    coeff = int(coeff)
  except:
    return coeff * mon
  else:
    return  0 if coeff == 0 else mon if coeff == 1 else coeff * mon

class ShapeError(Exception): pass

class ShapeSyntaxError(Exception): pass

# To denote some shape expressions (for annotations) we use a small language.
#
#   data ShapeSpec = ShapeSpec [Dim]
#   data Dim = Id PyObj
#            | Lit Int
#            | Mul Dim Dim
#            | Add Dim Dim
#            | MonomorphicDim
#
# We'll also make a simple concrete syntax for annotation. The grammar is
#
#   shape_spec ::= '(' dims ')'
#   dims       ::= dim ',' dims | ''
#   dim        ::= str | int | dim '*' dim | dim '+' dim | '_'
#
# ShapeSpecs encode ShapeExprs but can have some monomorphic dims inside them,
# which must be replaced with concrete shapes when known.

class ShapeSpec(tuple):
  def __str__(self):
    return 'ShapeSpec({})'.format(', '.join(map(str, self)))

def finalize_spec(spec, shape):
  return tuple(parse_lit(d) if e is monomorphic_dim else e
               for e, d in zip(spec, shape))

def parse_spec(spec=''):
  if not spec:
    return ShapeSpec(())
  if spec[0] == '(':
    if spec[-1] != ')': raise ShapeSyntaxError(spec)
    spec = spec[1:-1]
  dims = map(parse_dim, spec.replace(' ', '').strip(',').split(','))
  return ShapeSpec(dims)

def parse_dim(spec):
  if '+' in spec:
    terms = map(parse_dim, spec.split('+'))
    return functools.reduce(op.add, terms)
  elif '*' in spec:
    terms = map(parse_dim, spec.split('*'))
    return functools.reduce(op.mul, terms)
  elif spec.isdigit() or spec.startswith('-') and spec[1:].isdigit():
    return parse_lit(spec)
  elif spec in identifiers:
    return parse_id(spec)
  elif spec == '_':
    return monomorphic_dim
  else:
    raise ShapeSyntaxError(spec)
digits = frozenset(string.digits)
identifiers = frozenset(string.ascii_lowercase)

def parse_id(name): return Poly({Mon({name: 1}): 1})
def parse_lit(val_str): return constant_poly(int(val_str))
def constant_poly(val): return Poly({Mon(): val})

class MonomorphicDim(object):
  def __str__(self): return '_'
monomorphic_dim = MonomorphicDim()


# Two convenient ways to provide shape annotations:
#   1. '(m, n)'
#   2. s_['m', 'n']

class S_(object):
  def __getitem__(self, idx):
    if type(idx) is tuple:
      return parse_spec('(' + ','.join(map(str, idx)) + ')')
    else:
      return parse_spec(str(idx))
s_ = S_()


### automasking tracer machinery

class MaskTracer(Tracer):
  __slots__ = ["val", "shape_expr"]

  def __init__(self, trace, val, shape_expr):
    self._trace = trace
    self.val = val
    self.shape_expr = shape_expr

  @property
  def aval(self):
    return ShapedArray(self.shape_expr, self.val.dtype)

  def is_pure(self):
    return all(ensure_poly(poly).is_constant for poly in self.shape_expr)

  def full_lower(self):
    if self.is_pure():
      return core.full_lower(self.val)
    else:
      return self

class MaskTrace(Trace):
  def pure(self, val):
    return MaskTracer(self, val, onp.shape(val))

  def lift(self, val):
    return MaskTracer(self, val, onp.shape(val))

  def sublift(self, val):
    return MaskTracer(self, val.val, val.shape_expr)

  def process_primitive(self, primitive, tracers, params):
    vals, shape_exprs = unzip2((t.val, t.shape_expr) for t in tracers)
    if primitive in shape_parameterized_primitive_rules:
      rule = shape_parameterized_primitive_rules[primitive]
      out, out_shape = rule(shape_envs, vals, shape_exprs, **params)
    else:
      out_shape = shape_rules[primitive](*(t.aval for t in tracers), **params)
      logical_shapes = map(partial(eval_shape_expr, shape_envs.logical), shape_exprs)
      out = masking_rules[primitive](vals, logical_shapes, **params)
    if not primitive.multiple_results:
      return MaskTracer(self, out, out_shape)
    else:
      return map(partial(MaskTracer, self), out, out_shape)

  def process_call(self, call_primitive, f: lu.WrappedFun, tracers, params):
    raise NotImplementedError  # TODO mask-of-jit

shape_parameterized_primitive_rules = {}
masking_rules = {}
shape_rules = {}

def defvectorized(prim):
  masking_rules[prim] = partial(vectorized_masking_rule, prim)

def vectorized_masking_rule(prim, padded_vals, logical_shapes, **params):
  del logical_shapes  # Unused.
  padded_val, = padded_vals
  return prim.bind(padded_val, **params)


def defnaryop(prim):
  masking_rules[prim] = partial(naryop_masking_rule, prim)

def naryop_masking_rule(prim, padded_vals, logical_shapes):
  del logical_shapes  # Unused.
  return prim.bind(*padded_vals)


### definition-time (import-time) shape checker tracer machinery

def shapecheck(fun: lu.WrappedFun, in_shapes):
  with core.new_master(ShapeCheckTrace) as master:
    out_shapes = check_subtrace(fun, master).call_wrapped(in_shapes)
    del master
  return out_shapes

@lu.transformation
def check_subtrace(master, in_shapes):
  trace = ShapeCheckTrace(master, core.cur_sublevel())
  in_tracers = map(partial(ShapeCheckTracer, trace), in_shapes)
  outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  yield [t.shape_expr for t in out_tracers]


# TODO(mattjj): add dtypes?
class ShapeCheckTracer(Tracer):
  __slots__ = ["shape_expr"]

  def __init__(self, trace, shape_expr):
    self._trace = trace
    self.shape_expr = shape_expr

  @property
  def aval(self):
    return ShapedArray(self.shape_expr, None)

  def full_lower(self):
    return self

class ShapeCheckTrace(Trace):
  def pure(self, val):
    return ShapeCheckTracer(self, onp.shape(val))

  def lift(self, val):
    return ShapeCheckTracer(self, onp.shape(val))

  def sublift(self, val):
    return ShapeCheckTracer(self, val.shape_expr)

  def process_primitive(self, primitive, tracers, params):
    avals = [t.aval for t in tracers]
    shape_rule = shape_rules.get(primitive)
    if shape_rule is None:
      raise NotImplementedError('Shape rule for {} not implemented yet.'.format(primitive))
    out_shape = shape_rule(*avals, **params)
    return ShapeCheckTracer(self, out_shape)

  def process_call(self, call_primitive, f: lu.WrappedFun, tracers, params):
    # TODO apply proper subtrace:
    return map(self.full_raise, f.call_wrapped(*tracers))

