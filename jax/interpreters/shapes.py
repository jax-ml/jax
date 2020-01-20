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

from __future__ import print_function

import operator as op
import string
from collections import Counter
from functools import partial, reduce
from itertools import product

import numpy as onp

from .. import core
from .. import linear_util as lu
from ..abstract_arrays import ShapedArray
from ..core import Trace, Tracer
from ..util import safe_map, safe_zip

map = safe_map
zip = safe_zip

def prod(xs):
  xs = list(xs)
  return reduce(op.mul, xs) if xs else 1

def to_index(x):
  """Like operator.index, but allowing polymorphic dimensions.
  Not implemented as `Poly.__index__`, since operator.index only allows ints."""
  return x if isinstance(x, Poly) else op.index(x)

# TODO remove remaining usages:
def is_polymorphic(shape):
  return any(map(lambda d: type(d) is Poly, shape))

def eval_polymorphic_shape(shape, values_dict):
  return tuple(dim.evaluate(values_dict) if type(dim) is Poly else dim
               for dim in shape)

def _ensure_poly(p):
  if type(p) is Poly:
    return p

  return _constant_poly(p)

class Poly(dict):
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

    for mon, coeff in _ensure_poly(other).items():
      coeffs[mon] = coeffs.get(mon, 0) + coeff

    return Poly(coeffs)

  def __sub__(self, other):
    return self + -other

  def __neg__(self):
    return Poly({mon: -coeff for mon, coeff in self.items()})

  def __mul__(self, other):
    coeffs = dict()
    for (mon1, coeff1), (mon2, coeff2) \
            in product(self.items(), _ensure_poly(other).items()):
      mon = mon1 * mon2
      coeffs[mon] = coeffs.get(mon, 0) + coeff1 * coeff2

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
      return divmod(int(self), divisor)

    def divided(count):
      q, r = divmod(count, divisor)
      if r != 0:
        raise ValueError('shapecheck currently only supports strides '
                         'that exactly divide the strided axis length.')
      return q

    return Poly(
      {k: coeff // divisor if k.degree == 0 else divided(coeff)
       for k, coeff in self.items()}), self.get(Mon(), 0) % divisor

  def __hash__(self):
    return hash(super())

  def __eq__(self, other):
    return super().__eq__(_ensure_poly(other))

  def __ne__(self, other):
    return not self == other

  def __ge__(self, other):
    other = _ensure_poly(other)

    if other.is_constant and self.is_constant:
      return int(self) >= int(other)

    if other.is_constant and int(other) <= 1:
      # Assume polynomials > 0, allowing to use shape rules of binops, conv:
      return True

    if self.is_constant and int(self) <= 0:
      return False  # See above.

    if self == other:
      return True

    raise ValueError('Polynomials comparison "{} >= {}" is inconclusive.'
                     .format(self, other))

  def __le__(self, other):
    return _ensure_poly(other) >= self

  def __lt__(self, other):
    return not (self >= other)

  def __gt__(self, other):
    return not (_ensure_poly(other) >= self)

  def __str__(self):
    return ' + '.join('{} {}'.format(v, k)
                      if (v != 1 or k.degree == 0) else str(k)
                      for k, v in sorted(self.items())).strip()

  def __int__(self):
    assert self.is_constant

    return int(next(iter(self.values())))

  def evaluate(self, values_dict):
    return sum(coeff * prod([values_dict[id] ** deg for id, deg in mon.items()])
               for mon, coeff in self.items())

  @property
  def is_constant(self):
    return len(self) == 1 and next(iter(self)).degree == 0

class Mon(dict):  # type Mon = Map Id Int -- ids to degrees
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

  def __mul__(self, other):
    return Mon(Counter(self) + Counter(other))

  @property
  def degree(self):
    return sum(self.values())

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
  return tuple(_parse_lit(d) if e is monomorphic_dim else e
               for e, d in zip(spec, shape))

def parse_spec(spec=''):
  if not spec:
    return ShapeSpec(())
  if spec[0] == '(':
    if spec[-1] != ')': raise ShapeSyntaxError(spec)
    spec = spec[1:-1]
  dims = map(_parse_dim, spec.replace(' ', '').strip(',').split(','))
  return ShapeSpec(dims)

def _parse_dim(spec):
  if '+' in spec:
    terms = map(_parse_dim, spec.split('+'))
    return reduce(op.add, terms)
  elif '*' in spec:
    terms = map(_parse_dim, spec.split('*'))
    return reduce(op.mul, terms)
  elif spec.isdigit() or spec.startswith('-') and spec[1:].isdigit():
    return _parse_lit(spec)
  elif spec in identifiers:
    return _parse_id(spec)
  elif spec == '_':
    return monomorphic_dim
  else:
    raise ShapeSyntaxError(spec)

digits = frozenset(string.digits)
identifiers = frozenset(string.ascii_lowercase)

def _parse_id(name): return Poly({Mon({name: 1}): 1})

def _parse_lit(val_str): return _constant_poly(int(val_str))

def _constant_poly(val): return Poly({Mon(): op.index(val)})

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

shape_rules = {}

### definition-time (import-time) shape checker tracer machinery

def shapecheck(fun, in_shapes):
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
    self.trace = trace
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
    if shape_rule is None: raise NotImplementedError(
      'Shape rule for {} not implemented yet.'.format(primitive))
    out_shape = shape_rule(*avals, **params)

    if primitive.multiple_results:
      return map(partial(ShapeCheckTracer, self), out_shape)
    else:
      return ShapeCheckTracer(self, out_shape)

  def process_call(self, call_primitive, f, tracers, params):
    # TODO apply proper subtrace:
    return map(self.full_raise, f.call_wrapped(*tracers))
