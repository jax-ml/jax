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

from collections import defaultdict, Counter, namedtuple
from functools import partial, wraps
import itertools as it
import operator as op
import string

import numpy as onp
import six

from .. import core
from ..core import Trace, Tracer
from ..util import unzip2, safe_map, safe_zip, curry
from ..abstract_arrays import ShapedArray
from .. import linear_util as lu
from . import partial_eval as pe

map = safe_map
zip = safe_zip
reduce = six.moves.reduce

def prod(xs):
  xs = list(xs)
  return reduce(op.mul, xs) if xs else 1


### main transformation functions

def mask_fun(fun, shape_envs, in_vals, shape_exprs):
  with core.new_master(MaskTrace) as master:
    fun, out_shapes = mask_subtrace(fun, master, shape_envs)
    out_vals = fun.call_wrapped(in_vals, shape_exprs)
    del master
  return out_vals, out_shapes()

@lu.transformation_with_aux
def mask_subtrace(master, shape_envs, in_vals, shape_exprs):
  trace = MaskTrace(master, core.cur_sublevel())
  in_tracers = map(partial(MaskTracer, trace, shape_envs),
                   in_vals, shape_exprs)
  outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  out_vals, out_shapes = unzip2((t.val, t.shape_expr) for t in out_tracers)
  yield out_vals, out_shapes


### shape expressions

# Shape expressions model tuples of formal polynomials with integer
# coefficients. Here are the internal data structures we use to represent them.
#
#   type ShapeExpr = [Poly]
#   type Poly = Map Mon Int
#   type Mon = Map Str Int

class ShapeExpr(tuple):  # type ShapeExpr = [Poly]
  def __str__(self):
    return 'ShapeExpr({})'.format(', '.join(map(str, self)))

class Poly(Counter):  # type Poly = Map Mon Int -- monomials to coeffs
  def __mul__(p1, p2):
    new_poly = Poly()
    for (mon1, coeff1), (mon2, coeff2) in it.product(p1.items(), p2.items()):
      mon = Mon(mon1 + mon2)                        # add monomials' id degrees
      coeff = coeff1 * coeff2                       # multiply integer coeffs
      new_poly[mon] = new_poly.get(mon, 0) + coeff  # accumulate coeffs
    return new_poly

  def __add__(p1, p2):
    return Poly(Counter.__add__(p1, p2))

  def __hash__(self):
    return hash(tuple(self.items()))

  def __str__(self):
    return ' + '.join('{} {}'.format(v, k) if v != 1 else str(k)
                      for k, v in sorted(self.items())).strip()

class Mon(Counter):  # type Mon = Map Id Int -- ids to degrees
  def __hash__(self):
    return hash(tuple(self.items()))

  def __str__(self):
    return ' '.join('{}**{}'.format(k, v) if v != 1 else str(k)
                    for k, v in sorted(self.items()))

  def __lt__(self, other):
    # sort by total degree, then lexicographically on indets
    self_key = sum(self.values()), tuple(sorted(self))
    other_key = sum(other.values()), tuple(sorted(other))
    return self_key < other_key

def eval_shape_expr(env, expr):
  return tuple(eval_dim_expr(env, poly) for poly in expr)

def eval_dim_expr(env, poly):
  return sum(coeff * prod([env[id] ** deg for id, deg in mon.items()])
             for mon, coeff in poly.items())

class ShapeError(Exception): pass

# To denote some shape expressions (for annotations) we use a small language.
#
#   data Shape = Shape [Dim]
#   data Dim = Id Str
#            | Lit Int
#            | Mul Dim Dim
#            | Add Dim Dim

# We'll also make a simple concrete syntax for annotation. The grammar is
#
#   shape_spec ::= '(' dims ')'
#   dims       ::= dim ',' dims | ''
#   dim        ::= str | int | dim '*' dim | dim '+' dim

def parse_spec(spec=''):
  if not spec:
    return ShapeExpr(())
  if spec[0] == '(':
    if spec[-1] != ')': raise SyntaxError(spec)
    spec = spec[1:-1]
  dims = map(parse_dim, spec.replace(' ', '').strip(',').split(','))
  return ShapeExpr(dims)

def parse_dim(spec):
  if '+' in spec:
    terms = map(parse_dim, spec.split('+'))
    return reduce(op.add, terms)
  elif '*' in spec:
    terms = map(parse_dim, spec.split('*'))
    return reduce(op.mul, terms)
  elif spec in digits:
    return parse_lit(spec)
  elif spec in identifiers:
    return parse_id(spec)
  else:
    raise SyntaxError(spec)
digits = frozenset(string.digits)
identifiers = frozenset(string.ascii_lowercase)

def parse_id(name): return Poly({Mon({name: 1}): 1})
def parse_lit(val_str): return Poly({Mon(): int(val_str)})


# Two convenient ways to provide shape annotations:
#   1. Shape('(m, n)')
#   2. s_['m', 'n']

Shape = parse_spec

class S_(object):
  def __getitem__(self, idx):
    if type(idx) is tuple:
      return parse_spec('(' + ','.join(map(str, idx)) + ')')
    else:
      return parse_spec(str(idx))
s_ = S_()


### automasking tracer machinery

ShapeEnvs = namedtuple("ShapeEnvs", ["logical", "padded"])

class MaskTracer(Tracer):
  __slots__ = ["val", "shape_expr", "shape_envs"]

  def __init__(self, trace, shape_envs, val, shape_expr):
    self.trace = trace
    self.shape_envs = shape_envs
    self.val = val
    self.shape_expr = shape_expr

  @property
  def aval(self):
    return ShapedArray(self.shape_expr, self.val.dtype)

  def full_lower(self):
    if all(type(s) is int for s in self.shape_expr):
      return core.full_lower(self.val)
    else:
      return self

class MaskTrace(Trace):
  def pure(self, val):
    return MaskTracer(self, None, val, ShapeExpr(*onp.shape(val)))

  def lift(self, val):
    return MaskTracer(self, None, val, ShapeExpr(*onp.shape(val)))

  def sublift(self, val):
    return MaskTracer(self, val.shape_envs, val.val, val.shape_expr)

  def process_primitive(self, primitive, tracers, params):
    shape_envs = next(t.shape_envs for t in tracers if t.shape_envs is not None)
    vals, shape_exprs = unzip2((t.val, t.shape_expr) for t in tracers)
    if primitive in shape_parameterized_primitive_rules:
      rule = shape_parameterized_primitive_rules[primitive]
      out, out_shape = rule(shape_envs, vals, shape_exprs, **params)
    else:
      out_shape = shape_rules[primitive](shape_exprs, **params)
      logical_shapes = map(partial(eval_shape_expr, shape_envs.logical), shape_exprs)
      out = masking_rules[primitive](vals, logical_shapes, **params)
    if not primitive.multiple_results:
      return MaskTracer(self, shape_envs, out, out_shape)
    else:
      return map(partial(MaskTracer, self, shape_envs), out, out_shape)

  def process_call(self, call_primitive, f, tracers, params):
    raise NotImplementedError  # TODO mask-of-jit

shape_parameterized_primitive_rules = {}
masking_rules = {}
shape_rules = {}

def defvectorized(prim):
  shape_rules[prim] = vectorized_shape_rule
  masking_rules[prim] = partial(vectorized_masking_rule, prim)

def vectorized_shape_rule(shape_exprs, **unused_params):
  shape_expr, = shape_exprs
  return shape_expr

def vectorized_masking_rule(prim, padded_vals, logical_shapes):
  del logical_shapes  # Unused.
  padded_val, = padded_vals
  return prim.bind(padded_val)


def defbinop(prim):
  shape_rules[prim] = binop_shape_rule
  masking_rules[prim] = partial(binop_masking_rule, prim)

def binop_shape_rule(shape_exprs):
  x_shape_expr, y_shape_expr = shape_exprs
  if not x_shape_expr == y_shape_expr: raise ShapeError
  return x_shape_expr

def binop_masking_rule(prim, padded_vals, logical_shapes):
  del logical_shapes  # Unused.
  padded_x, padded_y = padded_vals
  return prim.bind(padded_x, padded_y)


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
    return ShapeCheckTracer(self, Shape(*onp.shape(val)), onp.result_type(val))

  def lift(self, val):
    return ShapeCheckTracer(self, Shape(*onp.shape(val)), onp.result_type(val))

  def sublift(self, val):
    return ShapeCheckTracer(self, val.shape_expr)

  def process_primitive(self, primitive, tracers, params):
    shape_exprs = [t.shape_expr for t in tracers]
    out_shape_expr = shape_rules[primitive](shape_exprs, **params)
    return ShapeCheckTracer(self, out_shape_expr)

  def process_call(self, call_primitive, f, tracers, params):
    raise NotImplementedError  # TODO check-of-jit
