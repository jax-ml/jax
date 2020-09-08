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
from collections import Counter, namedtuple
from functools import partial, reduce
from itertools import chain, product
import operator as op
import string
from typing import Callable, Dict, Sequence, Union, Any

import numpy as np

from .. import abstract_arrays
from .. import core, dtypes
from ..tree_util import tree_unflatten
from ..core import Trace, Tracer
from ..util import safe_map, safe_zip, unzip2, prod, wrap_name
from ..abstract_arrays import ShapedArray
from .. import linear_util as lu

map = safe_map
zip = safe_zip
def identity(x): return x


### Shape environments (global state)

ShapeEnvs = namedtuple("ShapeEnvs", ["logical", "padded"])
shape_envs = ShapeEnvs({}, {})  # TODO(mattjj): make this a stack for efficiency

@contextmanager
def extend_shape_envs(logical_env, padded_env):
  global shape_envs
  new_logical = dict(chain(shape_envs.logical.items(), logical_env.items()))
  new_padded = dict(chain(shape_envs.padded.items(), padded_env.items()))
  shape_envs, prev = ShapeEnvs(new_logical, new_padded), shape_envs
  try:
    yield
  finally:
    shape_envs = prev

def eval_polymorphic_shape(env: Dict[Any, int],
                           shape: Sequence[Union[int, 'Poly']]):
  return tuple(dim if type(dim) is int else eval_poly(env, dim)
               for dim in shape)

def shape_as_value(shape: Sequence[Union[int, 'Poly']]):
  return eval_polymorphic_shape(shape_envs.logical, shape)

def padded_shape_as_value(shape: Sequence[Union[int, 'Poly']]):
  return eval_polymorphic_shape(shape_envs.padded, shape)


### Transforms

def mask_fun(fun, logical_env, padded_env, in_vals, polymorphic_shapes):
  with core.new_main(MaskTrace) as main:
    fun, out_shapes = mask_subtrace(fun, main)
    with extend_shape_envs(logical_env, padded_env):
      out_vals = fun.call_wrapped(in_vals, polymorphic_shapes)
    del main
  out_shapes = [x.aval.shape if type(x) is NotPolymorphic else x
                for x in out_shapes()]
  return out_vals, out_shapes

@lu.transformation_with_aux
def mask_subtrace(main, in_vals, polymorphic_shapes):
  trace = MaskTrace(main, core.cur_sublevel())
  in_tracers = [MaskTracer(trace, x, s).full_lower()
                for x, s in zip(in_vals, polymorphic_shapes)]
  outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  out_vals, out_shapes = unzip2((t.val, t.polymorphic_shape) for t in out_tracers)
  yield out_vals, out_shapes


### Polymorphic shape representation

class Mon:
  def __init__(self, components: Dict[Any, int]):
    self.components = {indet: deg for indet, deg in sorted(components.items())
                       if deg != 0}

  def __mul__(self, other: 'Mon'):
    return Mon({v: self.components.get(v, 0) + other.components.get(v, 0)
                for v in chain(self.components, other.components)})

  def __str__(self):
    return ' '.join('{}**{}'.format(v, c) if c != 1 else str(v)
                    for v, c in self.components.items())

  def __hash__(self):
    return hash(tuple(self.components.items()))

  def __eq__(self, other):
    return type(other) is Mon and self.components == other.components

  def __lt__(self, other: 'Mon'):
    # sort by total degree, then lexicographically on indets
    self_key = -sum(self.components.values()), tuple(self.components)
    other_key = -sum(other.components.values()), tuple(other.components)
    return self_key < other_key


class Poly:
  """Polynomial with nonnegative integer coefficients for polymorphic shapes."""

  def __init__(self, terms: Dict[Mon, int]):
    self.terms = {mon: coeff for mon, coeff in sorted(terms.items())
                  if coeff != 0}

  def __add__(self, other: 'Poly') -> 'Poly':
    terms = self.terms.copy()
    for mon, coeff in other.terms.items():
      terms[mon] = terms.get(mon, 0) + coeff
    return Poly(terms)

  def __mul__(self, other: 'Poly') -> 'Poly':
    terms = {}
    for (mon1, c1), (mon2, c2) in product(self.terms.items(), other.terms.items()):
      mon = mon1 * mon2
      terms[mon] = terms.get(mon, 0) + c1 * c2
    return Poly(terms)

  def __eq__(self, other) -> bool:
    if type(other) is Poly:
      return self.terms == other.terms
    elif type(other) is int:
      return self.terms.get(Mon({}), 0) == other
    else:
      return False

  # hashing is convenient to de-duplicate a collection of Polys, and also to
  # cache results of broadcasting
  def __hash__(self) -> int:
    return hash(tuple(self.terms.items()))

  # comparing against int zero is convenient for checks, otherwise error
  def __ge__(self, other):
    if other == 0:
      return True
    else:
      raise TypeError(f"'>=' not supported between Poly and {type(other)}")

  def __lt__(self, other):
    if other == 0:
      return False
    else:
      raise TypeError(f"'<' not supported between Poly and {type(other)}")

  def __str__(self) -> str:
    out = ' + '.join('{} {}'.format(c, m) if c != 1 else str(m)
                     for m, c in self.terms.items()).strip()
    return out or '0'

  def __repr__(self) -> str:
    return f'Poly({str(self)})'
abstract_arrays._DIMENSION_TYPES.add(Poly)

def _is_constant(poly):
  try:
    mon, = poly.terms
    () = mon.components
    return True
  except (ValueError, TypeError):
    return False

def eval_poly(env, poly):
  assert isinstance(env, dict) and isinstance(poly, Poly), (env, poly)
  terms = [mul(coeff, prod([pow(env[id], deg) for id, deg in mon.components.items()]))
           for mon, coeff in poly.terms.items()]
  return sum(terms) if len(terms) > 1 else terms[0]

def prod(xs):
  return reduce(op.mul, xs) if xs else 1

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
    return 0 if coeff == 0 else mon if coeff == 1 else coeff * mon


### Shape annotation language

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
# ShapeSpecs can have some monomorphic dims inside them, which must be replaced
# with concrete shapes when known.

class ShapeSpec(tuple):
  def __str__(self):
    return 'ShapeSpec({})'.format(', '.join(map(str, self)))

def finalize_spec(polymorphic_shape, padded_shape):
  return tuple(_parse_lit(d) if e is _monomorphic_dim else e
               for e, d in zip(polymorphic_shape, padded_shape))

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
    return np.sum(map(_parse_dim, spec.split('+')))
  elif '*' in spec:
    return prod(map(_parse_dim, spec.split('*')))
  elif spec.isdigit() or spec.startswith('-') and spec[1:].isdigit():
    return _parse_lit(spec)
  elif spec in _identifiers:
    return _parse_id(spec)
  elif spec == '_':
    return _monomorphic_dim
  else:
    raise ShapeSyntaxError(spec)

_identifiers = frozenset(string.ascii_lowercase)

def _parse_id(name): return Poly({Mon({name: 1}): 1})

def _parse_lit(val_str): return Poly({Mon({}): int(val_str)})

class MonomorphicDim:
  def __str__(self): return '_'

_monomorphic_dim = MonomorphicDim()

# Two convenient ways to provide shape annotations:
#   1. '(m, n)'
#   2. s_['m', 'n']

class S_(object):
  def __getitem__(self, idx):
    return parse_spec(('(' + ','.join(map(str, idx)) + ')')
                             if type(idx) is tuple else str(idx))
s_ = S_()

def _shape_spec_consistent(spec, expr):
  return all(a == b for a, b in zip(spec, expr) if a is not _monomorphic_dim)


### Utilities for generating unique ids from user annotations

class UniqueId:
  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return self.name

  def __lt__(self, other):
    return self.name < other.name

class UniqueIds(dict):
  def __missing__(self, key):
    unique_id = UniqueId(key)
    self[key] = unique_id
    return unique_id

def remap_ids(names, shape_spec):
  return ShapeSpec(Poly({Mon({names[id] : deg for id, deg in mon.components.items()})
                         : coeff for mon, coeff in poly.terms.items()})
                   if poly is not _monomorphic_dim else
                   _monomorphic_dim for poly in shape_spec)

def bind_shapes(polymorphic_shapes, shapes):
  env = {}
  for polymorphic_shape, shape in zip(polymorphic_shapes, shapes):
    for poly, d in zip(polymorphic_shape, shape):
      if _is_constant(poly):
        continue
      else:
        mon, = poly.terms  # TODO(mattjj): generalize
        binder, = mon.components
        if env.setdefault(binder, d) != d: raise ShapeError
  return env

def check_shapes(specs, spec_tree, shapes, tree, message_prefix="Output"):
  if spec_tree != tree or not all(map(_shape_spec_consistent, specs, shapes)):
    specs = tree_unflatten(spec_tree, specs)
    shapes = tree_unflatten(tree, shapes)
    raise ShapeError(f"{message_prefix} shapes should be {specs} but are {shapes}.")


### Trace/Tracer machinery

class MaskTracer(Tracer):
  __slots__ = ["val", "polymorphic_shape"]

  def __init__(self, trace, val, polymorphic_shape):
    self._trace = trace
    self.val = val
    self.polymorphic_shape = polymorphic_shape

  @property
  def aval(self):
    if type(self.polymorphic_shape) is NotPolymorphic:
      return core.get_aval(self.val)
    else:
      assert isinstance(core.get_aval(self.val), core.ShapedArray)
      return ShapedArray(self.polymorphic_shape, self.val.dtype)

  def full_lower(self):
    if type(self.polymorphic_shape) is NotPolymorphic:
      return core.full_lower(self.val)
    else:
      return self

class NotPolymorphic:
  __slots__ = ["aval"]
  def __init__(self, aval):
    self.aval = aval

  @staticmethod
  def from_value(val):
    return NotPolymorphic(core.raise_to_shaped(core.get_aval(val)))

class MaskTrace(Trace):
  def pure(self, val):
    return MaskTracer(self, val, NotPolymorphic.from_value(val))

  def lift(self, val):
    return MaskTracer(self, val, NotPolymorphic.from_value(val))

  def sublift(self, val):
    return MaskTracer(self, val.val, val.polymorphic_shape)

  def process_primitive(self, primitive, tracers, params):
    masking_rule = masking_rules.get(primitive)
    if masking_rule is None:
      raise NotImplementedError(f'Masking rule for {primitive} not implemented.')
    vals, poly_shapes = unzip2((t.val, t.polymorphic_shape) for t in tracers)
    out, out_shape = masking_rule(vals, poly_shapes, **params)
    if primitive.multiple_results:
      return map(partial(MaskTracer, self), out, out_shape)
    else:
      return MaskTracer(self, out, out_shape)

  def process_call(self, call_primitive, f, tracers, params):
    assert False  # TODO
    assert call_primitive.multiple_results
    params = dict(params, name=wrap_name(params.get('name', f.__name__), 'mask'))
    vals, shapes = unzip2((t.val, t.polymorphic_shape) for t in tracers)
    logical_env, padded_env = shape_envs
    env_keys, padded_env_vals = unzip2(sorted(padded_env.items()))
    logical_env_vals = tuple(logical_env[k] for k in env_keys)
    # Make padded_env hashable
    padded_env = (env_keys, padded_env_vals)
    f, shapes_out = mask_subtrace(f, self.main, shapes, padded_env)
    if 'donated_invars' in params:
      params = dict(params, donated_invars=((False,) * len(logical_env_vals) +
                                            params['donated_invars']))
    vals_out = call_primitive.bind(f, *(logical_env_vals + vals), **params)
    return [MaskTracer(self, v, s) for v, s in zip(vals_out, shapes_out())]

  def post_process_call(self, call_primitive, out_tracers, params):
    vals, shapes = unzip2((t.val, t.polymorphic_shape) for t in out_tracers)
    main = self.main
    def todo(vals):
      trace = MaskTrace(main, core.cur_sublevel())
      return map(partial(MaskTracer, trace), vals, shapes)
    return vals, todo


### Rules

polymorphic_shape_rules: Dict[core.Primitive, Callable] = {}
masking_rules: Dict[core.Primitive, Callable] = {}


def defvectorized(prim):
  polymorphic_shape_rules[prim] = vectorized_shape_rule
  masking_rules[prim] = partial(vectorized_masking_rule, prim)

def vectorized_shape_rule(polymorphic_shapes, **unused_params):
  polymorphic_shape, = polymorphic_shapes
  return polymorphic_shape

def vectorized_masking_rule(prim, padded_vals, polymorphic_shapes, **params):
  logical_shape, = polymorphic_shapes
  padded_val, = padded_vals
  return prim.bind(padded_val, **params), logical_shape


def defnaryop(prim):
  polymorphic_shape_rules[prim] = naryop_shape_rule
  masking_rules[prim] = partial(naryop_masking_rule, prim)

def naryop_shape_rule(polymorphic_shapes):
  shapes = [s.aval.shape if type(s) is NotPolymorphic else s
            for s in polymorphic_shapes]
  ranks = {len(s) for s in shapes if s}
  if len(ranks) > 1:
    raise TypeError(f"got arrays of different rank: {polymorphic_shapes}")
  rank, = ranks or {()}
  if rank:
    shapes = [s or (1,) * rank for s in shapes]
    axis_sizes = zip(*shapes)
    for sizes in axis_sizes:
      sizes = [d for d in sizes if d != 1]
      if sizes[:-1] != sizes[1:]:
        raise TypeError(f"incompatible shapes for broadcasting: {polymorphic_shapes}")
    return tuple(next((d for d in sizes if d != 1), 1)
                 for sizes in zip(*shapes))
  else:
    return ()

def naryop_masking_rule(prim, padded_vals, polymorphic_shapes):
  out_shape = naryop_shape_rule(polymorphic_shapes)
  return prim.bind(*padded_vals), out_shape
