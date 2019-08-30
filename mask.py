from __future__ import print_function

from collections import defaultdict, Counter
from functools import partial
import itertools as it
import operator as op
import string

import numpy as onp
import six

from jax import core
from jax.core import Trace, Tracer
from jax.util import unzip2, safe_map, safe_zip, split_list
from jax.api_util import tree_flatten, tree_unflatten, flatten_fun_nokwargs
from jax import linear_util as lu
from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe
from jax import lax

map = safe_map
zip = safe_zip
reduce = six.moves.reduce

def prod(xs):
  xs = list(xs)
  return reduce(op.mul, xs) if xs else 1


### main transformation functions

def mask_fun(fun, shape_env, in_vals, shape_exprs):
  with core.new_master(MaskTrace) as master:
    fun, out_shapes = mask_subtrace(fun, master, shape_env)
    out_vals = fun.call_wrapped(in_vals, shape_exprs)
    del master
  return out_vals, out_shapes()

@lu.transformation_with_aux
def mask_subtrace(master, shape_env, in_vals, shape_exprs):
  trace = MaskTrace(master, core.cur_sublevel())
  in_tracers = map(partial(MaskTracer, trace, shape_env),
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

  def __str__(self):
    return ' + '.join('{} {}'.format(v, k) if v != 1 else str(k)
                      for k, v in sorted(self.items())).strip()

class Mon(Counter):  # type Mon = Map Id Int -- ids to degrees
  def __hash__(self):
    return hash(tuple(self.items()))

  def __str__(self):
    return ' '.join('{}**{}'.format(k, v) if v != 1 else str(k)
                    for k, v in sorted(self.items()))

def eval_shape_expr(env, expr):
  return tuple(eval_poly(env, poly) for poly in expr)

def eval_poly(env, poly):
  return sum(coeff * prod([env[id] ** deg for id, deg in mon.items()])
             for mon, coeff in poly.items())

class ShapeError(Exception): pass

# To denote some shape expressions (for annotations) we use a small language.
#
#   data Shape = Shape [Dim]
#   data Dim = Id Str
#            | Lit Int
#            | Mul Dim Dim

# We'll also make a simple concrete syntax for annotation. The grammar is
#
#   shape_spec ::= '(' dims ')'
#   dims       ::= dim ',' dims | ''
#   dim        ::= str | int | dim '*' dim

def parse_spec(spec=''):
  if not spec:
    return ShapeExpr(())
  if spec[0] == '(':
    if spec[-1] != ')': raise SyntaxError
    spec = spec[1:-1]
  dims = map(parse_dim, spec.replace(' ', '').strip(',').split(','))
  return ShapeExpr(dims)

def parse_dim(spec):
  if '*' in spec:
    terms = map(parse_dim, spec.split('*'))
    return reduce(op.mul, terms)
  elif spec in digits:
    return parse_lit(spec)
  elif spec in identifiers:
    return parse_id(spec)
  else:
    raise SyntaxError
digits = frozenset(string.digits)
identifiers = frozenset(string.lowercase)

def parse_id(name): return Poly({Mon({name: 1}): 1})
def parse_lit(val_str): return Poly({Mon(): int(val_str)})

print(parse_spec('(m, n)'))     # ShapeExpr(m, n)
print(parse_spec('(m * n)'))    # ShapeExpr(m n)
print(parse_spec('(m * n,)'))   # ShapeExpr(m n)
print(parse_spec('(3, m)'))     # ShapeExpr(3, m)
print(parse_spec('(3 * m)'))    # ShapeExpr(3 m)
print(parse_spec('m'))          # ShapeExpr(m)
print(parse_spec(''))           # ShapeExpr()

Shape = parse_spec

### tracer machinery

class MaskTracer(Tracer):
  __slots__ = ["val", "shape_expr", "shape_env"]

  def __init__(self, trace, shape_env, val, shape_expr):
    self.trace = trace
    self.shape_env = shape_env
    self.val = val
    self.shape_expr = shape_expr

  @property
  def aval(self):
    # TODO can avoid some blowups, also improve error messages
    if self.shape_env is not None:
      shape = eval_shape_expr(self.shape_env, self.shape_expr)
      return ShapedArray(tuple(shape), self.val.dtype)
    else:
      return ShapedArray(self.val.shape, self.val.dtype)

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
    return MaskTracer(self, val.shape_env, val.val, val.shape_expr)

  def process_primitive(self, primitive, tracers, params):
    shape_env = next(t.shape_env for t in tracers if t.shape_env is not None)
    vals, shape_exprs = unzip2((t.val, t.shape_expr) for t in tracers)
    out_shape = shape_rules[primitive](shape_exprs, **params)
    logical_shapes = map(partial(eval_shape_expr, shape_env), shape_exprs)
    out = masking_rules[primitive](vals, logical_shapes, **params)
    if not primitive.multiple_results:
      return MaskTracer(self, shape_env, out, out_shape)
    else:
      return map(partial(MaskTracer, self, shape_env), out, out_shape)

  def process_call(self, call_primitive, f, tracers, params):
    raise NotImplementedError  # TODO

masking_rules = {}
shape_rules = {}


def reduce_sum_shape_rule(shape_exprs, axes, input_shape):
  del input_shape  # Unused.
  shape_expr, = shape_exprs
  return ShapeExpr(*(d for i, d in enumerate(shape_expr) if i not in axes))
shape_rules[lax.reduce_sum_p] = reduce_sum_shape_rule

def reduce_sum_masking_rule(padded_vals, logical_shapes, axes, input_shape):
  del input_shape  # Unused.
  (padded_val,), (logical_shape,) = padded_vals, logical_shapes
  masks = [lax.broadcasted_iota(onp.int32, padded_val.shape, i) < d
           for i, d in enumerate(logical_shape)]
  mask = reduce(op.and_, masks)
  masked_val = lax.select(mask, padded_val, lax.zeros_like_array(padded_val))
  return lax.reduce_sum_p.bind(masked_val, axes=axes,
                               input_shape=padded_val.shape)
masking_rules[lax.reduce_sum_p] = reduce_sum_masking_rule


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

defbinop(lax.add_p)
defbinop(lax.mul_p)
defbinop(lax.sub_p)
defbinop(lax.div_p)
defbinop(lax.pow_p)


def defvectorized(prim):
  shape_rules[prim] = vectorized_shape_rule
  masking_rules[prim] = partial(vectorized_masking_rule, prim)

def vectorized_shape_rule(shape_exprs):
  shape_expr, = shape_exprs
  return shape_expr

def vectorized_masking_rule(prim, padded_vals, logical_shapes):
  del logical_shapes  # Unused.
  padded_val, = padded_vals
  return prim.bind(padded_val)

defvectorized(lax.neg_p)
defvectorized(lax.sin_p)
defvectorized(lax.cos_p)
defvectorized(lax.exp_p)
defvectorized(lax.log_p)
defvectorized(lax.tanh_p)


def scan_shape_rule(shape_exprs, forward, length, jaxpr, num_consts, num_carry,
                    linear):
  const_shexprs, init_shexprs, xs_shexprs = split_list(shape_exprs, [num_consts, num_carry])
  if (any(any(type(d) is Id for d in shexpr) for shexpr in const_shexprs)
      or any(any(type(d) is Id for d in shexpr) for shexpr in init_shexprs)
      or any(any(type(d) is Id for d in shexpr[1:]) for shexpr in xs_shexprs)):
    raise NotImplementedError
  _, y_avals = split_list(jaxpr.out_avals, [num_carry])
  ys_shapes = [ShapeExpr(length, *y_aval.shape) for y_aval in y_avals]
  return init_shexprs + ys_shapes
shape_rules[lax.scan_p] = scan_shape_rule

def scan_masking_rule(padded_vals, logical_shapes, forward, length, jaxpr,
                      num_consts, num_carry, linear):
  masked_jaxpr = _masked_scan_jaxpr(jaxpr, num_consts, num_carry)
  consts, init, xs = split_list(padded_vals, [num_consts, num_carry])
  max_length, = {x.shape[0] for x in xs}
  const_linear, init_linear, xs_linear = split_list(linear, [num_consts, num_carry])
  out_vals = lax.scan_p.bind(
      *it.chain([length] + consts, [0], init, xs),
      forward=forward, length=max_length, jaxpr=masked_jaxpr,
      num_consts=1 + num_consts, num_carry=1 + num_carry,
      linear=[False] + const_linear + [False] + init_linear + xs_linear)
  return out_vals[1:]
masking_rules[lax.scan_p] = scan_masking_rule

def _masked_scan_jaxpr(jaxpr, num_consts, num_carry):
  fun = core.jaxpr_as_fun(jaxpr)

  @lu.wrap_init
  def masked(*args):
    [dynamic_length], consts, [i], carry, xs = split_list(
        args, [1, num_consts, 1, num_carry])
    out = fun(*(consts + carry + xs))
    new_carry, ys = split_list(out, [num_carry])
    new_carry = [lax.select(i < dynamic_length, new_c, c)
                 for new_c, c in zip(new_carry, carry)]
    return [i + 1] + new_carry + ys

  aval = ShapedArray((), onp.int32)
  const_avals, carry_avals, x_avals = split_list(jaxpr.in_avals, [num_consts, num_carry])
  return _make_typed_jaxpr(masked, [aval] + const_avals + [aval] + carry_avals + x_avals)

def _make_typed_jaxpr(traceable, in_avals):
  pvals = [pe.PartialVal((aval, core.unit)) for aval in in_avals]
  jaxpr, pvals_out, consts = pe.trace_to_jaxpr(traceable, pvals, instantiate=True)
  assert not consts
  out_avals, _ = unzip2(pvals_out)
  return core.TypedJaxpr(jaxpr, consts, in_avals, out_avals)


def dot_shape_rule(shape_exprs, precision):
  del precision  # Unused.
  lhs_shape, rhs_shape = shape_exprs
  lhs_ndim, rhs_ndim = len(lhs_shape), len(rhs_shape)

  assert False  # TODO update w/ new ShapeExpr stuff
  if lhs_ndim == rhs_ndim == 1:
    if not lhs_shape == rhs_shape: raise ShapeError
    return Shape()
  elif lhs_ndim == rhs_ndim == 2:
    if not lhs_shape[1] == rhs_shape[0]: raise ShapeError
    return Shape(lhs_shape[0], rhs_shape[1])
  elif rhs_ndim == 1:
    if not lhs_shape[1] == rhs_shape[0]: raise ShapeError
    return Shape(lhs_shape[0])
  else:
    if not lhs_shape[0] == rhs_shape[0]: raise ShapeError
    return Shape(rhs_shape[1])
shape_rules[lax.dot_p] = dot_shape_rule

def dot_masking_rule(padded_vals, logical_shapes, precision):
  lhs, rhs = padded_vals
  lhs_shape, rhs_shape = logical_shapes
  lhs_ndim, rhs_ndim = len(lhs_shape), len(rhs_shape)

  if lhs_ndim == rhs_ndim == 1:
    masked_lhs = lax.select(lax.iota(onp.int32, lhs.shape[0]) < lhs_shape[0],
                            lhs, lax.zeros_like_array(lhs))
    return lax.dot_p.bind(masked_lhs, rhs, precision=precision)
  elif lhs_ndim == rhs_ndim == 2:
    # TODO could avoid select if we check whether contracted axis is masked
    masked_lhs = lax.select(lax.broadcasted_iota(onp.int32, lhs.shape, 1) < lhs_shape[1],
                            lhs, lax.zeros_like_array(lhs))
    return lax.dot_p.bind(masked_lhs, rhs, precision=precision)
  elif rhs_ndim == 1:
    raise NotImplementedError
  else:
    raise NotImplementedError
masking_rules[lax.dot_p] = dot_masking_rule


def reshape_shape_rule(shape_exprs, new_sizes, dimensions, old_sizes):
  import ipdb; ipdb.set_trace()
  del old_sizes  # Unused.
  if dimensions is not None: raise NotImplementedError
  shape_expr, = shape_exprs
  if prod(shape_expr) != prod(new_sizes): raise Exception
  return ShapeExpr(new_sizes)
shape_rules[lax.reshape_p] = reshape_shape_rule

def reshape_masking_rule(padded_vals, logical_shapes, new_sizes, dimensions,
                         old_sizes):
  import ipdb; ipdb.set_trace()
  del new_sizes, old_sizes  # Unused.
  if dimensions is not None: raise NotImplementedError
  padded_operand, = padded_vals
  new_shape, = logical_shapes
  return lax.reshape(padded_operand)  # TODO
masking_rules[lax.reshape_p] = reshape_masking_rule


###

def mask(fun, in_shapes, out_shape):
  in_shapes_flat, in_shapes_tree = tree_flatten(in_shapes)
  out_shapes_flat, out_shapes_tree = tree_flatten(out_shape)

  def wrapped_fun(args, shape_env):
    f = lu.wrap_init(fun)
    args_flat, in_tree = tree_flatten(args)
    assert in_tree == in_shapes_tree
    # padded_sizes = _check_shape_agreement(args_flat, in_shapes_flat)  # TODO
    flat_fun, out_tree = flatten_fun_nokwargs(f, in_tree)
    outs, out_shapes_ = mask_fun(flat_fun, shape_env, args_flat, in_shapes_flat)
    assert out_shapes_flat == list(out_shapes_)
    # _check_shape_agreement(outs, out_shapes_flat, padded_sizes)
    return tree_unflatten(out_tree(), outs)
  return wrapped_fun

def _check_shape_agreement(padded_args, shape_exprs, shape_values=None):
  shape_values = shape_values or defaultdict(set)
  for arg, shexpr in zip(padded_args, shape_exprs):
    for padded_size, size_expr in zip(arg.shape, shexpr):
      if type(size_expr) is Id:
        shape_values[size_expr].add(padded_size)
      elif type(size_expr) is int:
        if padded_size != size_expr: raise ShapeError
      else:
        raise TypeError(size_expr)
  for shape_var, sizes in shape_values.items():
    if len(sizes) != 1:
      raise ShapeError
  return shape_values

###

import jax.numpy as np
from jax import vmap, jit


@partial(mask, in_shapes=[Shape('n')], out_shape=Shape())
def padded_sum(x):
  return np.sum(x)  # output shape ()

print(padded_sum([np.arange(5)], dict(n=3)))
print(vmap(padded_sum)([np.ones((5, 10))], dict(n=np.arange(5))))


@partial(mask, in_shapes=[Shape('n'), Shape('n')], out_shape=Shape('n'))
def addvecs(x, y):
  return x + y
print(addvecs([np.arange(5), np.arange(5)], dict(n=3)))
# try: addvecs([np.arange(5), np.arange(6)], dict(n=3))  # TODO
# except ShapeError: print("good error")
# else: raise Exception


def cumsum_(arr):
  out, _ = lax.scan(lambda c, x: (c + x, ()), 0, arr)
  return out

@partial(mask, in_shapes=[Shape('n')], out_shape=Shape())
def cumsum(x):
  return cumsum_(x)

print(cumsum([np.array([5, 2, 9, 1, 4])], dict(n=3)))
print(vmap(cumsum)([np.arange(6).reshape(2, 3)], dict(n=np.array([1, 2]))))

@jit
def jit_cumsum(args, shape_env):
  print("Python!")
  return cumsum(args, shape_env)
print(jit_cumsum([np.array([5, 2, 9, 1, 4])], dict(n=3)))
print(jit_cumsum([np.array([5, 2, 9, 1, 4])], dict(n=4)))


@partial(mask, in_shapes=[Shape('(m, n)')], out_shape=Shape('m * n'))
def flatten(x):
  pass  # TODO


# @partial(mask, in_shapes=[Shape('m', 'k'), Shape('k', 'n')],
#          out_shapes=[Shape('m', 'n')])
# def dot(x, y):
#   return lax.dot(x, y)
# x = onp.arange(6, dtype=onp.float32).reshape((2, 3))
# y = onp.arange(12, dtype=onp.float32).reshape((3, 4))
# print(dot([x, y], dict(m=2, k=2, n=2))[:2, :2])
# print(onp.dot(x[:2, :2], y[:2, :2]))

# notes!
# - a shape variable is associated with a max size and a dynamic size. we carry
#   around the dynamic size explicitly in the shape_env attached to every
#   tracer, while the max size we get off the val
# - we don't want to do the padding at the start and slicing at the end in the
#   transformation because we want to be able to vmap it, also want to jit it
#   - we could have a ragged array data type, or other api options
# - we should probably pass the max size explicitly rather than getting it off
#   the values, e.g. the iota problem, should think of it as an independent type
#   argument


# next steps:
#   0. generic test setup
#   1. clean up shape expression language (maybe handle reshape/conat)
#   2. write example colab with two applications:
#      (a) batching ragged sequences
#      (b) jit bucketing
