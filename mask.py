from __future__ import print_function

from collections import defaultdict
from functools import partial
import itertools as it
import operator

import numpy as onp

from jax import core
from jax.core import Trace, Tracer
from jax.util import unzip2, prod, safe_map, safe_zip, split_list
from jax.api_util import tree_flatten, tree_unflatten, flatten_fun_nokwargs
from jax import linear_util as lu
from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe
from jax import lax

map = safe_map
zip = safe_zip


Var = str

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

class ShapeExpr(object):
  def __init__(self, *shape):
    assert all(isinstance(s, (int, Var)) for s in shape)
    self.shape = tuple(shape)
  def __len__(self):
    return len(self.shape)
  def __iter__(self):
    return iter(self.shape)
  def __getitem__(self, idx):
    return list(self)[idx]
  def __repr__(self):
    return 'ShapeExpr({})'.format(repr(self.shape))
  __str__ = __repr__
  def __eq__(self, other):
    return type(other) is ShapeExpr and self.shape == other.shape
Shape = ShapeExpr

class ShapeError(Exception): pass

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

def eval_shape_expr(shape_env, shape_expr):
  return tuple(shape_env[d] if type(d) is Var else d for d in shape_expr)

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
  mask = reduce(operator.and_, masks)
  masked_val = lax.select(mask, padded_val, lax.zeros_like_array(padded_val))
  return lax.reduce_sum_p.bind(masked_val, axes=axes,
                               input_shape=padded_val.shape)
masking_rules[lax.reduce_sum_p] = reduce_sum_masking_rule


def add_shape_rule(shape_exprs):
  x_shape_expr, y_shape_expr = shape_exprs
  if not x_shape_expr == y_shape_expr: raise ShapeError
  return x_shape_expr
shape_rules[lax.add_p] = add_shape_rule

def add_masking_rule(padded_vals, logical_shapes):
  del logical_shapes  # Unused.
  padded_x, padded_y = padded_vals
  return padded_x + padded_y
masking_rules[lax.add_p] = add_masking_rule


def scan_shape_rule(shape_exprs, forward, length, jaxpr, num_consts, num_carry,
                    linear):
  const_shexprs, init_shexprs, xs_shexprs = split_list(shape_exprs, [num_consts, num_carry])
  if (any(any(type(d) is Var for d in shexpr) for shexpr in const_shexprs)
      or any(any(type(d) is Var for d in shexpr) for shexpr in init_shexprs)
      or any(any(type(d) is Var for d in shexpr[1:]) for shexpr in xs_shexprs)):
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


###

def mask(fun, in_shapes, out_shapes):
  in_shapes_flat, in_shapes_tree = tree_flatten(in_shapes)
  out_shapes_flat, out_shapes_tree = tree_flatten(out_shapes)

  def wrapped_fun(args, shape_env):
    f = lu.wrap_init(fun)
    args_flat, in_tree = tree_flatten(args)
    assert in_tree == in_shapes_tree
    padded_sizes = _check_shape_agreement(args_flat, in_shapes_flat)
    flat_fun, out_tree = flatten_fun_nokwargs(f, in_tree)
    outs, out_shapes_ = mask_fun(flat_fun, shape_env, args_flat, in_shapes_flat)
    assert out_shapes_flat == out_shapes
    _check_shape_agreement(outs, out_shapes_flat, padded_sizes)
    return tree_unflatten(out_tree(), outs)
  return wrapped_fun

def _check_shape_agreement(padded_args, shape_exprs, shape_values=None):
  shape_values = shape_values or defaultdict(set)
  for arg, shexpr in zip(padded_args, shape_exprs):
    for padded_size, size_expr in zip(arg.shape, shexpr):
      if type(size_expr) is Var:
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


@partial(mask, in_shapes=[Shape('n')], out_shapes=[Shape()])
def padded_sum(x):
  return np.sum(x)  # output shape ()
print(padded_sum([np.arange(5)], dict(n=3)))
print(vmap(padded_sum)([np.ones((5, 10))], dict(n=np.arange(5))))


@partial(mask, in_shapes=[Shape('n'), Shape('n')], out_shapes=[Shape('n')])
def addvecs(x, y):
  return x + y
print(addvecs([np.arange(5), np.arange(5)], dict(n=3)))
try: addvecs([np.arange(5), np.arange(6)], dict(n=3))
except ShapeError: print("good error")
else: raise Exception


def cumsum_(arr):
  out, _ = lax.scan(lambda c, x: (c + x, ()), 0, arr)
  return out
@partial(mask, in_shapes=[Shape('n')], out_shapes=[Shape()])
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
