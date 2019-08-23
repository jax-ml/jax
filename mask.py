from functools import partial
import operator

import numpy as onp

from jax import core
from jax.core import Trace, Tracer
from jax.util import unzip2, prod
from jax import linear_util as lu
from jax.abstract_arrays import ShapedArray
from jax import lax


Var = str


def mask(fun, shape_env, in_vals, shape_exprs):
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

class Shape(object):
  def __init__(self, *shape):
    assert all(isinstance(s, (int, str)) for s in shape)
    self.shape = tuple(shape)
  def __iter__(self):
    return iter(self.shape)
  def __repr__(self):
    return 'Shape({})'.format(repr(self.shape))
  __str__ = __repr__
  def __eq__(self, other):
    return type(other) is Shape and self.shape == other.shape

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
      shape = [self.shape_env[d] if type(d) is Var else d
              for d in self.shape_expr]
      return ShapedArray(shape, self.val.dtype)
    else:
      return ShapedArray(self.val.shape, self.val.dtype)

  def full_lower(self):
    if all(type(s) is int for s in self.shape_expr):
      return core.full_lower(self.val)
    else:
      return self

class MaskTrace(Trace):
  def pure(self, val):
    return MaskTracer(self, None, val, Shape(*val.shape))

  def lift(self, val):
    return MaskTracer(self, None, val, Shape(*val.shape))

  def sublift(self, val):
    return MaskTracer(self, val.shape_env, val.val, val.shape_expr)

  def process_primitive(self, primitive, tracers, params):
    assert not primitive.multiple_results
    shape_env = next(t.shape_env for t in tracers if t.shape_env is not None)
    vals, shape_exprs = unzip2((t.val, t.shape_expr) for t in tracers)
    rule = masking_rules[primitive]
    out, out_shape = rule(shape_env, vals, shape_exprs, **params)
    return MaskTracer(self, shape_env, out, out_shape)

  def process_call(self, call_primitive, f, tracers, params):
    raise NotImplementedError  # TODO

masking_rules = {}


def reduce_sum_masking_rule(shape_env, vals, shape_exprs, axes, input_shape):
  val, = vals
  in_shape, = shape_exprs
  masks = [lax.broadcasted_iota(onp.int32, val.shape, i) < shape_env[d]
           for i, d in enumerate(in_shape) if type(d) is Var]
  mask = reduce(operator.and_, masks)
  masked_val = lax.select(mask, val, lax.zeros_like_array(val))
  out_val = lax.reduce_sum_p.bind(masked_val, axes=axes,
                                  input_shape=masked_val.shape)
  out_shape = Shape(*(d for i, d in enumerate(in_shape) if i not in axes))
  return out_val, out_shape

masking_rules[lax.reduce_sum_p] = reduce_sum_masking_rule


###


def pad(fun, in_shapes, out_shapes):
  def wrapped_fun(args, shape_env):
    outs, out_shapes_ = mask(lu.wrap_init(fun), shape_env, args, in_shapes)
    assert tuple(out_shapes_) == tuple(out_shapes)
    return outs
  return wrapped_fun

###

import jax.numpy as np
from jax import vmap

@partial(pad, in_shapes=[Shape('n')], out_shapes=[Shape()])
def padded_sum(x):
  return np.sum(x),  # output shape ()

print padded_sum([np.arange(5)], dict(n=3))
print vmap(padded_sum)([np.ones((5, 10))], dict(n=np.arange(5)))
