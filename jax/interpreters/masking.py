from collections import namedtuple
from functools import partial
from itertools import chain

import numpy as onp
from jax.interpreters.shapes import _ensure_poly, eval_shape_expr, \
  eval_dim_expr, Poly, shape_rules

from contextlib import contextmanager
from .. import core
from ..core import Trace, Tracer
from ..util import safe_map, safe_zip, unzip2
from ..abstract_arrays import ShapedArray
from .. import linear_util as lu

map = safe_map
zip = safe_zip

shape_parameterized_primitive_rules = {}
masking_rules = {}

def defvectorized(prim):
  masking_rules[prim] = partial(vectorized_masking_rule, prim)

def defnaryop(prim):
  masking_rules[prim] = partial(naryop_masking_rule, prim)

def vectorized_masking_rule(prim, padded_vals, logical_shapes, **params):
  del logical_shapes  # Unused.
  padded_val, = padded_vals
  return prim.bind(padded_val, **params)

def naryop_masking_rule(prim, padded_vals, logical_shapes):
  del logical_shapes  # Unused.
  return prim.bind(*padded_vals)

ShapeEnvs = namedtuple("ShapeEnvs", ["logical", "padded"])
shape_envs = ShapeEnvs({}, {})  # TODO(mattjj): make this a stack for efficiency

@contextmanager
def extend_shape_envs(logical_env, padded_env):
  global shape_envs
  new_logical = dict(chain(shape_envs.logical.items(), logical_env.items()))
  new_padded = dict(chain(shape_envs.padded.items(), padded_env.items()))
  shape_envs, prev = ShapeEnvs(new_logical, new_padded), shape_envs
  yield
  shape_envs = prev

def shape_as_value(expr):
  return tuple(eval_dim_expr(shape_envs.logical, d) if type(d) is Poly else d
               for d in expr)

def padded_shape_as_value(expr):
  return tuple(eval_dim_expr(shape_envs.padded, d) if type(d) is Poly else d
               for d in expr)
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

class MaskTracer(Tracer):
  __slots__ = ["val", "shape_expr"]

  def __init__(self, trace, val, shape_expr):
    self.trace = trace
    self.val = val
    self.shape_expr = shape_expr

  @property
  def aval(self):
    return ShapedArray(self.shape_expr, self.val.dtype)

  def is_pure(self):
    return all(_ensure_poly(poly).is_constant for poly in self.shape_expr)

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

  def process_call(self, call_primitive, f, tracers, params):
    raise NotImplementedError  # TODO mask-of-jit