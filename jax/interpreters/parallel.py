# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import warnings

import numpy as onp
import six
from six.moves import reduce

from .. import core
from .. import linear_util as lu
from ..core import Trace, Tracer, Primitive, new_master
from ..abstract_arrays import ShapedArray, ConcreteArray, make_shaped_array
from ..util import safe_zip, unzip2, unzip3, partialmethod, prod
from ..lib import xla_bridge as xb
from . import partial_eval as pe
from . import batching

zip = safe_zip

def identity(x): return x


### serial_pmap is like pmap but executes in a single-machine vectorized way


def serial_pmap(fun, name, in_vals, in_axes, out_axis_target):
  sizes = reduce(set.union, map(batching.dimsize, in_axes, in_vals))
  if not sizes:
    return fun.call_wrapped(*in_vals)
  elif len(sizes) == 1:
    fun, out_axis = serial_pmap_transform(fun, name, in_axes)
    out_val = fun.call_wrapped(*in_vals)
    return batching.moveaxis(sizes.pop(), out_axis_target, out_axis(), out_val)
  else:
    raise TypeError("got inconsistent map dimension sizes: {}".format(sizes))

@lu.transformation_with_aux
def serial_pmap_transform(name, axes, *vals):
  with new_master(SerialPmapTrace) as master:
    trace = SerialPmapTrace(master, core.cur_sublevel())
    in_tracers = map(partial(SerialPmapTracer, trace, name), vals, axes)
    ans = yield in_tracers
    out_tracer = trace.full_raise(ans)
    out_val, out_axis = out_tracer.val, out_tracer.axis
    del master, out_tracer
  yield out_val, out_axis

@lu.transformation_with_aux
def serial_pmap_subtrace(master, name, axes, *vals):
  trace = SerialPmapTrace(master, core.cur_sublevel())
  ans = yield map(partial(SerialPmapTracer, trace, name), vals, axes)
  out_tracer = trace.full_raise(ans)
  out_val, out_axis = out_tracer.val, out_tracer.axis
  yield out_val, out_axis

class SerialPmapTracer(Tracer):
  def __init__(self, trace, name, val, axis):
    self.trace = trace
    self.name = name
    self.val = val
    self.axis = axis

  @property
  def aval(self):
    batched_aval = batching.get_aval(self.val)
    return batching.remove_batch_dim_from_aval(self.axis, batched_aval)

  def unpack(self):
    t = type(self.axis)
    if t is tuple:
      axes = self.axis
    elif t is int:
      axes = [self.axis] * len(self.val)
    elif t is type(None):
      return tuple(self.val)
    else:
      raise TypeError(t)
    return map(partial(SerialPmapTracer, self.trace, self.name), self.val, axes)

  def full_lower(self):
    if self.axis is None:
      return core.full_lower(self.val)
    else:
      return self

class SerialPmapTrace(Trace):
  def pure(self, val):
    return SerialPmapTracer(self, None, val, None)

  def lift(self, val):
    return SerialPmapTracer(self, None, val, None)

  def sublift(self, val):
    return SerialPmapTracer(self, val.name, val.val, val.axis)

  def process_primitive(self, primitive, tracers, params):
    names_in, vals_in, axes_in = unzip3((t.name, t.val, t.axis) for t in tracers)
    if all(axis is None for axis in axes_in):
      return primitive.bind(*vals_in, **params)
    else:
      name = next(name for name in names_in if name is not None)  # all same
      if primitive in serial_pmap_primitive_rules:
        # if it's a pmap collective primitive, do something special
        val_in, = vals_in
        axis_in, = axes_in
        if name == params['axis_name']:
          # if the name matches this tracer's name, apply the pmap rule
          rule = serial_pmap_primitive_rules[primitive]
          params = {k: params[k] for k in params if k != 'axis_name'}
          val_out, axis_out = rule(val_in, axis_in, **params)
          return SerialPmapTracer(self, name, val_out, axis_out)
        else:
          # if not, bind the primitive so that any other pmap tracers can see it
          val_out = primitive.bind(val_in, **params)
          return SerialPmapTracer(self, name, val_out, axis_in)
      else:
        # if it's not a pmap collective primitive, act just like vmap
        rule = batching.get_primitive_batcher(primitive)
        val_out, axis_out = rule(vals_in, axes_in, **params)
        return SerialPmapTracer(self, name, val_out, axis_out)

  def process_call(self, call_primitive, f, tracers, params):
    names, vals, axes = unzip3((t.name, t.val, t.axis) for t in tracers)
    if all(axis is None for axis in axes):
      return call_primitive.bind(f, *vals, **params)
    else:
      name = next(name for name in names if name is not None)  # all same
      f, axis_out = serial_pmap_subtrace(f, self.master, name, axes)
      val_out = call_primitive.bind(f, *vals, **params)
      return SerialPmapTracer(self, name, val_out, axis_out())

  def post_process_call(self, _, out_tracer):
    name, val, axis = out_tracer.name, out_tracer.val, out_tracer.axis
    master = self.master
    def todo(x):
      trace = SerialPmapTrace(master, core.cur_sublevel())
      return SerialPmapTracer(trace, name, x, axis)

    return val, todo

  def pack(self, tracers):
    vals = core.pack([t.val for t in tracers])
    axis = tuple(t.axis for t in tracers)
    name = next(t.name for t in tracers if t.name)
    return SerialPmapTracer(self, name, vals, axis)


serial_pmap_primitive_rules = {}


### papply


newvar = pe.gensym('_axis')

def papply(fun, name, in_vals, axis_size, in_axes, out_axis):
  out_val = papply_transform(fun).call_wrapped(
      name, in_vals, axis_size, in_axes, out_axis)
  return out_val

def ensure_axis(dst, src, x):
  aval = batching.get_aval(x)
  if type(aval) is core.AbstractTuple:
    if type(src) is tuple and type(dst) is tuple:
      return core.pack(map(ensure_axis, dst, src, x))
    elif type(src) is tuple:
      return core.pack(map(partial(ensure_axis, dst), src, x))
    elif type(dst) is tuple:
      srcs = (src,) * len(dst)
      return core.pack(map(ensure_axis, dst, srcs, x))
    else:
      return core.pack(map(partial(ensure_axis, dst, src), x))
  elif isinstance(aval, ShapedArray):
    if src == dst:
      return x
    elif src is None:
      warnings.warn('split output axis requested for an array with no split')
      return x
    else:
      perm = list(range(x.ndim))
      perm[src] = dst
      perm[dst] = src
      return x.transpose(perm)
  else:
    raise TypeError(type(aval))

@lu.transformation
def papply_transform(name, args, axis_size, in_axes, out_axis):
  with new_master(PapplyTrace) as master:
    trace = PapplyTrace(master, core.cur_sublevel())
    in_tracers = map(partial(PapplyTracer, trace, name, axis_size), args, in_axes)
    out_tracer = yield in_tracers
    out_tracer = trace.full_raise(out_tracer)
    out_tracer = ensure_axis(out_axis, out_tracer.axis, out_tracer)
    out_val = out_tracer.val
    del master, out_tracer
  yield out_val

class PapplyTracer(Tracer):
  def __init__(self, trace, name, axis_size, val, axis):
    self.trace = trace
    self.name = name
    self.axis_size = axis_size
    self.val = val
    self.axis = axis

  @property
  def aval(self):
    batched_aval = batching.get_aval(self.val)
    return batching.add_batch_dim_to_aval(
        self.axis, self.axis_size, batched_aval)

  def unpack(self):
    raise NotImplementedError  # TODO(mattjj,frostig)

  def full_lower(self):
    if self.axis is None:
      return core.full_lower(self.val)
    else:
      return self

class PapplyTrace(Trace):
  def pure(self, val):
    return PapplyTracer(self, None, None, val, None)

  def lift(self, val):
    return PapplyTracer(self, None, None, val, None)

  def sublift(self, val):
    return PapplyTracer(self, val.name, val.axis_size, val.val, val.axis)

  def process_primitive(self, primitive, tracers, params):
    names, vals, axes = unzip3((t.name, t.val, t.axis) for t in tracers)
    if all(axis is None for axis in axes):
      return primitive.bind(*vals, **params)
    else:
      name = next(n for n in names if n is not None)
      size = next(t.axis_size for t in tracers if t.axis_size is not None)
      rule = papply_primitive_rules[primitive]
      val_out, axis_out = rule(name, vals, axes, **params)
      return PapplyTracer(self, name, size, val_out, axis_out)

  def process_call(self, call_primitive, f, tracers, params):
    raise NotImplementedError  # TODO(mattjj,frostig)

  def post_process_call(self, _, out_tracer):
    raise NotImplementedError  # TODO(mattjj,frostig)

  def pack(self, tracers):
    vals = core.pack([t.val for t in tracers])
    axis = tuple(t.axis for t in tracers)
    name = tuple(t.name for t in tracers)
    size = tuple(t.axis_size for t in tracers)
    return PapplyTracer(self, name, size, vals, axis)


def vectorized_papply(prim, name, vals, axes, **params):
  assert all(axes[0] == a for a in axes[1:])
  return prim.bind(*vals, **params), axes[0]

def reducer_papply(prim, cprim, name, vals, papply_axes, axes, **kwargs):
  operand, = vals
  papply_axis, = papply_axes

  other_axes = [i for i in axes if i != papply_axis]
  other_axes = [i - 1 if i > papply_axis else i for i in other_axes]

  if other_axes:
    if 'input_shape' in kwargs:  # special to the reduce-sum family
      s = kwargs['input_shape']
      kwargs['input_shape'] = s[:papply_axis] + s[papply_axis + 1:]
    result = prim.bind(operand, axes=tuple(other_axes), **kwargs)
  else:
    result = operand

  if not axes or papply_axis in axes:
    return cprim.bind(result, axis_name=name), None
  else:
    new_papply_axis = papply_axis - onp.sum(onp.less(other_axes, papply_axis))
    return result, new_papply_axis


def broadcasting_papply(prim, name, vals, axes, **params):
  x, y = vals
  xdim, ydim = axes

  if xdim is None:
    assert x.shape[ydim] == 1
    x = x.reshape(onp.delete(x.shape, ydim))
    return prim.bind(x, y, **params), ydim
  elif ydim is None:
    assert y.shape[xdim] == 1
    y = y.reshape(onp.delete(y.shape, xdim))
    return prim.bind(x, y, **params), xdim
  elif xdim == ydim:
    return prim.bind(x, y, **params), xdim
  else:
    x = psplit(x, axis_name, ydim)
    return prim.bind(x, y, **params), ydim


papply_primitive_rules = {}

def defvectorized(prim):
  papply_primitive_rules[prim] = partial(vectorized_papply, prim)

def defreducer(prim, collective_prim):
  papply_primitive_rules[prim] = partial(reducer_papply, prim,
                                         collective_prim)

def defbroadcasting(prim):
  papply_primitive_rules[prim] = partial(broadcasting_papply, prim)
