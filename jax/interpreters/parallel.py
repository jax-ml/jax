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
from ..abstract_arrays import ShapedArray, ConcreteArray, raise_to_shaped
from ..util import safe_map, safe_zip, unzip2, unzip3, partialmethod, prod
from ..lib import xla_bridge as xb
from . import partial_eval as pe
from . import batching
from . import pxla

map = safe_map
zip = safe_zip

def identity(x):
  return x

### papply
def papply(fun, name, in_vals, axis_size):
  # this function is for testing purposes, so we drop the out_axis
  fun, _ = papply_transform(fun, name, axis_size)
  return fun.call_wrapped(*in_vals)

@lu.transformation_with_aux
def papply_transform(name, axis_size, *args):
  with new_master(PapplyTrace) as master:
    trace = PapplyTrace(master, core.cur_sublevel())
    in_tracers = map(partial(PapplyTracer, trace, name, axis_size, axis=0), args)
    outs = yield in_tracers, {}
    out_tracers = map(trace.full_raise, outs)
    out_vals, out_axes = unzip2((t.val, t.axis) for t in out_tracers)
    del master, out_tracers
  yield out_vals, out_axes

@lu.transformation_with_aux
def papply_subtrace(master, name, axis_size, axes, *vals):
  trace = PapplyTrace(master, core.cur_sublevel())
  outs = yield map(partial(PapplyTracer, trace, name, axis_size), vals, axes), {}
  out_tracers = map(trace.full_raise, outs)
  out_vals, out_axes = unzip2((t.val, t.axis) for t in out_tracers)
  yield out_vals, out_axes

# TODO(mattjj); use a special sentinel type rather than None
NotSharded = type(None)
not_sharded = None

class PapplyTracer(Tracer):
  def __init__(self, trace, name, axis_size, val, axis):
    self.trace = trace
    self.name = name
    self.axis_size = axis_size
    self.val = val
    self.axis = axis

  @property
  def aval(self):
    aval = raise_to_shaped(core.get_aval(self.val))
    if self.axis is not_sharded:
      return aval
    else:
      if aval is core.abstract_unit:
        return aval
      elif type(aval) is ShapedArray:
        assert 0 <= self.axis < aval.ndim + 1
        new_shape = list(aval.shape)
        new_shape.insert(self.axis, self.axis_size)
        return ShapedArray(tuple(new_shape), aval.dtype)
      else:
        raise TypeError(aval)

  def full_lower(self):
    if self.axis is not_sharded:
      return core.full_lower(self.val)
    else:
      return self

class PapplyTrace(Trace):
  def pure(self, val):
    return PapplyTracer(self, None, None, val, not_sharded)

  def lift(self, val):
    return PapplyTracer(self, None, None, val, not_sharded)

  def sublift(self, val):
    return PapplyTracer(self, val.name, val.axis_size, val.val, val.axis)

  def process_primitive(self, primitive, tracers, params):
    names, vals, axes = unzip3((t.name, t.val, t.axis) for t in tracers)
    if all(axis is not_sharded for axis in axes):
      return primitive.bind(*vals, **params)
    else:
      name, = {n for n in names if n is not None}
      size, = {t.axis_size for t in tracers if t.axis_size is not None}
      rule = papply_primitive_rules[primitive]
      val_out, axis_out = rule(name, size, vals, axes, **params)
      return PapplyTracer(self, name, size, val_out, axis_out)

  def process_call(self, call_primitive, f, tracers, params):
    if call_primitive in pe.map_primitives:
      return self.process_map(call_primitive, f, tracers, params)
    names, vals, axes = unzip3((t.name, t.val, t.axis) for t in tracers)
    if all(axis is not_sharded for axis in axes):
      return call_primitive.bind(f, *vals, **params)
    else:
      name, = {n for n in names if n is not None}
      size, = {t.axis_size for t in tracers if t.axis_size is not None}
      f_papply, axes_out = papply_subtrace(f, self.master, name, size, axes)
      vals_out = call_primitive.bind(f_papply, *vals, **params)
      return [PapplyTracer(self, name, size, x, a) for x, a in zip(vals_out, axes_out())]

  def post_process_call(self, call_primitive, out_tracer):
    t = out_tracer
    name, val, axis, size = t.name, t.val, t.axis, t.axis_size
    master = self.master

    def todo(x):
      trace = PapplyTrace(master, core.cur_sublevel())
      return PapplyTracer(trace, name, size, x, axis)

    return val, todo

  def process_map(self, map_primitive, f, tracers, params):
    raise NotImplementedError  # TODO(mattjj,frostig)

papply_primitive_rules = {}
