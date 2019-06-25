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

from contextlib import contextmanager
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
from . import pxla

zip = safe_zip

def identity(x): return x


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
    ans = yield in_tracers, {}
    out_tracer = trace.full_raise(ans)
    out_val, out_axis = out_tracer.val, out_tracer.axis
    del master, out_tracer
  yield out_val, out_axis

def match_axis(src, dst, x):
  assert type(src) is int
  if src == dst:
    return x
  else:
    return _match_axis(src, dst, x, core.get_aval(x))

def _match_axis(src, dst, x, aval):
  if type(aval) is core.AbstractTuple:
    if type(dst) is tuple:
      return core.pack(map(partial(_match_axis, src), dst, x, aval))
    else:
      return core.pack(map(partial(_match_axis, src, dst), x, aval))
  elif isinstance(aval, ShapedArray):
    if type(dst) is int:
      perm = [i for i in range(x.ndim) if i != src]
      perm.insert(dst, src)
      return x.transpose(perm)
    elif dst is None:
      return x[src]
    else:
      raise TypeError(dst)
  else:
    raise TypeError(aval)

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
      val_out, axis_out = rule(name, size, vals, axes, **params)
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


papply_primitive_rules = {}
