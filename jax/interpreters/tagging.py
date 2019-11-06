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

"""Tracer for tagging and manipulating a computation's intermediate values.

This relies on two features added to the jaxpr language. First, JAX values can
optionally carry a stack of tags as compile-time metadata, and the `push_tag`
primitive can add a string or other Python value to this stack. Second, the
`yield_value` primitive marks a value as a tagged intermediate that can be
referenced and manipulated using its tag stack.

Other primitives propagate their shortest argument path by default, except for 
`tie_in`, which propagates the path of its first argument.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import itertools as it

import numpy as onp

from .. import core
from ..core import Trace, Tracer, new_master
from ..abstract_arrays import ShapedArray, make_shaped_array, array_types, raise_to_shaped
from ..ad_util import add_jaxvals, add_jaxvals_p, zeros_like_jaxval, zeros_like_p
from ..linear_util import transformation, transformation_with_aux, wrap_init
from ..dict_util import maybe_set_dict
from ..util import unzip2, partial, safe_map
from . import xla
from . import partial_eval as pe

map = safe_map

yield_primitives = set()
custom_tagging_rules = {}

def _standard_tagging_rule(prim, args, paths, **params):
  # propagate the shortest non-None path
  paths = [path for path in paths if path is not None]
  if len(paths) > 0:
    out_path = min(paths, key=lambda path: len(path))
  else:
    out_path = None
  out = prim.bind(*args, **params)
  return out, out_path, (prim in yield_primitives)

def get_primitive_tagger(p):
  if p in custom_tagging_rules:
    return custom_tagging_rules[p]
  else:
    return partial(_standard_tagging_rule, p)


class TagTrace(core.Trace):

  def __init__(self, master, sublevel, accum_fn, tree=None):
    super(TagTrace, self).__init__(master, sublevel)
    self.tree = dict() if tree is None else tree
    self.accum_fn = accum_fn

  def with_sublevel(self, sublevel):
    return type(self)(self.master, sublevel, self.accum_fn, self.tree)

  def pure(self, val):
    return TagTracer(self, val, None)

  def lift(self, val):
    return TagTracer(self, val, None)

  def sublift(self, val):
    return TagTracer(self, val.val, val.path)

  def maybe_set(self, path, val):
    return maybe_set_dict(self.tree, path, val, self.accum_fn)

  def process_primitive(self, primitive, tracers, params):
    args, paths = unzip2((t.val, t.path) for t in tracers)
    tagging_rule = get_primitive_tagger(primitive)
    out, out_path, is_sample = tagging_rule(args, paths)
    if is_sample:
      assert not primitive.multiple_results
      out = self.maybe_set(out_path, out)
    if primitive.multiple_results:
      return [TagTracer(self, val, out_path) for val in out]
    else:
      return TagTracer(self, out, out_path)

  def process_call(self, call_primitive, f, tracers, params):
    in_vals, in_paths = unzip2((t.val, t.path) for t in tracers)
    f_key, out_paths = tag_subtrace(f, self, in_paths)
    out_vals = call_primitive.bind(f_key, *in_vals, **params)
    return [TagTracer(self, v, p) for v, p in zip(out_vals, out_paths())]

  def process_map(self, map_primitive, f, tracers, params):
    raise NotImplementedError

  def post_process_call(self, call_primitive, out_tracers, params):
    raise NotImplementedError

class TagTracer(core.Tracer):
  __slots__ = ['trace', 'val', 'path']

  def __init__(self, trace, val, path):
    self.trace = trace
    self.val = val
    self.path = path

  def __repr__(self):
    return 'Traced<{}:{}>'.format(self.val, self.trace)

  @property
  def aval(self):
    return core.get_aval(self.val)

  def full_lower(self):
    if self.path is None:
      return self.val
    else:
      return self


@transformation_with_aux
def tag_subtrace(parent, in_paths, *in_vals):
  trace = parent.with_sublevel(core.cur_sublevel())
  in_tracers = map(partial(TagTracer, trace), in_vals, in_paths)
  outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  out_vals, out_paths = unzip2((t.val, t.path) for t in out_tracers)
  yield out_vals, out_paths

def tag_fun(fun, in_vals, in_paths, accum_fn, in_tree):
  with core.new_master(TagTrace) as master:
    trace = TagTrace(master, core.cur_sublevel(), accum_fn, in_tree)
    fun, out_paths = tag_subtrace(fun, trace, in_paths)
    out_vals = fun.call_wrapped(*in_vals)
    del master
  return out_vals, out_paths(), trace.tree
