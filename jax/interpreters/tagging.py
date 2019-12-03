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

This relies on two features added to the jaxpr language. First, we introduce
Scope, a new JAX type consisting entirely of compile-time Python metadata
(a stack of namespaces/names), and a `push_scope` primitive to add a string
or other Python value to this stack. Second, the `tag` primitive marks a value
as a tagged intermediate with a name or nested scope; tge value can then be
referenced and manipulated using this tag.

The two primary public APIs for this functionality are `collect` and `inject`.
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
from ..api_util import flatten_fun, flatten_fun_nokwargs
from ..tree_util import tree_flatten, tree_unflatten
from . import xla
from . import partial_eval as pe

map = safe_map
tag_primitives = set()

class Scope(object):
  """Carries a path made of a stack of names/namespaces."""
  __slots__ = ['path']
  def __init__(self, path=()):
    self.path = path

class AbstractScope(core.AbstractValue):
  __slots__ = ['path']
  def __init__(self, path=()):
    self.path = path

core.pytype_aval_mappings[Scope] = lambda scope: AbstractScope(scope.path)
xla.pytype_aval_mappings[Scope] = lambda scope: AbstractScope(scope.path)
xla.xla_shape_handlers[AbstractScope] = xla.xla_shape_handlers[core.AbstractUnit]
xla.canonicalize_dtype_handlers[Scope] = xla.identity
xla.device_put_handlers[Scope] = xla.device_put_handlers[core.Unit]

class TagTrace(core.Trace):

  def __init__(self, master, sublevel, accum_fn, tree=None):
    super(TagTrace, self).__init__(master, sublevel)
    self.tree = dict() if tree is None else tree
    self.accum_fn = accum_fn

  def with_sublevel(self, sublevel):
    return type(self)(self.master, sublevel, self.accum_fn, self.tree)

  def pure(self, val):
    return TagTracer(self, val)

  def lift(self, val):
    return TagTracer(self, val)

  def sublift(self, val):
    return TagTracer(self, val.val)

  def maybe_set(self, path, val):
    return maybe_set_dict(self.tree, path, val, self.accum_fn)

  def process_primitive(self, primitive, tracers, params):
    in_vals = [t.val for t in tracers]
    out = primitive.bind(*in_vals, **params)
    if primitive in tag_primitives:
      path = in_vals[-1].path
      out = self.maybe_set(path, out)
    if primitive.multiple_results:
      return [TagTracer(self, val) for val in out]
    else:
      return TagTracer(self, out)

  def process_call(self, call_primitive, f, tracers, params):
    in_vals = [t.val for t in tracers]
    f_tag = tag_subtrace(f, self)
    all_args, in_treedef = tree_flatten((in_vals, self.tree))
    f_tag_flat, out_treedef = flatten_fun_nokwargs(f_tag, in_treedef)
    out_flat = call_primitive.bind(f_tag_flat, *all_args, **params)
    out_vals, out_tree = tree_unflatten(out_treedef(), out_flat)
    self.tree.update(out_tree)
    return [TagTracer(self, val) for val in out_vals]

  def process_map(self, map_primitive, f, tracers, params):
    raise NotImplementedError

  def post_process_call(self, call_primitive, out_tracers, params):
    raise NotImplementedError

class TagTracer(core.Tracer):
  __slots__ = ['trace', 'val']

  def __init__(self, trace, val):
    self.trace = trace
    self.val = val

  def __repr__(self):
    return 'Traced<{}:{}>'.format(self.val, self.trace)

  @property
  def aval(self):
    return core.get_aval(self.val)

  def full_lower(self):
    if isinstance(self.aval, AbstractScope):
      return self
    else:
      return self.val


@transformation
def tag_subtrace(parent, in_vals, in_tree):
  trace = parent.with_sublevel(core.cur_sublevel())
  trace.tree = in_tree
  in_tracers = [TagTracer(trace, val) for val in in_vals]
  outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  out_vals = [t.val for t in out_tracers]
  out_tree = trace.tree
  yield out_vals, out_tree

def tag_fun(fun, in_vals, in_tree, accum_fn):
  with core.new_master(TagTrace) as master:
    trace = TagTrace(master, core.cur_sublevel(), accum_fn)
    f_tag = tag_subtrace(fun, trace)
    all_args, in_treedef = tree_flatten((in_vals, in_tree))
    f_tag_flat, out_treedef = flatten_fun_nokwargs(f_tag, in_treedef)
    out_flat = f_tag_flat.call_wrapped(*all_args)
    out_vals, out_tree = tree_unflatten(out_treedef(), out_flat)
    del master
  return out_vals, out_tree
