# Copyright 2022 Google LLC
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

import functools
from typing import Callable, Optional

from jax import core
from jax import linear_util as lu
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.tree_util import (tree_flatten, tree_leaves, tree_unflatten,
                           treedef_tuple)
from jax._src import ad_util
from jax._src import custom_api_util
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import util
from jax._src.api_util import flatten_fun_nokwargs


source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


@custom_api_util.register_custom_decorator_type
class custom_transpose:
  fun: Callable
  transpose: Optional[Callable]

  def __init__(self, fun: Callable):
    functools.update_wrapper(self, fun)
    self.fun = fun  # type: ignore[assignment]
    self.transpose = None

  __getattr__ = custom_api_util.forward_attr

  def def_transpose(self, transpose: Callable):
    self.transpose = transpose
    return transpose

  @traceback_util.api_boundary
  def __call__(self, residual_arg, linear_arg):
    res_arg, lin_arg = residual_arg, linear_arg
    _, res_tree = tree_flatten(res_arg)
    _, lin_tree = tree_flatten(lin_arg)
    args_flat, in_tree = tree_flatten((res_arg, lin_arg))

    flat_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(self.fun), in_tree)
    in_avals = [core.raise_to_shaped(core.get_aval(x)) for x in args_flat]
    debug = pe.debug_info(self.fun, in_tree, False, "custom_transpose")
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals, debug)
    assert not len(consts)
    closed_call = core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())
    out_flat = custom_transpose_p.bind(*consts, *args_flat,
                                       call=closed_call,
                                       rule=self.transpose,
                                       lin_tree=lin_tree,
                                       res_tree=res_tree,
                                       out_tree=out_tree())
    return tree_unflatten(out_tree(), out_flat)


### utils

def rule_name(rule):
  return getattr(rule, '__name__', '<unnamed transpose rule>')

def check_transpose_rule_trees(rule, lin_tree, rule_out_tree):
  if lin_tree != rule_out_tree and len(lin_tree.children()) == 1:
    lin_tree2, = lin_tree.children()
  else:
    lin_tree2 = lin_tree
  if lin_tree2 != rule_out_tree:
    raise ValueError(
        'structure of custom transpose rule\'s output does not match '
        'structure of primal function\'s linear inputs under '
        f'custom transpose rule ({rule_name(rule)}).\n'
        f'Transpose rule output: {rule_out_tree}\n'
        f'Linear primal inputs: {lin_tree}')


### custom_transpose_p rules


def custom_transpose_impl(*args, call, rule, res_tree, lin_tree, out_tree):
  del rule, res_tree, lin_tree, out_tree
  return core.jaxpr_as_fun(call)(*args)


def custom_transpose_transpose_rule(
    cts, *args, call, rule, res_tree, lin_tree, out_tree):
  call_in_tree = treedef_tuple((res_tree, lin_tree))

  res_arg, lin_arg = tree_unflatten(call_in_tree, args)
  assert all(ad.is_undefined_primal(x)     for x in tree_leaves(lin_arg))
  assert all(not ad.is_undefined_primal(x) for x in tree_leaves(res_arg))

  cts = [ad_util.zeros_like_aval(ct_aval) if type(ct) is ad_util.Zero else ct
         for ct, ct_aval in zip(cts, call.out_avals)]
  ct_out = tree_unflatten(out_tree, cts)
  ct_lin = rule(res_arg, ct_out)
  ct_lin_flat, ct_lin_tree = tree_flatten(ct_lin)
  check_transpose_rule_trees(rule, lin_tree, ct_lin_tree)
  return [None] * len(tree_leaves(res_arg)) + ct_lin_flat


def custom_transpose_abstract_eval(*in_avals, call, **_):
  return call.out_avals


custom_transpose_p = core.Primitive('custom_transpose_call')
custom_transpose_p.multiple_results = True
custom_transpose_p.def_impl(custom_transpose_impl)
custom_transpose_p.def_abstract_eval(custom_transpose_abstract_eval)
ad.primitive_transposes[custom_transpose_p] = custom_transpose_transpose_rule
xla.register_translation(custom_transpose_p,
                         xla.lower_fun(custom_transpose_impl, new_style=True,
                                       multiple_results=True),
                         initial_style=True)
mlir.register_lowering(custom_transpose_p, mlir.lower_fun(
    custom_transpose_impl, multiple_results=True))
