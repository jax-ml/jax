# Copyright 2021 Google LLC
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

from typing import Callable, Optional, Sequence

from jax import core
from jax import linear_util as lu
from jax.interpreters import batching
from jax.interpreters.batching import not_mapped
from jax.interpreters import partial_eval as pe
from jax.tree_util import tree_flatten, tree_unflatten
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import util
from jax._src.api_util import flatten_fun, flatten_fun_nokwargs


source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


class custom_vmap:
  fun: Callable
  vmap_rule: Optional[Callable]

  def __init__(self, fun: Callable) -> None:
    self.fun = fun  # type: ignore[assignment]
    self.vmap_rule = None

  def def_vmap(self, vmap_rule: Callable) -> None:
    self.vmap_rule = vmap_rule

  @traceback_util.api_boundary
  def __call__(self, *args, **kwargs):
    assert not kwargs
    args_flat, in_tree = tree_flatten(args)
    flat_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(self.fun), in_tree)
    in_avals = [core.raise_to_shaped(core.get_aval(x)) for x in args_flat]
    debug = pe.debug_info(self.fun, in_tree, False, "custom_vmap")
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals, debug)
    assert not len(consts)
    out_flat = custom_vmap_p.bind(*consts, *args_flat,
                                  call=pe.convert_constvars_jaxpr(jaxpr),
                                  rule=self.vmap_rule,
                                  in_tree=in_tree)
    return tree_unflatten(out_tree(), out_flat)


### utils


def ensure_list(xs):
  return xs if type(xs) is list else list(xs)

def rule_name(rule):
  return getattr(rule, '__name__', '<unnamed rule>')

def call_rule(rule, axis_size, in_batched, args):
  outs, out_batched = rule(axis_size, ensure_list(in_batched), *args)
  if not isinstance(outs, Sequence):
    raise TypeError(
        'custom vmap rule output values must be a sequence, '
        f'rule ({rule_name(rule)}) returned {type(outs)}')
  if not isinstance(out_batched, Sequence):
    raise TypeError(
        'custom vmap rule output batching specification must be a sequence, '
        f'rule ({rule_name(rule)}) returned {type(out_batched)}')
  return ensure_list(outs), ensure_list(out_batched)

def check_vmap_rule_trees(rule, out_tree, out_batched_tree):
  if out_tree != out_batched_tree:
    raise ValueError(
        'structure of output values and output batching specification returned '
        f'by custom vmap rule ({rule_name(rule)}) do not match.\n'
        f'Output values: {out_tree}\n'
        f'Batching spec: {out_batched_tree}')

# Like batching.bdim_at_front, but doesn't broadcast if not mapped
def maybe_bdim_at_front(x, bdim):
  if core.get_aval(x) is core.abstract_unit:
    return core.unit
  if bdim is not_mapped:
    return x
  else:
    return util.moveaxis(x, bdim, 0)


### custom_vmap_p rules


def custom_vmap_impl(*args, call, rule, in_tree):
  del rule, in_tree
  return core.eval_jaxpr(call, (), *args)


def custom_vmap_batching(args_flat, dims, *, call, rule, in_tree):
  del call
  axis_size, = {x.shape[d] for x, d in zip(args_flat, dims) if d is not None}
  args_flat = map(maybe_bdim_at_front, args_flat, dims)
  flat_in_batched = [d is not not_mapped for d in dims]

  args = tree_unflatten(in_tree, args_flat)
  in_batched = tree_unflatten(in_tree, flat_in_batched)
  outs, out_batched = call_rule(rule, axis_size, in_batched, args)
  flat_outs, tree1 = tree_flatten(outs)
  flat_out_batched, tree2 = tree_flatten(out_batched,
                                         is_leaf=lambda x: x is None)
  check_vmap_rule_trees(rule, tree1, tree2)
  flat_out_dims = [0 if b else not_mapped for b in flat_out_batched]
  return flat_outs, flat_out_dims


custom_vmap_p = core.Primitive('custom_vmap_call')
custom_vmap_p.multiple_results = True
custom_vmap_p.def_impl(custom_vmap_impl)
batching.primitive_batchers[custom_vmap_p] = custom_vmap_batching
