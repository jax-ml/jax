# Copyright 2025 The JAX Authors.
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

"""Fusible primitive."""
import functools
from typing import Any

import jax
from jax._src import api_util
from jax._src import core as jax_core
from jax._src import linear_util as lu
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.pallas.fuser import block_spec
from jax._src.pallas.fuser import fusion as fusion_lib

fusible_p = jax_core.Primitive('fusible')
fusible_p.multiple_results = True


def _make_trivial_fusion(x: jax.Array) -> fusion_lib.Fusion:
  return fusion_lib.Fusion(
      func=lambda: x,
      in_type=((), {}),
      out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
  )


def fusible(f=None, *, output_fusion_prefix: Any = True):
  def decorator(f):
    def wrapper(*args):
      def wrapped(*args):
        in_fusions = tree_util.tree_map(_make_trivial_fusion, args)
        return f(*in_fusions, None)

      flat_args, in_tree = tree_util.tree_flatten(args)
      debug_info = api_util.debug_info('fusible', wrapped, args, {})
      flat_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
          lu.wrap_init(wrapped, debug_info=debug_info), in_tree
      )
      flat_avals = [jax_core.get_aval(x) for x in flat_args]
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, flat_avals)
      out_tree = out_tree_thunk()
      out = fusible_p.bind(
          *consts,
          *flat_args,
          jaxpr=jaxpr,
          num_consts=len(consts),
          in_tree=in_tree,
          out_tree=out_tree,
          func=f,
          output_fusion_prefix=output_fusion_prefix,
      )
      return tree_util.tree_unflatten(out_tree, out)

    return wrapper

  if f is not None:
    return decorator(f)
  return decorator


@fusible_p.def_impl
def _(*consts_and_args, jaxpr, num_consts, **_):
  consts, args = util.split_list(consts_and_args, [num_consts])
  return jax_core.eval_jaxpr(jaxpr, consts, *args)


mlir.register_lowering(fusible_p, mlir.lower_fun(fusible_p.impl))


@fusible_p.def_effectful_abstract_eval
def _(*args, jaxpr, **kwargs):
  del args, kwargs
  return [v.aval for v in jaxpr.outvars], jaxpr.effects


def _unbatch(fusion: fusion_lib.Fusion, axis: int | None, axis_size: int):
  """Returns a new function with `axis` removed and infers axes for values."""
  args, kwargs = fusion.in_type
  assert not kwargs
  if not isinstance(args, (tuple, list)):
    args = (args,)

  fn, values, prefetch = block_spec.get_fusion_values(fusion, *args)
  if prefetch:
    raise NotImplementedError('Prefetch is not supported yet.')
  if axis is None:
    return fn, values, None

  def get_spec(x):
    block_shape = list(x.shape)
    block_shape[axis] = None

    def index_map(batch_idx):
      idxs = [0] * x.ndim
      idxs[axis] = batch_idx
      return tuple(idxs)

    return pallas_core.BlockSpec(block_shape, index_map)

  if args:
    arg_specs = tuple(map(get_spec, args))
    out_spec = block_spec.push_block_spec(fusion, *arg_specs)(*args)
  else:
    out_spec = tree_util.tree_map(get_spec, fusion.out_type)

  spec_puller = block_spec.pull_block_spec(fn, out_spec, grid=(axis_size,))
  fn, (value_specs, *_), _ = spec_puller(values, *args)

  def get_batch_axis(spec):
    batch_idx = object()
    idxs = spec.index_map(batch_idx)
    return idxs.index(batch_idx) if batch_idx in idxs else None

  value_axes = tree_util.tree_map(get_batch_axis, value_specs)
  return fn, values, value_axes


def _fusible_batching_rule(
    axis_data, args, in_axes, *, func, output_fusion_prefix, in_tree, **_
):
  def new_func(*fusions):
    unbatch = functools.partial(_unbatch, axis_size=axis_data.size)
    *in_fusions, out_fusions = fusions
    in_fns, in_values, in_value_axes = zip(*map(unbatch, in_fusions, in_axes))

    if out_fusions is None:
      out_fns = out_values = out_value_axes = None
    else:
      unbatch_out = functools.partial(unbatch, axis=0)
      unbatched_outs = tree_util.tree_map(unbatch_out, out_fusions)
      out_tree = tree_util.tree_structure(out_fusions)
      out_fns, out_values, out_value_axes = map(
          out_tree.unflatten, zip(*out_tree.flatten_up_to(unbatched_outs))
      )

    axis_name = object()

    def unbatched(in_values, out_values):
      def get_fusion(fn, values, *arg_types):
        def fn_closed(*args):
          return fn((jax.lax.axis_index(axis_name),), (), values, *args)

        in_type = (arg_types, {})
        out_type = jax.eval_shape(fn_closed, *arg_types)
        return fusion_lib.Fusion(fn_closed, in_type=in_type, out_type=out_type)

      new_in_fusions = in_tree.unflatten(map(get_fusion, in_fns, in_values))

      if out_fusions is None:
        new_out_fusions = None
      else:
        out_type = jax.eval_shape(lambda: func(*new_in_fusions, None))
        new_out_fusions = tree_util.tree_map(
            get_fusion, out_fns, out_values, out_type
        )

      return func(*new_in_fusions, new_out_fusions)

    in_axes_ = (in_value_axes, out_value_axes)
    batched = jax.vmap(unbatched, in_axes=in_axes_, axis_name=axis_name)
    return batched(in_values, out_values)

  out = fusible(new_func, output_fusion_prefix=output_fusion_prefix)(*args)
  out_flat = tree_util.tree_leaves(out)
  return out_flat, (0,) * len(out_flat)


batching.fancy_primitive_batchers[fusible_p] = _fusible_batching_rule
