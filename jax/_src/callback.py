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
"""Module for JAX callbacks."""
from __future__ import annotations

import functools

from typing import Any, Callable, Sequence

from jax import core
from jax import tree_util
from jax._src import lib as jaxlib
from jax._src import util
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir


# `pure_callback_p` is the main primitive for staging out Python pure callbacks.
pure_callback_p = core.Primitive("pure_callback")
pure_callback_p.multiple_results = True

map, unsafe_map = util.safe_map, map


@pure_callback_p.def_impl
def pure_callback_impl(*args, result_avals, callback: Callable[..., Any],
                       vectorized: bool):
  del vectorized, result_avals
  return callback(*args)


@pure_callback_p.def_abstract_eval
def pure_callback_abstract_eval(*avals, callback: Callable[..., Any],
                                result_avals, vectorized: bool):
  del avals, callback, vectorized
  return result_avals


def pure_callback_jvp_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError(
      "Pure callbacks do not support JVP. "
      "Please use `jax.custom_jvp` to use callbacks while taking gradients.")


ad.primitive_jvps[pure_callback_p] = pure_callback_jvp_rule


def pure_callback_transpose_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError(
      "Pure callbacks do not support transpose. "
      "Please use `jax.custom_vjp` to use callbacks while taking gradients.")

ad.primitive_transposes[pure_callback_p] = pure_callback_transpose_rule


def pure_callback_batching_rule(args, dims, *, callback, vectorized: bool,
                                result_avals: Sequence[core.ShapedArray]):
  axis_size = next(a.shape[0] for a, d in zip(args, dims)
                   if d is not batching.not_mapped)
  new_args = [arg if dim is batching.not_mapped else
              batching.moveaxis(arg, dim, 0) for arg, dim in zip(args, dims)]
  if vectorized:
    result_avals = tuple(
        core.unmapped_aval(axis_size, core.no_axis_name, 0, aval)  # type: ignore
        for aval in result_avals)
    outvals = pure_callback_p.bind(
        *new_args, callback=callback, vectorized=vectorized,
        result_avals=result_avals)
  else:
    is_batched = [d is not batching.not_mapped for d in dims]
    unbatched_args, batched_args = util.partition_list(is_batched, new_args)
    def _batch_fun(*batched_args):
      merged_args = util.merge_lists(is_batched, unbatched_args, batched_args)
      return pure_callback_p.bind(
          *merged_args, callback=callback, result_avals=result_avals,
          vectorized=vectorized)
    from jax._src.lax.control_flow import map as lax_map
    outvals = lax_map(_batch_fun, *batched_args)
  return tuple(outvals), (0,) * len(outvals)


batching.primitive_batchers[pure_callback_p] = pure_callback_batching_rule


def pure_callback_lowering(ctx, *args, callback, **params):

  if ctx.module_context.platform == "TPU" and jaxlib.version < (0, 3, 15):
    raise NotImplementedError("Pure callbacks on TPU not supported. "
                              "Please upgrade to a jaxlib >= 0.3.15.")

  def _callback(*flat_args):
    return tuple(pure_callback_p.impl(*flat_args, callback=callback, **params))

  result, _, keepalive = mlir.emit_python_callback(
      ctx, _callback, None, list(args), ctx.avals_in, ctx.avals_out, False,
      sharding=None)
  ctx.module_context.add_keepalive(keepalive)
  return result

mlir.register_lowering(pure_callback_p, pure_callback_lowering)


def pure_callback(callback: Callable[..., Any], result_shape_dtypes: Any,
                  *args: Any, vectorized: bool = False, **kwargs: Any):
  def _flat_callback(*flat_args):
    args, kwargs = tree_util.tree_unflatten(in_tree, flat_args)
    return tree_util.tree_leaves(callback(*args, **kwargs))

  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  result_avals = tree_util.tree_map(
      lambda x: core.ShapedArray(x.shape, x.dtype), result_shape_dtypes)
  flat_result_avals, out_tree = tree_util.tree_flatten(result_avals)
  out_flat = pure_callback_p.bind(
      *flat_args, callback=_flat_callback,
      result_avals=tuple(flat_result_avals), vectorized=vectorized)
  return tree_util.tree_unflatten(out_tree, out_flat)
