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
import functools

from typing import Any, Callable

from jax import core
from jax import lax
from jax import tree_util
from jax._src import lib as jaxlib
from jax._src import util
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
import jax.numpy as jnp


# `pure_callback_p` is the main primitive for staging out Python pure callbacks.
pure_callback_p = core.Primitive("pure_callback")
pure_callback_p.multiple_results = True

map, unsafe_map = util.safe_map, map


@pure_callback_p.def_impl
def pure_callback_impl(*args, result_avals, callback: Callable[..., Any]):
  del result_avals
  return callback(*args)


@pure_callback_p.def_abstract_eval
def pure_callback_abstract_eval(*avals, callback: Callable[..., Any],
                                result_avals):
  del avals, callback
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


def pure_callback_batching_rule(args, dims, *, callback, **params):
  new_args = []
  for arg, dim in zip(args, dims):
    new_args.append(jnp.rollaxis(arg, dim))
  outvals = lax.map(
      functools.partial(pure_callback_p.bind, callback=callback, **params),
      *new_args)
  return tuple(outvals), (0,) * len(outvals)


batching.primitive_batchers[pure_callback_p] = pure_callback_batching_rule


def pure_callback_lowering(ctx, *args, callback, **params):

  if ctx.module_context.platform == "TPU" and jaxlib.version < (0, 3, 15):
    raise NotImplementedError("Pure callbacks on TPU not supported. "
                              "Please upgrade to a jaxlib >= 0.3.15.")
  if isinstance(ctx.module_context.axis_context,
                (mlir.SPMDAxisContext, mlir.ShardingContext)):
    raise NotImplementedError("Sharding for pure callback not implemented.")

  def _callback(*flat_args):
    return tuple(pure_callback_p.impl(*flat_args, callback=callback, **params))

  result, _, keepalive = mlir.emit_python_callback(
      ctx, _callback, None, list(args), ctx.avals_in, ctx.avals_out, False,
      sharding=None)
  ctx.module_context.add_keepalive(keepalive)
  return result

mlir.register_lowering(pure_callback_p, pure_callback_lowering)


def pure_callback(callback: Callable[..., Any], result_shape_dtypes: Any,
                  *args: Any, **kwargs: Any):
  """Calls a pure Python callback function from staged out JAX programs.

  ``pure_callback`` enables calling a Python function in JIT-ed JAX functions.
  The input ``callback`` will be passed NumPy arrays in place of JAX arrays and
  should also return NumPy arrays. Execution takes place on the CPU host.

  The callback is treated as "pure" meaning it can be called multiple times when
  transformed (for example in a ``vmap`` or ``pmap``), and it can also
  potentially be removed from JAX programs via dead-code elimination. Pure
  callbacks can also be reordered if data-dependence allows.

  When both `pmap` and `vmap`-ed, the pure callback will be called several times
  (one on each axis of the map). In the `pmap` case, these calls will happen
  across several threads whereas in the `vmap` case, they will happen serially.

  Args:
    callback: A Python callable. The callable will be passed in NumPy arrays and
      should return a PyTree of NumPy arrays that matches
      ``result_shape_dtypes``.
    result_shape_dtypes: A PyTree of Python objects that have ``shape`` and
      ``dtype`` properties that correspond to the shape and dtypes of the
      outputs of ``callback``.
    *args: The positional arguments to the callback. Must be PyTrees of JAX
      types.
    **kwargs: The keyword arguments to the callback. Must be PyTrees of JAX
      types.

  Returns:
    The value of ``callback(*args, **kwargs)``.
  """

  def _flat_callback(*flat_args):
    args, kwargs = tree_util.tree_unflatten(in_tree, flat_args)
    return tree_util.tree_leaves(callback(*args, **kwargs))

  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  result_avals = tree_util.tree_map(
      lambda x: core.ShapedArray(x.shape, x.dtype), result_shape_dtypes)
  flat_result_avals, out_tree = tree_util.tree_flatten(result_avals)
  out_flat = pure_callback_p.bind(
      *flat_args, callback=_flat_callback, result_avals=flat_result_avals)
  return tree_util.tree_unflatten(out_tree, out_flat)
