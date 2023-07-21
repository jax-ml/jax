# Copyright 2022 The JAX Authors.
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

from collections.abc import Sequence
import functools
from typing import Any, Callable

import numpy as np

from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import sharding_impls
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lib import xla_client as xc
from jax._src.lax.control_flow.loops import map as lax_map

# `pure_callback_p` is the main primitive for staging out Python pure callbacks.
pure_callback_p = core.Primitive("pure_callback")
pure_callback_p.multiple_results = True

map, unsafe_map = util.safe_map, map


def pure_callback_impl(*args, result_avals, callback: Callable[..., Any],
                       vectorized: bool):
  del vectorized, result_avals
  return callback(*args)
pure_callback_p.def_impl(functools.partial(dispatch.apply_primitive,
                                           pure_callback_p))


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
    def _batch_fun(batched_args):
      merged_args = util.merge_lists(is_batched, unbatched_args, batched_args)
      return pure_callback_p.bind(
          *merged_args, callback=callback, result_avals=result_avals,
          vectorized=vectorized)
    outvals = lax_map(_batch_fun, batched_args)
  return tuple(outvals), (0,) * len(outvals)


batching.primitive_batchers[pure_callback_p] = pure_callback_batching_rule


def pure_callback_lowering(ctx, *args, callback, **params):

  def _callback(*flat_args):
    return tuple(pure_callback_impl(*flat_args, callback=callback, **params))

  sharding = None
  axis_context = ctx.module_context.axis_context
  if isinstance(axis_context, sharding_impls.SPMDAxisContext):
    # If we have fully manual sharding during lowering, that means the JAX
    # program has per-device semantics, so we run the callback on each device.
    if axis_context.manual_axes != frozenset(axis_context.mesh.axis_names):
      raise NotImplementedError(
          "pure_callback is only supported in spmd computations when all mesh"
          " axes are partitioned manually (no partial automatic sharding)."
      )
    sharding = xc.OpSharding()
    sharding.type = xc.OpSharding.Type.MANUAL
  elif isinstance(axis_context, sharding_impls.ShardingContext):
    # If we have fully automatic sharding during lowering, that means the JAX
    # program has bulk array semantics, so we run the callback with a MAXIMAL
    # sharding and hence execute it only once on the full logical value).
    sharding = xc.OpSharding()
    sharding.type = xc.OpSharding.Type.MAXIMAL
    sharding.tile_assignment_dimensions = [1]
    sharding.tile_assignment_devices = [0]
  else:
    # When there's no SPMD partitioning going on, don't annotate a sharding.
    sharding = None
  if isinstance(axis_context, sharding_impls.SPMDAxisContext):
    if axis_context.manual_axes != frozenset(axis_context.mesh.axis_names):
      raise NotImplementedError(
          "pure_callback is only supported in spmd computations when all mesh"
          " axes are partitioned manually (no partial automatic sharding)."
      )
    sharding = xc.OpSharding()
    sharding.type = xc.OpSharding.Type.MANUAL

  result, _, keepalive = mlir.emit_python_callback(
      ctx, _callback, None, list(args), ctx.avals_in, ctx.avals_out, False,
      sharding=sharding)
  ctx.module_context.add_keepalive(keepalive)
  return result

mlir.register_lowering(pure_callback_p, pure_callback_lowering)

def _check_shape_dtype(shape_dtype):
  dt = np.dtype(shape_dtype.dtype)
  if dtypes.canonicalize_dtype(dt) != dt:
    raise ValueError(
        "Cannot return 64-bit values when `jax_enable_x64` is disabled")

def pure_callback(callback: Callable[..., Any], result_shape_dtypes: Any,
                  *args: Any, vectorized: bool = False, **kwargs: Any):
  """Calls a pure Python callback.

  For more explanation, see `External Callbacks`_.

  Args:
    callback: function to execute on the host. The callback is assumed to be a pure
      function (i.e. one without side-effects): if an impure function is passed, it
      may behave in unexpected ways, particularly under transformation.
    result_shape_dtypes: pytree whose leaves have ``shape`` and ``dtype`` attributes,
      whose structure matches the expected output of the callback function at runtime.
    *args: arguments to be passed to the callback function
    vectorized: boolean specifying whether the callback function can operate in a
      vectorized manner.
    **kwargs: keyword arguments to be passed to the callback function

  Returns:
    result: a pytree of :class:`jax.Array` objects whose structure matches that of
      ``result_shape_dtypes``.

  See Also:
    - :func:`jax.experimental.io_callback`: callback designed for impure functions.
    - :func:`jax.debug.callback`: callback designed for general-purpose debugging.
    - :func:`jax.debug.print`: callback designed for printing.

  .. _External Callbacks: https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html
  """
  def _flat_callback(*flat_args):
    args, kwargs = tree_util.tree_unflatten(in_tree, flat_args)
    return tree_util.tree_leaves(callback(*args, **kwargs))

  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  tree_util.tree_map(_check_shape_dtype, result_shape_dtypes)
  result_avals = tree_util.tree_map(
      lambda x: core.ShapedArray(x.shape, x.dtype), result_shape_dtypes)
  flat_result_avals, out_tree = tree_util.tree_flatten(result_avals)
  out_flat = pure_callback_p.bind(
      *flat_args, callback=_flat_callback,
      result_avals=tuple(flat_result_avals), vectorized=vectorized)
  return tree_util.tree_unflatten(out_tree, out_flat)



def pure_callback_api(callback: Callable[..., Any], result_shape_dtypes: Any,
                      *args: Any, vectorized: bool = False, **kwargs: Any):
  """Applies a functionally pure Python callable. Works under :func:`jit`/:func:`~pmap`/etc.

  ``pure_callback`` enables calling a Python function in JIT-ed JAX functions.
  The input ``callback`` will be passed NumPy arrays in place of JAX arrays and
  should also return NumPy arrays. Execution takes place on CPU, like any
  Python+NumPy function.

  The callback is treated as functionally pure, meaning it has no side-effects
  and its output value depends only on its argument values. As a consequence, it
  is safe to be called multiple times (e.g. when transformed by :func:`~vmap` or
  :func:`~pmap`), or not to be called at all when e.g. the output of a
  `jit`-decorated function has no data dependence on its value. Pure callbacks
  may also be reordered if data-dependence allows.

  When :func:`~pmap`-ed, the pure callback will be called several times (one on each
  axis of the map). When `vmap`-ed the behavior will depend on the value of the
  ``vectorized`` keyword argument. When ``vectorized`` is ``True``, the callback
  is assumed to obey
  ``jax.vmap(callback)(xs) == callback(xs) == jnp.stack([callback(x) for x in xs])``.
  Therefore, the callback will be called directly on batched inputs (where the
  batch axes are the leading dimensions). Additionally, the callbacks should
  return outputs that have corresponding leading batch axes. If not vectorized
  ``callback`` will be mapped sequentially across the batched axis.
  For example, if ``callback = lambda x, y: np.matmul(x, y)``, then we are free
  to set ``vectorized=True`` because the ``np.matmul`` function handles
  arbitrary leading batch dimensions.

  Args:
    callback: A Python callable. The callable will be passed PyTrees of NumPy
      arrays as arguments, and should return a PyTree of NumPy arrays that
      matches ``result_shape_dtypes``.
    result_shape_dtypes: A PyTree with leaves that are objects with ``shape``
      and ``dtype`` attributes which represent to the shapes and dtypes of the
      value of ``callback`` applied to ``args`` and ``kwargs``.
    *args: The positional arguments to the callback. Must be PyTrees of JAX
      types.
    vectorized: A boolean that indicates whether or not ``callback`` is
      vectorized, meaning it can handle arrays with additional leading
      dimensions. If ``vectorized`` is `True`, when the callback is mapped
      via `jax.vmap`, it will be called directly on inputs with leading batch
      dimensions instead of executing ``callback`` on each mapped input
      individually. The callback should also return outputs batched across the
      leading axis. By default, ``vectorized`` is ``False``.
    **kwargs: The keyword arguments to the callback. Must be PyTrees of JAX
      types.

  Returns:
    The value of ``callback(*args, **kwargs)``.
  """
  return pure_callback(callback, result_shape_dtypes, *args,
                       vectorized=vectorized, **kwargs)


# IO Callback

io_callback_p = core.Primitive("io_callback")
io_callback_p.multiple_results = True

class IOEffect(effects.Effect):
  __str__ = lambda _: "IO"

class OrderedIOEffect(effects.Effect):
  __str__ = lambda _: "OrderedIO"

_IOEffect = IOEffect()
_OrderedIOEffect = OrderedIOEffect()
effects.lowerable_effects.add_type(IOEffect)
effects.lowerable_effects.add_type(OrderedIOEffect)
effects.control_flow_allowed_effects.add_type(IOEffect)
effects.control_flow_allowed_effects.add_type(OrderedIOEffect)
effects.ordered_effects.add_type(OrderedIOEffect)


def io_callback_impl(*args, result_avals, callback: Callable[..., Any],
                     ordered: bool):
  del result_avals, ordered
  return callback(*args)
io_callback_p.def_impl(functools.partial(dispatch.apply_primitive,
                                         io_callback_p))

@io_callback_p.def_effectful_abstract_eval
def io_callback_abstract_eval(*avals, callback: Callable[..., Any],
                              result_avals, ordered: bool):
  del avals, callback
  effect = _OrderedIOEffect if ordered else _IOEffect
  return result_avals, {effect}

def io_callback_jvp_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError("IO callbacks do not support JVP.")
ad.primitive_jvps[io_callback_p] = io_callback_jvp_rule


def io_callback_transpose_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError("IO callbacks do not support transpose.")
ad.primitive_transposes[io_callback_p] = io_callback_transpose_rule


def io_callback_batching_rule(args, dims, callback, result_avals, ordered):
  if ordered:
    raise ValueError("Cannot `vmap` ordered IO callback.")
  return pure_callback_batching_rule(args, dims, callback=callback,
      vectorized=False, result_avals=result_avals)
batching.primitive_batchers[io_callback_p] = io_callback_batching_rule

def io_callback_lowering(ctx, *args, callback, ordered, **params):

  def _callback(*flat_args):
    return tuple(io_callback_impl(*flat_args, callback=callback,
                                  ordered=ordered, **params))

  # TODO(sharadmv): figure out the best API for sharding callbacks. For now, we
  # can only safely maximally shard. Should we allow device_index to be passed
  # in like host_callback?
  if isinstance(ctx.module_context.axis_context,
                (sharding_impls.SPMDAxisContext, sharding_impls.ShardingContext)):
    # Apply maximal sharding so pjit only executes the callback on device 0.
    sharding = xc.OpSharding()
    sharding.type = xc.OpSharding.Type.MAXIMAL
    sharding.tile_assignment_dimensions = [1]
    sharding.tile_assignment_devices = [0]
  else:
    sharding = None

  if ordered:
    token = ctx.tokens_in.get(_OrderedIOEffect)[0]
    result, token, keepalive = mlir.emit_python_callback(
        ctx, _callback, token, list(args), ctx.avals_in, ctx.avals_out, True,
        sharding=sharding)
    ctx.set_tokens_out(mlir.TokenSet({_OrderedIOEffect: (token,)}))
  else:
    result, token, keepalive = mlir.emit_python_callback(
        ctx, _callback, None, list(args), ctx.avals_in, ctx.avals_out, True,
        sharding=sharding)
  ctx.module_context.add_keepalive(keepalive)
  return result
mlir.register_lowering(io_callback_p, io_callback_lowering)

def io_callback(callback: Callable[..., Any], result_shape_dtypes: Any,
                *args: Any, ordered: bool = False, **kwargs: Any):
  """Calls an impure Python callback.

  For more explanation, see `External Callbacks`_.

  Args:
    callback: function to execute on the host. It is assumet to be an impure function.
      If ``callback`` is pure, using :func:`jax.pure_callback` instead may lead to
      more efficient execution.
    result_shape_dtypes: pytree whose leaves have ``shape`` and ``dtype`` attributes,
      whose structure matches the expected output of the callback function at runtime.
    *args: arguments to be passed to the callback function
    ordered: boolean specifying whether sequential calls to callback must be ordered.
    **kwargs: keyword arguments to be passed to the callback function

  Returns:
    result: a pytree of :class:`jax.Array` objects whose structure matches that of
      ``result_shape_dtypes``.

  See Also:
    - :func:`jax.pure_callback`: callback designed for pure functions.
    - :func:`jax.debug.callback`: callback designed for general-purpose debugging.
    - :func:`jax.debug.print`: callback designed for printing.

  .. _External Callbacks: https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html
  """
  def _flat_callback(*flat_args):
    args, kwargs = tree_util.tree_unflatten(in_tree, flat_args)
    return tree_util.tree_leaves(callback(*args, **kwargs))

  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  tree_util.tree_map(_check_shape_dtype, result_shape_dtypes)
  flat_shape_dtypes, out_tree = tree_util.tree_flatten(result_shape_dtypes)
  flat_result_avals = map(lambda x: core.ShapedArray(x.shape, x.dtype),
                          flat_shape_dtypes)
  flat_args = map(core.raise_as_much_as_possible, flat_args)
  out_flat = io_callback_p.bind(
      *flat_args, callback=_flat_callback,
      result_avals=tuple(flat_result_avals),
      ordered=ordered)
  return tree_util.tree_unflatten(out_tree, out_flat)
