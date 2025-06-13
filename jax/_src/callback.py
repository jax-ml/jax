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

from collections.abc import Callable, Sequence
import dataclasses
import functools
import logging
from typing import Any

from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import ffi
from jax._src import pickle_util
from jax._src import sharding_impls
from jax._src import tree_util
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import xla
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.sharding_impls import SdyArray, SdyArrayList, SdyDim, SingleDeviceSharding
from jax._src.typing import Array, DeprecatedArg
import numpy as np

logger = logging.getLogger(__name__)

# `pure_callback_p` is the main primitive for staging out Python pure callbacks.
pure_callback_p = core.Primitive("pure_callback")
pure_callback_p.multiple_results = True
dispatch.prim_requires_devices_during_lowering.add(pure_callback_p)

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


@dataclasses.dataclass(frozen=True)
class _FlatCallback:
  """A Python function callable with flat arguments and results.

  An instance of this class is used as a parameter for the callback primitives.
  We prefer it to an anonymous flattened function because it produces
  equal objects when we call the same Python function with the same argument
  structure.
  """
  callback_func: Callable[..., Any]
  in_tree: tree_util.PyTreeDef  # (args, kwargs) pytree for `callback_func`.

  def __call__(self, *flat_args: Array) -> Sequence[Array]:
    args, kwargs = tree_util.tree_unflatten(self.in_tree, flat_args)
    return tree_util.tree_leaves(self.callback_func(*args, **kwargs))


def pure_callback_impl(
    *args,
    result_avals,
    callback: _FlatCallback,
    sharding: SingleDeviceSharding | None,
    vmap_method: str | None,
):
  del sharding, vmap_method, result_avals
  try:
    cpu_device, *_ = xb.local_devices(backend="cpu")
  except RuntimeError as e:
    raise RuntimeError(
        "jax.pure_callback failed to find a local CPU device to place the"
        " inputs on. Make sure \"cpu\" is listed in --jax_platforms or the"
        " JAX_PLATFORMS environment variable."
    ) from e
  args = api.device_put(args, cpu_device)
  with config.default_device(cpu_device):
    try:
      return tree_util.tree_map(np.asarray, callback(*args))
    except BaseException:
      logger.exception("jax.pure_callback failed")
      raise


pure_callback_p.def_impl(functools.partial(dispatch.apply_primitive,
                                           pure_callback_p))


@pure_callback_p.def_abstract_eval
def pure_callback_abstract_eval(
    *avals,
    callback: _FlatCallback,
    result_avals,
    sharding: SingleDeviceSharding | None,
    vmap_method: str | None,
):
  del avals, callback, sharding, vmap_method
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


batching.primitive_batchers[pure_callback_p] = functools.partial(
    ffi.ffi_batching_rule, pure_callback_p
)


def _callback_op_sharding(
    axis_context, sharding: SingleDeviceSharding | None, avals_out
):
  if isinstance(axis_context, sharding_impls.SPMDAxisContext):
    # If we have fully manual sharding during lowering, that means the JAX
    # program has per-device semantics, so we run the callback on each device.
    if axis_context.manual_axes != frozenset(axis_context.mesh.axis_names):
      raise NotImplementedError(
          "callbacks are only supported in spmd computations when all mesh"
          " axes are partitioned manually (no partial automatic sharding)."
      )
    if sharding is not None:
      raise NotImplementedError(
          "callbacks do not support specifying sharding inside spmd"
          " computations"
      )
    if config.use_shardy_partitioner.value:
      ndim = 0
      if avals_out and isinstance(avals_out[0], core.ShapedArray):
        ndim = avals_out[0].ndim
      op_sharding = SdyArrayList([
          SdyArray(
              mesh_shape=(),
              dim_shardings=[
                  SdyDim(axes=[], is_open=False)
              ] * ndim,
              logical_device_ids=())])
    else:
      op_sharding = xc.OpSharding()  # type: ignore[assignment]
      op_sharding.type = xc.OpSharding.Type.MANUAL
    return op_sharding

  if isinstance(axis_context, sharding_impls.ShardingContext):
    if sharding is not None:
      if not isinstance(sharding, SingleDeviceSharding):
        raise NotImplementedError(
            "pure_callback only supports SingleDeviceSharding, but got"
            f" {type(sharding)}"
        )
      device = next(iter(sharding.device_set))
      device_assignment = axis_context.device_assignment
      if device_assignment is None:
        raise AssertionError(
            "Please file a bug at https://github.com/jax-ml/jax/issues")
      try:
        device_index = device_assignment.index(device)
      except IndexError as e:
        raise ValueError(
            "Sharding provided to pure_callback specifies a device"
            f" {device} that is not in the device assignment"
            f" ({device_assignment})") from e
    else:
      device_index = 0

    # If we have fully automatic sharding during lowering, that means the JAX
    # program has bulk array semantics, so we run the callback with a MAXIMAL
    # sharding and hence execute it only once on the full logical value).
    if config.use_shardy_partitioner.value:
      # For shardy, we need to have the same number of shardy annotations as the
      # number of result ops. If there are no result ops, we need 1 shardy
      # annotation.
      num_sdy_shardings = max(1, len(avals_out))
      op_sharding = SdyArrayList(num_sdy_shardings * [
          SdyArray(
              mesh_shape=(),
              dim_shardings=[],
              logical_device_ids=(device_index,))])
    else:
      op_sharding = xc.OpSharding()  # type: ignore[assignment]
      op_sharding.type = xc.OpSharding.Type.MAXIMAL
      op_sharding.tile_assignment_dimensions = [1]
      op_sharding.tile_assignment_devices = [device_index]
    return op_sharding

  # When there's no SPMD partitioning going on, don't annotate a sharding.
  return None


def pure_callback_lowering(
    ctx, *args, callback: _FlatCallback, sharding: SingleDeviceSharding | None, **params
):
  def _callback(*flat_args):
    return tuple(
        pure_callback_impl(
            *flat_args,
            callback=callback,
            sharding=None,  # unused.
            **params,
        )
    )

  op_sharding = _callback_op_sharding(
      ctx.module_context.axis_context, sharding, ctx.avals_out)
  result, _, _ = emit_python_callback(
      ctx,
      _callback,
      None,
      list(args),
      ctx.avals_in,
      ctx.avals_out,
      has_side_effect=False,
      sharding=op_sharding,
  )
  return result


mlir.register_lowering(pure_callback_p, pure_callback_lowering)

def _check_shape_dtype(shape_dtype):
  dt = np.dtype(shape_dtype.dtype)
  if dtypes.canonicalize_dtype(dt) != dt:
    raise ValueError(
        "result_shape_dtypes cannot specify 64-bit types when `jax_enable_x64` is disabled")


def pure_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: Any,
    *args: Any,
    sharding: SingleDeviceSharding | None = None,
    vectorized: bool | None | DeprecatedArg = DeprecatedArg(),
    vmap_method: str | None = None,
    **kwargs: Any,
):
  """Calls a pure Python callback. Works under :func:`jit`/:func:`~vmap`/etc.

  For more explanation, see `External Callbacks`_.

  ``pure_callback`` enables calling a Python function in JIT-ed JAX functions.
  The input ``callback`` will be passed JAX arrays placed on a local CPU, and
  it should also return JAX arrays on CPU.

  The callback is treated as functionally pure, meaning it has no side-effects
  and its output value depends only on its argument values. As a consequence, it
  is safe to be called multiple times (e.g. when transformed by :func:`~vmap` or
  :func:`~pmap`), or not to be called at all when e.g. the output of a
  `jit`-decorated function has no data dependence on its value. Pure callbacks
  may also be reordered if data-dependence allows.

  .. warning::

     In the context of JAX transformations, Python exceptions should be
     considered side-effects: this means that intentionally raising an error
     within a `pure_callback` breaks the API contract, and the behavior of
     the resulting program is undefined.

  When `vmap`-ed the behavior will depend on the value of the ``vmap_method``.

  * Calling :func:`~jax.vmap` on a callback without an explicit ``vmap_method``
    raises a ``NotImplementedError``.
  * ``vmap_method="sequential"`` uses :func:`~jax.lax.map` to loop over
    the batched arguments, calling ``callback`` once for each batch element.
  * ``vmap_method="sequential_unrolled"`` is like ``sequential``, but the loop
    is unrolled.
  * ``vmap_method="expand_dims"`` calls ``callback`` with new axes of size ``1``
    added as the leading dimension unbatched inputs.
  * ``vmap_method="broadcast_all"`` behaves like ``expand_dims``, but the
    inputs are tiled to the expected batched shape.

  If necessary, the legacy behavior provided by the removed ``vectorized=True``
  argument can be recovered using ``vmap_method="legacy_vectorized"``.

  The current default behavior is to use ``vmap_method="sequential"`` when
  not specified, but this behavior is deprecated, and in the future, the
  default will be to raise a ``NotImplementedError`` unless ``vmap_method`` is
  explicitly specified.

  Args:
    callback: function to execute on the host. The callback is assumed to be a pure
      function (i.e. one without side-effects): if an impure function is passed, it
      may behave in unexpected ways, particularly under transformation. The callable
      will be passed PyTrees of arrays as arguments, and should return a PyTree of
      arrays that matches ``result_shape_dtypes``.
    result_shape_dtypes: pytree whose leaves have ``shape`` and ``dtype`` attributes,
      whose structure matches the expected output of the callback function at runtime.
      :class:`jax.ShapeDtypeStruct` is often used to define leaf values.
    *args: arguments to be passed to the callback function
    sharding: optional sharding that specifies the device from which the callback should
      be invoked.
    vmap_method: string specifying how the callback transforms under
      :func:`~jax.vmap` as described above.
    **kwargs: keyword arguments to be passed to the callback function

  Returns:
    result: a pytree of :class:`jax.Array` objects whose structure matches that of
      ``result_shape_dtypes``.

  See Also:
    - :func:`jax.experimental.io_callback`: callback designed for impure functions.
    - :func:`jax.debug.callback`: callback designed for general-purpose debugging.
    - :func:`jax.debug.print`: callback designed for printing.

  Examples:
    The behavior of ``pure_callback`` under :func:`~jax.vmap` is controlled by
    the ``vmap_method`` argument as described above. It is useful to consider
    some explicit examples that demonstrate the semantics. For example,
    consider the following function:

    >>> def callback(x, y):
    ...   print(jnp.shape(x), jnp.shape(y))
    ...   return x + y

    >>> def fun(x, y, *, vmap_method):
    ...   shape = jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y))
    ...   dtype = jnp.result_type(x, y)
    ...   out_type = jax.ShapeDtypeStruct(shape, dtype)
    ...   return jax.pure_callback(callback, out_type, x, y,
    ...                            vmap_method=vmap_method)

    Calling this with ``vmap_method="expand_dims"`` adds a new axis of size ``1``
    to ``y``:

    >>> from functools import partial
    >>> x = jnp.arange(4)
    >>> y = 1.0
    >>> jax.vmap(partial(fun, vmap_method="expand_dims"), in_axes=(0, None))(x, y)
    (4,) (1,)
    Array([1., 2., 3., 4.], dtype=float32)

    Whereas, ``vmap_method="broadcast_all"`` adds an axis of size ``4`` to
    ``y``:

    >>> jax.vmap(partial(fun, vmap_method="broadcast_all"),
    ...          in_axes=(0, None))(x, y)
    (4,) (4,)
    Array([1., 2., 3., 4.], dtype=float32)

  .. _External Callbacks: https://docs.jax.dev/en/latest/external-callbacks.html
  """
  # TODO(danfm): Remove this check 3 months after v0.6.0 is released.
  if not isinstance(vectorized, DeprecatedArg):
    raise ValueError(
        "The 'vectorized' argument of jax.pure_callback was removed in JAX "
        "v0.6.0. Use 'vmap_method' instead.")
  allowed_vmap_methods = ["sequential", "sequential_unrolled", "expand_dims",
                          "broadcast_all", "legacy_vectorized", None]
  if vmap_method not in allowed_vmap_methods:
    raise ValueError(
        f"vmap_method must be on of the allowed methods {allowed_vmap_methods}, "
        f"but got: {vmap_method}")

  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  tree_util.tree_map(_check_shape_dtype, result_shape_dtypes)
  result_avals = tree_util.tree_map(
      lambda x: core.ShapedArray(x.shape, x.dtype), result_shape_dtypes)
  flat_result_avals, out_tree = tree_util.tree_flatten(result_avals)
  out_flat = pure_callback_p.bind(
      *flat_args,
      callback=_FlatCallback(callback, in_tree),
      result_avals=tuple(flat_result_avals),
      sharding=sharding,
      vmap_method=vmap_method,
  )
  return tree_util.tree_unflatten(out_tree, out_flat)


# IO Callback

io_callback_p = core.Primitive("io_callback")
io_callback_p.multiple_results = True
dispatch.prim_requires_devices_during_lowering.add(io_callback_p)

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
effects.shardable_ordered_effects.add_type(OrderedIOEffect)


def io_callback_impl(
    *args,
    result_avals,
    callback: _FlatCallback,
    sharding: SingleDeviceSharding | None,
    ordered: bool,
):
  del result_avals, sharding, ordered
  try:
    cpu_device, *_ = xb.local_devices(backend="cpu")
  except RuntimeError as e:
    raise RuntimeError(
        "jax.io_callback failed to find a local CPU device to place the"
        " inputs on. Make sure \"cpu\" is listed in --jax_platforms or the"
        " JAX_PLATFORMS environment variable."
    ) from e
  args = api.device_put(args, cpu_device)
  with config.default_device(cpu_device):
    try:
      return tree_util.tree_map(np.asarray, callback(*args))
    except BaseException:
      logger.exception("jax.io_callback failed")
      raise


io_callback_p.def_impl(functools.partial(dispatch.apply_primitive,
                                         io_callback_p))


@io_callback_p.def_effectful_abstract_eval
def io_callback_abstract_eval(
    *avals,
    callback: _FlatCallback,
    result_avals,
    sharding: SingleDeviceSharding | None,
    ordered: bool,
):
  del avals, sharding, callback
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


def io_callback_batching_rule(
    args, dims, callback, result_avals, sharding, ordered
):
  from jax._src.lax.control_flow.loops import map as lax_map  # pytype: disable=import-error
  if ordered:
    raise ValueError("Cannot `vmap` ordered IO callback.")
  is_batched = [d is not batching.not_mapped for d in dims]
  new_args = [arg if dim is batching.not_mapped else
              batching.moveaxis(arg, dim, 0) for arg, dim in zip(args, dims)]
  unbatched_args, batched_args = util.partition_list(is_batched, new_args)
  def _batch_fun(batched_args):
    merged = util.merge_lists(is_batched, unbatched_args, batched_args)
    return io_callback_p.bind(*merged, callback=callback, sharding=sharding,
                              result_avals=result_avals, ordered=False)
  out_vals = lax_map(_batch_fun, batched_args)
  return out_vals, (0,) * len(out_vals)
batching.primitive_batchers[io_callback_p] = io_callback_batching_rule


def io_callback_lowering(ctx, *args, callback, sharding, ordered, **params):
  def _callback(*flat_args):
    return tuple(
        io_callback_impl(
            *flat_args,
            callback=callback,
            sharding=None,  # unused.
            ordered=ordered,
            **params,
        )
    )

  op_sharding = _callback_op_sharding(
      ctx.module_context.axis_context, sharding, ctx.avals_out)
  if ordered:
    token = ctx.tokens_in.get(_OrderedIOEffect)
    result, token, _ = emit_python_callback(
        ctx,
        _callback,
        token,
        list(args),
        ctx.avals_in,
        ctx.avals_out,
        has_side_effect=True,
        sharding=op_sharding,
    )
    ctx.set_tokens_out(mlir.TokenSet({_OrderedIOEffect: token}))
  else:
    result, _, _ = emit_python_callback(
        ctx,
        _callback,
        None,
        list(args),
        ctx.avals_in,
        ctx.avals_out,
        has_side_effect=True,
        sharding=op_sharding,
    )
  return result


mlir.register_lowering(io_callback_p, io_callback_lowering)


def io_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: Any,
    *args: Any,
    sharding: SingleDeviceSharding | None = None,
    ordered: bool = False,
    **kwargs: Any,
):
  """Calls an impure Python callback.

  For more explanation, see `External Callbacks`_.

  Args:
    callback: function to execute on the host. It is assumed to be an impure function.
      If ``callback`` is pure, using :func:`jax.pure_callback` instead may lead to
      more efficient execution.
    result_shape_dtypes: pytree whose leaves have ``shape`` and ``dtype`` attributes,
      whose structure matches the expected output of the callback function at runtime.
      :class:`jax.ShapeDtypeStruct` is often used to define leaf values.
    *args: arguments to be passed to the callback function
    sharding: optional sharding that specifies the device from which the callback should
      be invoked.
    ordered: boolean specifying whether sequential calls to callback must be ordered.
    **kwargs: keyword arguments to be passed to the callback function

  Returns:
    result: a pytree of :class:`jax.Array` objects whose structure matches that of
      ``result_shape_dtypes``.

  See Also:
    - :func:`jax.pure_callback`: callback designed for pure functions.
    - :func:`jax.debug.callback`: callback designed for general-purpose debugging.
    - :func:`jax.debug.print`: callback designed for printing.

  .. _External Callbacks: https://docs.jax.dev/en/latest/notebooks/external_callbacks.html
  """
  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  tree_util.tree_map(_check_shape_dtype, result_shape_dtypes)
  flat_shape_dtypes, out_tree = tree_util.tree_flatten(result_shape_dtypes)
  flat_result_avals = map(lambda x: core.ShapedArray(x.shape, x.dtype),
                          flat_shape_dtypes)
  out_flat = io_callback_p.bind(
      *flat_args,
      callback=_FlatCallback(callback, in_tree),
      result_avals=tuple(flat_result_avals),
      sharding=sharding,
      ordered=ordered,
  )
  return tree_util.tree_unflatten(out_tree, out_flat)


def is_empty_shape(s: core.Shape) -> bool:
  return any(d == 0 for d in s)


def send_to_host(
    channel: int,
    token: hlo.TokenType,
    operand: Any,
    name: str,
    *,
    sharding: SdyArrayList | xc.OpSharding | None = None,
) -> ir.Value:
  channel_handle = hlo.ChannelHandle.get(channel, mlir.SEND_TO_HOST_TYPE)
  send_op = hlo.SendOp([operand], token, channel_handle,
                        is_host_transfer=ir.BoolAttr.get(True))
  send_op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
      dict(
          _xla_host_transfer_handler_name=ir.StringAttr.get(str(name)),
          _xla_host_transfer_rendezvous=ir.StringAttr.get(str(name))))
  if sharding is not None:
    if config.use_shardy_partitioner.value:
      # `SendOp`'s return type is a StableHLO `TokenType`. However JAX passed
      # in the maximal sharding of the array type. Since a token has no rank,
      # we need to create an equivalent sharding with no dimensions. If there
      # are multiple shardings, just grab the first one since all these
      # shardings should be the same.
      assert isinstance(sharding, SdyArrayList)
      assert len(sharding.shardings) >= 1
      sharding = SdyArrayList([
          SdyArray(
              mesh_shape=(), dim_shardings=[],
              logical_device_ids=sharding.shardings[0].logical_device_ids)])
    mlir.set_sharding(send_op, sharding)
  return send_op.result


def receive_from_host(
    channel: int,
    token: hlo.TokenType,
    out_aval: core.ShapedArray,
    name: str,
    *,
    sharding: SdyArrayList | xc.OpSharding | None = None,
) -> tuple[ir.Value, ir.Value]:
  channel_handle = hlo.ChannelHandle.get(channel, mlir.RECV_FROM_HOST_TYPE)
  recv_op = hlo.RecvOp([mlir.aval_to_ir_type(out_aval),
                        hlo.TokenType.get()], token, channel_handle,
                        is_host_transfer=ir.BoolAttr.get(True))
  recv_op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
      dict(
          _xla_host_transfer_handler_name=ir.StringAttr.get(str(name)),
          _xla_host_transfer_rendezvous=ir.StringAttr.get(str(name))))
  if sharding is not None:
    if config.use_shardy_partitioner.value:
      assert isinstance(sharding, SdyArrayList)
      assert len(sharding.shardings) >= 1
       # `RecvOp`'s last argument is a `TokenType`. Since Shardy requires the
      # number of shardings to match the number of results, but JAX only sees
      # the array result, we need to add an equivalent sharding for the token.
      # Note that even if a function returns N results, we will end up with N
      # `RecvOp`s, so we only need to get the first sharding. All shardings are
      # the same anyways, operating on the same single device ID.
      sharding = SdyArrayList([
          sharding.shardings[0],
          SdyArray(
              mesh_shape=(), dim_shardings=[],
              logical_device_ids=sharding.shardings[0].logical_device_ids)])
    mlir.set_sharding(recv_op, sharding)
  # Token should be at the end of the results
  result, token = recv_op.results
  return token, result



def _aval_to_xla_shape(aval: core.AbstractValue) -> xc.Shape:
  try:
    return _xla_shape_handlers[type(aval)](aval)
  except KeyError as err:
    raise TypeError(f"No xla_shape_handler for type: {type(aval)}") from err

_xla_shape_handlers: dict[type[core.AbstractValue],
                         Callable[[Any], xc.Shape]] = {}

def _make_array_shape(aval: core.ShapedArray) -> xc.Shape:
  aval = core.physical_aval(aval)
  dtype = np.dtype('bool') if aval.dtype == dtypes.float0 else aval.dtype
  return xc.Shape.array_shape(dtype, aval.shape)
_xla_shape_handlers[core.ShapedArray] = _make_array_shape

_xla_shape_handlers[core.AbstractToken] = lambda _: xc.Shape.token_shape()


def _emit_tpu_python_callback(
    backend: xb.XlaBackend,
    ctx: mlir.LoweringRuleContext,
    callback,
    token: Any | None,
    operands: Sequence[ir.Value],
    operand_avals: Sequence[core.ShapedArray],
    operand_shapes: Sequence[xc.Shape],
    result_avals: Sequence[core.ShapedArray],
    result_shapes: Sequence[xc.Shape],
    *,
    sharding: SdyArrayList | xc.OpSharding | None = None,
) -> tuple[Sequence[ir.Value], Any]:
  token = token or hlo.create_token()
  _wrapped_callback = callback

  send_channels = []
  if not operand_avals:
    # If there are no operands to the callback, we need to insert a dummy send
    # op or the callback will never be triggered!
    # TODO(sharadmv,chky): Enable this fix in the runtime as opposed to in
    # MLIR builder.
    callback_without_args = _wrapped_callback
    def _wrapped_callback(*args):  # pylint: disable=function-redefined
      del args
      return callback_without_args()
    send_channel = ctx.module_context.new_channel()
    dummy_send_aval = core.ShapedArray((1,), np.float32)
    dummy_send_val = mlir.ir_constant(np.zeros(1, np.float32))
    operand_shapes = [*operand_shapes, _aval_to_xla_shape(dummy_send_aval)]
    token = send_to_host(send_channel, token, dummy_send_val, callback.__name__,
                         sharding=sharding)
    send_channels.append(send_channel)
  else:
    for operand in operands:
      channel = ctx.module_context.new_channel()
      token = send_to_host(channel, token, operand, callback.__name__,
                           sharding=sharding)
      send_channels.append(channel)

  recv_channels = []
  outputs = []
  for result_aval in result_avals:
    channel = ctx.module_context.new_channel()
    assert isinstance(result_aval, core.ShapedArray)
    token, out = receive_from_host(channel, token, result_aval,
                                   callback.__name__, sharding=sharding)
    outputs.append(out)
    recv_channels.append(channel)
  ifrt_callback = backend.make_python_callback_from_host_send_and_recv(
      _wrapped_callback, operand_shapes, result_shapes, send_channels,
      recv_channels, pickle_util.dumps)
  ctx.module_context.add_host_callback(ifrt_callback)
  return outputs, token


def emit_python_callback(
    ctx: mlir.LoweringRuleContext,
    callback,
    token: Any | None,
    operands: Sequence[ir.Value],
    operand_avals: Sequence[core.ShapedArray],
    result_avals: Sequence[core.ShapedArray],
    *,
    has_side_effect: bool,
    partitioned: bool = False,
    sharding: SdyArrayList | xc.OpSharding | None = None,
) -> tuple[Sequence[mlir.IrValues], Any, Any]:
  """Emits MLIR that calls back to a provided Python function.

  Args:
    ctx: The lowering context.
    callback: The Python callback function.
    token: The token to use for the callback.
    operands: The operands to the callback.
    operand_avals: The abstract values of the operands.
    result_avals: The abstract values of the results.
    has_side_effect: Whether the callback has side effects.
    partitioned: If True, then `callback` is called on local shards only. If
      False, then `callback` is called on all shards.
    sharding: The sharding of the callback.

  Returns:
    A tuple of MLIR result values, a new token (if any), and the host callback
    object.
  """
  if len(ctx.module_context.platforms) > 1:
    raise NotImplementedError("multi-platform lowering for python_callback")
  platform = ctx.module_context.platforms[0]
  if platform not in {"cpu", "cuda", "rocm", "tpu"}:
    raise ValueError(
        f"`EmitPythonCallback` not supported on {platform} backend.")
  if partitioned:
    if platform not in {"cpu", "cuda", "rocm"}:
      raise ValueError(
          f"Partitioned callback not supported on {platform} backend.")
    if result_avals:
      raise ValueError("Partitioned callback not supported with return values.")
  backend = ctx.module_context.get_backend()
  result_shapes = [_aval_to_xla_shape(aval) for aval in result_avals]
  operand_shapes = [_aval_to_xla_shape(aval) for aval in operand_avals]

  # First we apply checks to ensure output shapes and dtypes match the expected
  # ones.
  def _wrapped_callback(*args):
    out_vals = callback(*args)
    if len(out_vals) != len(result_avals):
      raise RuntimeError(
          "Mismatched number of outputs from callback. "
          "Expected: {}, Actual: {}".format(len(result_avals), len(out_vals)))
    # Handle Python literals, and custom arrays, e.g., tf.Tensor.
    out_vals = tuple(xla.canonicalize_dtype(np.asarray(a)) for a in out_vals)
    for i, (out_val, out_aval) in enumerate(zip(out_vals, result_avals)):
      if out_val.shape != out_aval.shape:
        raise RuntimeError(
            f"Incorrect output shape for return value #{i}: "
            f"Expected: {out_aval.shape}, Actual: {out_val.shape}")
      if out_val.dtype != out_aval.dtype:
        raise RuntimeError(
            f"Incorrect output dtype for return value #{i}: "
            f"Expected: {out_aval.dtype}, Actual: {out_val.dtype}")

    if platform == "tpu":
      # On TPU we cannot receive empty arrays. So, we return from the wrapped
      # callback only the non-empty results, and we will create empty constants
      # in the receiving computation.
      # TODO(b/238239458): fix TPU Recv to work with empty arrays.
      non_empty_out_vals = tuple(
          out_val
          for out_val, result_aval in zip(out_vals, result_avals)
          if not is_empty_shape(result_aval.shape))
      return non_empty_out_vals
    else:
      return out_vals

  if platform == "tpu":
    non_empty_result_avals, non_empty_result_shapes = util.unzip2([
        (aval, shape)
        for aval, shape in zip(result_avals, result_shapes)
        if not is_empty_shape(aval.shape)])
    non_empty_outputs, token = _emit_tpu_python_callback(
        backend, ctx, _wrapped_callback,  token,
        operands, operand_avals, operand_shapes,
        non_empty_result_avals, non_empty_result_shapes,
        sharding=sharding)
    non_empty_outputs_iter = iter(non_empty_outputs)
    outputs = [
        mlir.ir_constant(np.zeros(result_aval.shape, dtype=result_aval.dtype))
        if is_empty_shape(result_aval.shape) else next(non_empty_outputs_iter)
        for result_aval in result_avals]
    return outputs, token, None

  device = "gpu" if platform in {"cuda", "rocm"} else "cpu"
  partition = "_partitioned" if partitioned else ""
  call_target_name = f"xla_ffi{partition}_python_{device}_callback"
  if token:
    callback_without_token = _wrapped_callback
    def _wrapped_callback(token, *args):  # type: ignore  # pylint: disable=function-redefined
      return (token, *callback_without_token(*args))
    operands = [token, *operands]
    if (
        config.use_shardy_partitioner.value
        and sharding is not None
        and len(ctx.avals_out) > 0
        and isinstance(sharding, SdyArrayList)
    ):
      # Add a sharding annotation for the token if we have at least one
      # output. Otherwise, the single shardy annotation required of all ops
      # (even those without any results) can annotate the token.
      sharding = SdyArrayList([
          SdyArray(
              mesh_shape=(),
              dim_shardings=[],
              logical_device_ids=()),
          *sharding.shardings])
    ctx = dataclasses.replace(
        ctx,
        avals_in=[core.abstract_token, *ctx.avals_in],
        avals_out=[core.abstract_token, *ctx.avals_out],
    )

  # TODO(dsuo): Remove this line once we deprecate the XLA custom call
  # handler.
  ifrt_callback = _wrapped_callback
  ctx.module_context.add_host_callback(ifrt_callback)
  index = np.uint64(len(ctx.module_context.host_callbacks) - 1)
  result = ffi.build_ffi_lowering_function(  # type: ignore
      call_target_name,
      has_side_effect=has_side_effect,
  )(ctx, *operands, index=np.uint64(index))

  if sharding is not None:
    mlir.set_sharding(result, sharding)

  results = result.results  # type: ignore

  if token:
    token, *results = results  # type: ignore

  return results, token, ifrt_callback  # type: ignore
