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

from collections.abc import Callable, Sequence
import functools
from typing import Any

import numpy as np

from jax._src import core
from jax._src import dispatch
from jax._src import effects
from jax._src import ffi
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lib import ffi as ffi_lib

export = util.set_module("jax.experimental.buffer_callback")
Buffer = export(ffi_lib.Buffer)
ExecutionStage = export(ffi_lib.ExecutionStage)
ExecutionContext = export(ffi_lib.ExecutionContext)


def buffer_callback(
    callback: Callable[..., None],
    result_shape_dtypes: object,
    *,
    has_side_effect: bool = False,
    vmap_method: str | None = None,
    input_output_aliases: dict[int, int] | None = None,
    command_buffer_compatible: bool = False,
):
  """An experimental callback that operates in place on device buffers.

  Only supported on CPU and GPU backends.

  Note that the plan is for this to eventually be replaced by a consolidated
  callback API built using JAX mutable arrays, but for now this provides a
  mechanism for prototyping computational kernels using other Python libraries
  including Numpy, PyTorch, Cupy, and others.

  Let's start with a simple example:

    >>> def py_add_one_inplace(ctx, out, x):
    ...   np.asarray(out)[...] = np.asarray(x) + 1
    ...
    >>> x = jnp.array(41, dtype=jnp.int32)
    >>> out_type = jax.ShapeDtypeStruct(x.shape, x.dtype)
    >>> add_one = buffer_callback(py_add_one_inplace, out_type)
    >>> add_one(x)  # doctest: +SKIP
    Array(42, dtype=int32)

  In this example, we're executing a numpy computation via JAX, and this could
  have been implemented using :func:`jax.pure_callback`, but in this case, the
  output is being populated in-place. This means that JAX doesn't need to copy
  the output arrays upon returning from the callback. Note that even though the
  callback function operates on mutable buffers, JAX still sees this as an
  operation that consumes and produces regular immutable JAX arrays.

  Unlike the other JAX callback APIs, ``buffer_callback`` requires that the
  user-defined Python function have the following signature:

  .. code-block:: python

    def callback(ctx: ExecutionContext, out, *args) -> None:
      ...

  where ``ctx`` is an instance of
  :class:`~jax.experimental.buffer_callback.ExecutionContext`, which mainly
  provides access to XLA's computation stream when running on GPU, ``out`` is a
  pytree of mutable :class:`~jax.experimental.buffer_callback.Buffer` objects,
  and the ``args`` arguments have the same pytree structure as the inputs, but
  each leaf is :class:`~jax.experimental.buffer_callback.Buffer`. This callback
  should not return any values, and it should overwrite the ``out`` buffers in
  place to output values back to JAX.

  It's important to note that this Python function can't really be called
  except via ```buffer_callback`` itself, because it's not (yet!) possible to
  construct mutable JAX buffers directly in Python.

  The bespoke :class:`~jax.experimental.buffer_callback.Buffer` type is an
  array-like object that supports the ``__array__`` protocol on CPU, the
  ``__cuda_array_interface__`` protocol on GPU, and the ``__dlpack__`` protocol
  on both CPU and GPU.

  Args:
    callback: A Python function with the signature and behavior described above.
    result_shape_dtypes: A pytree whose leaves have ``shape`` and ``dtype``
      attributes, with a structure that matches the expected output of the
      callback function at runtime. :class:`jax.ShapeDtypeStruct` is often used
      to define leaf values.
    has_side_effect: Whether the callback has side effects.
    vmap_method: A string specifying how the callback transforms under
      :func:`~jax.vmap` as described in the docs for :func:`~jax.pure_callback`.
    input_output_aliases: a dictionary mapping the index of some inputs to
      the index of the output that aliases them. These indices are in the
      flattened inputs and outputs.
    command_buffer_compatible: if ``True``, the callback will be traced into
      the command buffer. This means that the Python code should only be
      executed once, and then the operations will be replayed for every
      subsequent call.

  Returns:
    A new callable that accepts :class:`jax.Array` inputs (and pytrees thereof),
    and  pytree of :class:`jax.Array` objects whose structure matches that
    of ``result_shape_dtypes``.

  See Also:
    - :func:`jax.pure_callback`: callback designed for pure host functions.
    - :func:`jax.experimental.io_callback`: callback designed for impure host
      functions.
    - :func:`jax.debug.callback`: callback designed for general-purpose
      debugging.
    - :func:`jax.debug.print`: callback designed for printing.
  """
  flat_shape_dtypes, out_tree = tree_util.tree_flatten(result_shape_dtypes)
  flat_result_avals = tuple(
      core.ShapedArray(x.shape, x.dtype) for x in flat_shape_dtypes
  )

  def wrapped_callback(*args, **kwargs):
    flat_args, in_tree = tree_util.tree_flatten((args, kwargs))

    in_avals = [core.get_aval(x) for x in flat_args]
    static_input_output_aliases: tuple[tuple[int, int], ...] = ()
    if input_output_aliases is not None:
      for i_idx, o_idx in sorted(input_output_aliases.items()):
        i_idx, o_idx = int(i_idx), int(o_idx)
        if i_idx >= len(args):
          raise ValueError(
              f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' "
              f"with input index {i_idx} outside the range [0, "
              f"{len(args)}).")
        if o_idx >= len(flat_result_avals):
          raise ValueError(
              f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' "
              f"with output index {o_idx} outside the range [0, "
              f"{len(flat_result_avals)}).")
        in_aval = in_avals[i_idx]
        out_aval = flat_result_avals[o_idx]
        if not ffi._check_compatible_avals(in_aval, out_aval):
          raise ValueError(
              f"input_output_aliases contains the mapping '{i_idx}:{o_idx}' "
              f"referring to an input with abstract value {in_aval} and an "
              f"output with a different abstract value {out_aval}.")
        static_input_output_aliases += ((i_idx, o_idx),)

    out_flat = buffer_callback_p.bind(
        *flat_args,
        callback=callback,
        result_avals=flat_result_avals,
        in_tree=in_tree,
        out_tree=out_tree,
        vmap_method=vmap_method,
        has_side_effect=has_side_effect,
        input_output_aliases=static_input_output_aliases,
        command_buffer_compatible=command_buffer_compatible,
    )
    return tree_util.tree_unflatten(out_tree, out_flat)

  return wrapped_callback


buffer_callback_p = core.Primitive("buffer_callback")
buffer_callback_p.multiple_results = True
dispatch.prim_requires_devices_during_lowering.add(buffer_callback_p)
dispatch.simple_impl(buffer_callback_p)


class BufferCallbackEffect(effects.Effect):
  def __str__(self):
    return "BufferCallback"

_BufferCallbackEffect = BufferCallbackEffect()
effects.lowerable_effects.add_type(BufferCallbackEffect)
effects.control_flow_allowed_effects.add_type(BufferCallbackEffect)


@buffer_callback_p.def_effectful_abstract_eval
def _buffer_callback_abstract_eval(
    *args,
    result_avals: tuple[core.ShapedArray, ...],
    has_side_effect: bool,
    **_,
):
  del args
  effects = {_BufferCallbackEffect} if has_side_effect else core.no_effects
  return result_avals, effects


def _buffer_callback_jvp_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError(
      "Buffer callbacks do not support JVP. "
      "Please use `jax.custom_jvp` to use callbacks while taking gradients.")
ad.primitive_jvps[buffer_callback_p] = _buffer_callback_jvp_rule


def _buffer_callback_transpose_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError(
      "Buffer callbacks do not support transpose. "
      "Please use `jax.custom_vjp` to use callbacks while taking gradients.")
ad.primitive_transposes[buffer_callback_p] = _buffer_callback_transpose_rule

batching.primitive_batchers[buffer_callback_p] = functools.partial(
    ffi.ffi_batching_rule, buffer_callback_p
)


def _buffer_callback_lowering(
    ctx: mlir.LoweringRuleContext,
    *args: Any,
    callback,
    in_tree: Any,
    out_tree: Any,
    has_side_effect: bool,
    input_output_aliases: Sequence[tuple[int, int]],
    command_buffer_compatible: bool,
    **_,
):

  if len(ctx.module_context.platforms) > 1:
    raise NotImplementedError("multi-platform lowering for buffer_callback")
  platform = ctx.module_context.platforms[0]
  target_name = {
      "cpu": "xla_buffer_python_cpu_callback",
      "cuda": "xla_buffer_python_gpu_callback",
      "rocm": "xla_buffer_python_gpu_callback",
  }.get(platform)
  if target_name is None:
    raise ValueError(f"`buffer_callback` not supported on {platform} backend.")

  if command_buffer_compatible and platform in ("cuda", "rocm"):
    target_name += "_cmd_buffer"

  def wrapped_callback(exec_ctx, *args: Any):
    args_in, args_out = util.split_list(args, [in_tree.num_leaves])
    py_args_in, py_kwargs_in = tree_util.tree_unflatten(in_tree, args_in)
    py_args_out = tree_util.tree_unflatten(out_tree, args_out)
    if callback(exec_ctx, py_args_out, *py_args_in, **py_kwargs_in) is not None:
      raise ValueError("buffer_callback callback must not return any values.")
    return ()

  ctx.module_context.add_host_callback(wrapped_callback)
  index = np.uint64(len(ctx.module_context.host_callbacks) - 1)
  rule = ffi.ffi_lowering(
      target_name,
      has_side_effect=has_side_effect,
      operand_output_aliases=dict(input_output_aliases),
  )
  return rule(ctx, *args, index=index)
mlir.register_lowering(buffer_callback_p, _buffer_callback_lowering)
