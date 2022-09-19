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
"""Module for JAX debugging primitives and related functionality."""
import enum
import functools
import string
import sys

from typing import Any, Dict, Callable, Sequence, Set, Tuple

from jax import core
from jax import tree_util
from jax import lax
from jax._src import ad_checkpoint
from jax._src import custom_derivatives
from jax._src import lib as jaxlib
from jax._src import util
from jax.config import config
from jax.experimental.sharding import Sharding
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax._src.lax import control_flow as lcf
from jax._src.lib import xla_client as xc
import jax.numpy as jnp

# pytype: disable=import-error
try:
  import rich
  import rich.align
  import rich.box
  import rich.padding
  import rich.table
  RICH_ENABLED = True
except:
  RICH_ENABLED = False
# pytype: enable=import-error

DebugEffect = enum.Enum('DebugEffect', ['PRINT', 'ORDERED_PRINT'])

core.ordered_effects.add(DebugEffect.ORDERED_PRINT)
mlir.lowerable_effects.add(DebugEffect.PRINT)
mlir.lowerable_effects.add(DebugEffect.ORDERED_PRINT)
lcf.allowed_effects.add(DebugEffect.PRINT)
lcf.allowed_effects.add(DebugEffect.ORDERED_PRINT)
ad_checkpoint.remat_allowed_effects.add(DebugEffect.PRINT)
ad_checkpoint.remat_allowed_effects.add(DebugEffect.ORDERED_PRINT)
custom_derivatives.allowed_effects.add(DebugEffect.PRINT)
custom_derivatives.allowed_effects.add(DebugEffect.ORDERED_PRINT)

# `debug_callback_p` is the main primitive for staging out Python callbacks.
debug_callback_p = core.Primitive('debug_callback')
debug_callback_p.multiple_results = True

map, unsafe_map = util.safe_map, map

@debug_callback_p.def_impl
def debug_callback_impl(*args, callback: Callable[..., Any],
                        effect: DebugEffect):
  del effect
  return callback(*args)

@debug_callback_p.def_effectful_abstract_eval
def debug_callback_abstract_eval(*flat_avals, callback: Callable[..., Any],
                                 effect: DebugEffect):
  del flat_avals, callback
  return [], {effect}

def debug_callback_batching_rule(args, dims, **params):
  """Unrolls the debug callback across the mapped axis."""
  axis_size = next(x.shape[i] for x, i in zip(args, dims)
                   if i is not None)
  # TODO(sharadmv): implement in terms of rolled loop unstead of
  # unrolled.
  def get_arg_at_dim(i, dim, arg):
    if dim is batching.not_mapped:
      # Broadcast unmapped argument
      return arg
    return lax.index_in_dim(arg, i, axis=dim, keepdims=False)
  outs = []
  for i in range(axis_size):
    args_idx = map(functools.partial(get_arg_at_dim, i), dims, args)
    outs.append(debug_callback_p.bind(*args_idx, **params))
  outs = [jnp.stack(xs) for xs in zip(*outs)]
  return outs, (0,) * len(outs)
batching.primitive_batchers[debug_callback_p] = debug_callback_batching_rule

def debug_callback_jvp_rule(primals, tangents, **params):
  return debug_callback_p.bind(*primals, **params), []
ad.primitive_jvps[debug_callback_p] = debug_callback_jvp_rule

def debug_callback_transpose_rule(*flat_args, callback: Callable[..., Any],
    effect: DebugEffect):
  del flat_args, callback, effect
  raise ValueError("Transpose doesn't support debugging callbacks.")
ad.primitive_transposes[debug_callback_p] = debug_callback_transpose_rule

def debug_callback_lowering(ctx, *args, effect, callback, **params):

  if isinstance(ctx.module_context.axis_context,
                (mlir.SPMDAxisContext, mlir.ShardingContext)):
    # Apply maximal sharding so pjit only executes the callback on device 0.
    sharding = xc.OpSharding()
    sharding.type = xc.OpSharding.Type.MAXIMAL
    sharding.tile_assignment_dimensions = [1]
    sharding.tile_assignment_devices = [0]
  else:
    sharding = None

  def _callback(*flat_args):
    return tuple(
        debug_callback_p.impl(
            *flat_args, effect=effect, callback=callback, **params))
  if effect in core.ordered_effects:
    token = ctx.tokens_in.get(effect)[0]
    result, token, keepalive = mlir.emit_python_callback(
        ctx, _callback, token, list(args), ctx.avals_in, ctx.avals_out, True)
    ctx.set_tokens_out(mlir.TokenSet({effect: (token,)}))
  else:
    result, token, keepalive = mlir.emit_python_callback(
        ctx, _callback, None, list(args), ctx.avals_in, ctx.avals_out, True,
        sharding=sharding)
  ctx.module_context.add_keepalive(keepalive)
  return result
mlir.register_lowering(debug_callback_p, debug_callback_lowering,
                       platform="cpu")
mlir.register_lowering(
    debug_callback_p, debug_callback_lowering, platform="gpu")
if jaxlib.version >= (0, 3, 15):
  mlir.register_lowering(
      debug_callback_p, debug_callback_lowering, platform="tpu")

def _debug_callback_partial_eval_custom(saveable, unks_in, inst_in, eqn):
  # The default behavior for effectful primitives is to not stage them if
  # possible. For debug callback, we actually want it to be staged to
  # provide more information to the user. This rule bypasses partial_eval's
  # regular behavior to do that. Specifically, we will stage the callback
  # if:
  # 1) the policy says debug_callbacks are not saveable
  # 2) the policy says debug_callbacks are saveable BUT all of the input
  #    values are instantiated.
  # The purpose is to call back with as much information as possible while
  # avoiding unnecessarily staging out other values.
  if any(unks_in):
    # The usual case (if we have any unknowns, we need to stage it out)
    res = [v for v, inst in zip(eqn.invars, inst_in) if not inst]
    return None, eqn, [], [], res
  if saveable(debug_callback_p, *[v.aval for v in eqn.invars], **eqn.params):
    # The policy is telling us we can save the debug callback.
    if all(inst_in):
      # If all of the inputs are instantiated, we also stage out the
      # debug_callback.
      return eqn, eqn, [], [], []
    else:
      # If any are not instantiated, we don't do any extra staging to avoid
      # affecting the computation.
      return eqn, None, [], [], []
  # If we can't save the debug callback (thanks to the policy) we listen to
  # the policy and stage out the debug callback.
  return eqn, eqn, [], [], []
pe.partial_eval_jaxpr_custom_rules[debug_callback_p] = (
    _debug_callback_partial_eval_custom)

def debug_callback(callback: Callable[..., Any], *args: Any,
                   ordered: bool = False, **kwargs: Any):
  """Calls a stageable Python callback.

  `debug_callback` enables you to pass in a Python function that can be called
  inside of a staged JAX program. A `debug_callback` follows existing JAX
  transformation *pure* operational semantics, which are therefore unaware of
  side-effects. This means the effect could be dropped, duplicated, or
  potentially reordered in the presence of higher-order primitives and
  transformations.

  We want this behavior because we'd like `debug_callback` to be "innocuous",
  i.e. we want these primitives to change the JAX computation as little as
  possible while revealing as much about them as possible, such as which parts
  of the computation are duplicated or dropped.

  Args:
    callback: A Python callable. Its return value will be ignored.
    *args: The positional arguments to the callback.
    ordered: A keyword only argument used to indicate whether or not the
      staged out computation will enforce ordering of this callback w.r.t.
      other ordered callbacks.
    **kwargs: The keyword arguments to the callback.
  Returns:
    The value of `callback(*args, **kwargs)`.
  """
  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  effect = DebugEffect.ORDERED_PRINT if ordered else DebugEffect.PRINT
  def _flat_callback(*flat_args):
    args, kwargs = tree_util.tree_unflatten(in_tree, flat_args)
    callback(*args, **kwargs)
    return []
  return debug_callback_p.bind(*flat_args, callback=_flat_callback,
                               effect=effect)

class _DebugPrintFormatChecker(string.Formatter):

  def check_unused_args(self, used_args, args, kwargs):
    unused_args = [arg for i, arg in enumerate(args) if i not in used_args]
    unused_kwargs = [k for k in kwargs if k not in used_args]
    if unused_args:
      raise ValueError(
          f"Unused positional arguments to `jax.debug.print`: {unused_args}")
    if unused_kwargs:
      raise ValueError(
          f"Unused keyword arguments to `jax.debug.print`: {unused_kwargs}. "
          "You may be passing an f-string (i.e, `f\"{x}\"`) into "
          "`jax.debug.print` and instead should pass in a regular string.")

formatter = _DebugPrintFormatChecker()

def _format_print_callback(fmt: str, *args, **kwargs):
  sys.stdout.write(fmt.format(*args, **kwargs) + "\n")

def debug_print(fmt: str, *args, ordered: bool = False, **kwargs) -> None:
  """Prints values and works in staged out JAX functions.

  Args:
    fmt: A format string, e.g. ``"hello {x}"``, that will be used to format
      input arguments.
    *args: A list of positional arguments to be formatted.
    ordered: A keyword only argument used to indicate whether or not the
      staged out computation will enforce ordering of this ``debug_print``
      w.r.t. other ordered ``debug_print`` calls.
    **kwargs: Additional keyword arguments to be formatted.
  """
  # Check that we provide the correct arguments to be formatted
  formatter.format(fmt, *args, **kwargs)

  debug_callback(functools.partial(_format_print_callback, fmt), *args,
                 **kwargs, ordered=ordered)

def _slice_to_chunk_idx(size: int, slc: slice) -> int:
  if slc.stop == slc.start == None:
    return 0
  slice_size = slc.stop - slc.start
  assert slc.start % slice_size == 0
  assert size % slice_size == 0
  return slc.start // slice_size

def visualize_sharding(shape: Sequence[int], sharding: Sharding, *,
                       use_color: bool = False, scale: float = 1.):
  """Visualizes a `Sharding`."""
  if not RICH_ENABLED:
    raise ValueError("`visualize_sharding` requires `rich` to be installed.")
  if use_color:
    # TODO(sharadmv): Implement color in the visualizer
    raise NotImplementedError
  if len(shape) > 2 or len(shape) < 1:
    raise ValueError(
        "`visualize_sharding` only works for shapes with 1 and 2 dimensions.")

  base_height = int(10 * scale)
  aspect_ratio = (shape[1] if len(shape) == 2 else 1) / shape[0]
  base_width = int(base_height * aspect_ratio)
  height_to_width_ratio = 2.5

  # Grab the device kind from the first device
  device_kind = next(iter(sharding.device_set)).device_kind.upper()

  device_indices_map = sharding.devices_indices_map(tuple(shape))
  slices: Dict[Tuple[int, ...], Set[int]] = {}
  heights: Dict[Tuple[int, ...], int] = {}
  widths: Dict[Tuple[int, ...], int] = {}
  for dev, slcs in device_indices_map.items():
    chunk_idxs = tuple(map(_slice_to_chunk_idx, shape, slcs))
    if slcs is None:
      raise NotImplementedError
    if len(slcs) == 2:
      vert, horiz = slcs
      vert_size  = ((vert.stop  - vert.start ) if vert.stop  is not None
                    else shape[0])
      horiz_size = ((horiz.stop - horiz.start) if horiz.stop is not None
                    else shape[1])
      chunk_height = vert_size / shape[0] * base_height
      chunk_width = (
          horiz_size / shape[1] * base_width *
          height_to_width_ratio)
      heights[chunk_idxs] = int(chunk_height)
      widths[chunk_idxs] = int(chunk_width)
      slices.setdefault(chunk_idxs, set()).add(dev.id)
    else:
      # In the 1D case, we set the height to 1.
      horiz, = slcs
      vert = slice(0, 1, None)
      horiz_size = (
          (horiz.stop - horiz.start) if horiz.stop is not None else shape[0])
      heights[(0, *chunk_idxs)] = 1
      widths[(0, *chunk_idxs)]  = int(horiz_size / shape[0] * base_width)
      slices.setdefault((0, *chunk_idxs), set()).add(dev.id)
  num_rows = max([a[0] for a in slices.keys()]) + 1
  if len(list(slices.keys())[0]) == 1:
    num_cols = 1
  else:
    num_cols = max([a[1] for a in slices.keys()]) + 1

  table = rich.table.Table(show_header=False, show_lines=True, padding=0,
                           highlight=True, pad_edge=False,
                           box=rich.box.SQUARE)
  for i in range(num_rows):
    col = []
    for j in range(num_cols):
      entry = f"{device_kind} "+",".join([str(s) for s in sorted(slices[i, j])])
      width, height = widths[i, j], heights[i, j]
      left_padding, remainder = divmod(width - len(entry), 2)
      right_padding = left_padding + remainder
      top_padding, remainder = divmod(height - 1, 2)
      bottom_padding = top_padding + remainder
      padding = (top_padding, right_padding, bottom_padding, left_padding)
      padding = tuple(max(x, 0) for x in padding)  # type: ignore
      col.append(
          rich.padding.Padding(
            rich.align.Align(entry, "center", vertical="middle"), padding))
    table.add_row(*col)
  rich.get_console().print(table, end='\n\n')

def visualize_array_sharding(arr, **kwargs):
  """Visualizes an array's sharding."""
  if not config.jax_array:
    raise NotImplementedError("`visualize_array_sharding` not implemented.")
  return visualize_sharding(arr.shape, arr.sharding, **kwargs)
