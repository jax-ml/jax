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
import sys

from typing import Callable, Any

from jax import core
from jax import tree_util
from jax import lax
from jax._src import lib as jaxlib
from jax._src import util
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax._src.lax import control_flow as lcf
import jax.numpy as jnp

DebugEffect = enum.Enum('DebugEffect', ['PRINT', 'ORDERED_PRINT'])

core.ordered_effects.add(DebugEffect.ORDERED_PRINT)
mlir.lowerable_effects.add(DebugEffect.PRINT)
mlir.lowerable_effects.add(DebugEffect.ORDERED_PRINT)
lcf.allowed_effects.add(DebugEffect.PRINT)
lcf.allowed_effects.add(DebugEffect.ORDERED_PRINT)

# `debug_callback_p` is the main primitive for staging out Python callbacks.
debug_callback_p = core.Primitive('debug_callback')
debug_callback_p.multiple_results = True

map, unsafe_map = util.safe_map, map

@debug_callback_p.def_impl
def debug_callback_impl(*flat_args, callback: Callable[..., Any],
    effect: DebugEffect, in_tree: tree_util.PyTreeDef):
  del effect
  args, kwargs = tree_util.tree_unflatten(in_tree, flat_args)
  out = callback(*args, **kwargs)
  return tree_util.tree_leaves(out)

@debug_callback_p.def_effectful_abstract_eval
def debug_callback_abstract_eval(*flat_avals, callback: Callable[..., Any],
    effect: DebugEffect, in_tree: tree_util.PyTreeDef):
  del flat_avals, callback, in_tree
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

def debug_callback_jvp_rule(*flat_args, callback: Callable[..., Any],
    effect: DebugEffect, in_tree: tree_util.PyTreeDef):
  del flat_args, callback, effect, in_tree
  # TODO(sharadmv): link to relevant documentation when it exists
  raise ValueError(
      "JVP doesn't support debugging callbacks. "
      "Instead, you can use them with `jax.custom_jvp` or `jax.custom_vjp`.")
ad.primitive_jvps[debug_callback_p] = debug_callback_jvp_rule

def debug_callback_transpose_rule(*flat_args, callback: Callable[..., Any],
    effect: DebugEffect, in_tree: tree_util.PyTreeDef):
  del flat_args, callback, effect, in_tree
  raise ValueError("Transpose doesn't support debugging callbacks.")
ad.primitive_transposes[debug_callback_p] = debug_callback_transpose_rule

def debug_callback_lowering(ctx, *args, effect, callback, **params):

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
        ctx, _callback, None, list(args), ctx.avals_in, ctx.avals_out, True)
  ctx.module_context.add_keepalive(keepalive)
  return result
mlir.register_lowering(debug_callback_p, debug_callback_lowering,
                       platform="cpu")
mlir.register_lowering(
    debug_callback_p, debug_callback_lowering, platform="gpu")
if jaxlib.version >= (0, 3, 15):
  mlir.register_lowering(
      debug_callback_p, debug_callback_lowering, platform="tpu")

def debug_callback(callback: Callable[..., Any], effect: DebugEffect, *args,
                   **kwargs):
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
    callback: A Python callable.
    effect: A `DebugEffect`.
    *args: The positional arguments to the callback.
    **kwargs: The positional arguments to the callback.
  Returns:
    The value of `callback(*args, **kwargs)`.
  """
  if not isinstance(effect, DebugEffect):
    raise ValueError("Can only use `DebugEffect` effects in `debug_callback`")
  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  return debug_callback_p.bind(*flat_args, callback=callback, effect=effect,
                               in_tree=in_tree)

def _format_print_callback(fmt: str, *args, **kwargs):
  sys.stdout.write(fmt.format(*args, **kwargs) + "\n")

def debug_print(fmt: str, *args, ordered=False, **kwargs) -> None:
  """Prints values and works in staged out JAX functions.

  Args:
    fmt: A format string, e.g. `"hello {x}"`, that will be used to format
      input arguments.
    *args: A list of positional arguments to be formatted.
    ordered: A keyword only argument used to indicate whether or not the
      staged out computation will enforce ordering of this `debug_print` w.r.t.
      other ordered `debug_print`s.
    **kwargs: Additional keyword arguments to be formatted.
  """
  effect = DebugEffect.ORDERED_PRINT if ordered else DebugEffect.PRINT
  debug_callback(functools.partial(_format_print_callback, fmt), effect, *args,
                 **kwargs)
