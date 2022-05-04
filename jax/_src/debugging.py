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
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir

DebugEffect = enum.Enum('DebugEffect', ['PRINT', 'ORDERED_PRINT'])

core.ordered_effects.add(DebugEffect.ORDERED_PRINT)
mlir.lowerable_effects.add(DebugEffect.PRINT)
mlir.lowerable_effects.add(DebugEffect.ORDERED_PRINT)

# `debug_callback_p` is the main primitive for staging out Python callbacks.
debug_callback_p = core.Primitive('debug_callback')
debug_callback_p.multiple_results = True

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

def debug_callback_batching_rule(*flat_args, callback: Callable[..., Any],
    effect: DebugEffect, in_tree: tree_util.PyTreeDef):
  del flat_args, callback, effect, in_tree
  # TODO(sharadmv): implement batching rule
  raise NotImplementedError('Batching not supported for `debug_callback`.')
batching.primitive_batchers[debug_callback_p] = debug_callback_batching_rule

def debug_callback_jvp_rule(*flat_args, callback: Callable[..., Any],
    effect: DebugEffect, in_tree: tree_util.PyTreeDef):
  del flat_args, callback, effect, in_tree
  # TODO(sharadmv): implement jvp rule
  raise NotImplementedError('JVP not supported for `debug_callback`.')
ad.primitive_jvps[debug_callback_p] = debug_callback_jvp_rule

def debug_callback_transpose_rule(*flat_args, callback: Callable[..., Any],
    effect: DebugEffect, in_tree: tree_util.PyTreeDef):
  del flat_args, callback, effect, in_tree
  # TODO(sharadmv): implement transpose rule
  raise NotImplementedError('Transpose not supported for `debug_callback`.')
ad.primitive_transposes[debug_callback_p] = debug_callback_transpose_rule

def _ordered_effect_lowering(ctx, token, *args, **params):
  avals_in = [core.abstract_token, *ctx.avals_in]
  avals_out = [core.abstract_token, *ctx.avals_out]
  args = (token, *args)
  def _callback(token, *flat_args):
    out = debug_callback_p.impl(*flat_args, **params)
    return (token, *out)
  (token, *result), keepalive = mlir.emit_python_callback(
      ctx.module_context.platform, _callback, list(args), avals_in, avals_out,
      True)
  return result, keepalive, token

def debug_callback_lowering(ctx, *args, effect, callback, **params):
  if effect in core.ordered_effects:
    token = ctx.tokens_in.get(effect)[0]
    result, keepalive, token = _ordered_effect_lowering(ctx, token,
        *args, effect=effect, callback=callback, **params)
    ctx.set_tokens_out(mlir.TokenSet({effect: (token,)}))
  else:
    def _callback(*flat_args):
      return tuple(debug_callback_p.impl(
        *flat_args, effect=effect, callback=callback, **params))
    result, keepalive = mlir.emit_python_callback(ctx.module_context.platform,
      _callback, list(args), ctx.avals_in, ctx.avals_out,  True)
  ctx.module_context.add_keepalive(keepalive)
  return result
mlir.register_lowering(debug_callback_p, debug_callback_lowering,
                       platform="cpu")

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
