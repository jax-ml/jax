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
"""Pallas helper functions."""

from collections.abc import Callable


from jax._src import api
from jax._src import checkify
from jax._src import config
from jax._src import core as jax_core
from jax._src import tree_util
from jax._src import typing as jax_typing
import jax._src.lax as lax
from jax._src.lax.control_flow import conditionals
from jax._src.pallas import core as pl_core
from jax._src.pallas import utils as pl_utils
from jax._src import numpy as jnp


empty = api.named_call(lax.empty)


@api.named_call
def empty_like(x: object):
  return tree_util.tree_map(lambda leaf: empty(leaf.shape, leaf.dtype), x)


def empty_ref_like(x: object) -> jax_typing.Array:
  """Returns an empty array Ref with same shape/dtype/memory space as x."""
  match x:
    case pl_core.MemoryRef():
      memory_space = x.memory_space
    case jax_core.ShapeDtypeStruct():
      memory_space = pl_core.MemorySpace.ANY
    case _:
      raise ValueError(f'empty_ref_like does not support {type(x)}')
  return jax_core.new_ref(empty_like(x), memory_space=memory_space)


def when(
    condition: bool | jax_typing.ArrayLike, /
) -> Callable[[Callable[[], None]], Callable[[], None]]:
  """Calls the decorated function when the condition is met.

  Args:
    condition: If a boolean, this is equivalent to ``if condition: f()``. If an
      array, ``when`` produces a :func:`jax.lax.cond` with the decorated
      function as the true branch.

  Returns:
    A decorator.
  """
  def _wrapped(f):
    if isinstance(condition, bool):
      if condition:
        f()
    else:
      conditionals.cond(condition, f, lambda: None)
  return _wrapped


def loop(
    lower: jax_typing.ArrayLike,
    upper: jax_typing.ArrayLike,
    *,
    step: jax_typing.ArrayLike = 1,
    unroll: int | bool | None = None,
) -> Callable[[Callable[[jax_typing.Array], None]], None]:
  """Returns a decorator that calls the decorated function in a loop."""
  zero: jax_typing.ArrayLike
  if not all(map(jax_core.is_concrete, (lower, upper, step))):
    idx_type = jnp.result_type(lower, upper, step)
    lower = lax.convert_element_type(lower, idx_type)
    upper = lax.convert_element_type(upper, idx_type)
    step = lax.convert_element_type(step, idx_type)
    zero = jnp.array(0, dtype=idx_type)
  else:
    zero = 0

  def decorator(body):
    lax.fori_loop(
        zero,
        pl_utils.cdiv(upper - lower, step),
        lambda idx, _: body(lower + idx * step),
        init_val=None,
        unroll=unroll,
    )

  return decorator


_ENABLE_DEBUG_CHECKS = config.bool_state(
    "jax_pallas_enable_debug_checks",
    default=False,
    help=(
        "If set, ``pl.debug_check`` calls are checked at runtime. Otherwise,"
        " they are a noop."
    ),
)


enable_debug_checks = _ENABLE_DEBUG_CHECKS


def debug_checks_enabled() -> bool:
  """Returns runtime checks are enabled."""
  return _ENABLE_DEBUG_CHECKS.value


def debug_check(condition, message):
  """Check the condition if
  :func:`~jax.experimental.pallas.enable_debug_checks` is set, otherwise
  do nothing.
  """
  return checkify.debug_check(condition, message)
