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

import jax
from jax._src import checkify
from jax._src import config
from jax._src.pallas import core as pl_core
from jax._src.pallas import pallas_call


@jax.named_call
def empty(
    shape: tuple[int, ...],
    dtype: jax.typing.DTypeLike,
    *,
    memory_space: object | None = None,
    interpret: bool = False,
    backend: pl_core.Backend | None = None,
):
  return empty_like(
      jax.ShapeDtypeStruct(shape, dtype),
      memory_space=memory_space,
      interpret=interpret,
      backend=backend,
  )


@jax.named_call
def empty_like(
    x: object,
    *,
    memory_space: object | None = None,
    interpret: bool = False,
    backend: pl_core.Backend | None = None,
):
  if memory_space is None:
    memory_space = pl_core.MemorySpace.ANY
  return pallas_call.pallas_call(
      # No-op to leave the out_ref uninitialized
      lambda *_: None,
      out_specs=jax.tree.map(
          lambda _: pl_core.BlockSpec(memory_space=memory_space), x
      ),
      out_shape=x,
      interpret=interpret,
      backend=backend,
  )()


def when(condition):
  def _wrapped(f):
    if isinstance(condition, bool):
      if condition:
        f()
    else:
      jax.lax.cond(condition, f, lambda: None)
  return _wrapped


def loop(
    lower: jax.typing.ArrayLike,
    upper: jax.typing.ArrayLike,
    *,
    unroll: int | bool | None = None,
) -> Callable[[Callable[[jax.Array], None]], None]:
  def decorator(body):
    jax.lax.fori_loop(
        lower, upper, lambda idx, _: body(idx), init_val=None, unroll=unroll
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
