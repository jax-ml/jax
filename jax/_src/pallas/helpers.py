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
import jax.numpy as jnp
from jax._src import core as jax_core
from jax._src import checkify
from jax._src import config
from jax._src.pallas import core as pl_core
from jax._src.pallas import utils as pl_utils
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
  if hasattr(x, 'memory_space'):
    if memory_space is not None:
      raise ValueError(
          'memory_space cannot be specified for a MemoryRef object.'
      )
    memory_space = x.memory_space
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


def empty_ref_like(
    x: object, *, backend: pl_core.Backend | None = None
) -> jax.Array:
  """Returns an empty array Ref with same shape/dtype/memory space as x."""
  match x:
    case pl_core.MemoryRef():
      memory_space = x.memory_space
    case jax.ShapeDtypeStruct():
      memory_space = pl_core.MemorySpace.ANY
    case _:
      raise ValueError(f'alloc_ref does not support {type(x)}')
  out = empty_like(x, backend=backend)
  return jax_core.mutable_array(out, memory_space=memory_space)


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
    step: jax.typing.ArrayLike = 1,
    unroll: int | bool | None = None,
) -> Callable[[Callable[[jax.Array], None]], None]:
  """Returns a decorator that calls the decorated function in a loop."""
  idx_type = jnp.result_type(lower, upper, step)
  lower = jax.lax.convert_element_type(lower, idx_type)
  upper = jax.lax.convert_element_type(upper, idx_type)
  step = jax.lax.convert_element_type(step, idx_type)

  def decorator(body):
    jax.lax.fori_loop(
        0,
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
