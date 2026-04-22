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
import functools
from typing import Any, TypeVar, cast, overload
from collections.abc import Hashable

from jax._src import api
from jax._src import checkify
from jax._src import config
from jax._src import core as jax_core
from jax._src import numpy as jnp
from jax._src import tree_util
from jax._src import typing as jax_typing
from jax._src import util
import jax._src.lax as lax
from jax._src.lax.control_flow import conditionals
from jax._src.pallas import core as pl_core
from jax._src.pallas import primitives as pl_primitives
from jax._src.pallas import utils as pl_utils


empty = api.named_call(lax.empty)


@api.named_call
def empty_like(x: object):
  """Create an empty PyTree of possibly uninitialized values.

  Args:
    x: A PyTree with leaves specifying the shape and dtype of
      the uninitialized object.

  Returns:
    A PyTree with the same structure as ``x``, but with uninitialized
    values.

  See Also:
    :func:`jax.lax.empty`
  """
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
) -> Callable[[Callable[[], None]], None]:
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


_T = TypeVar("_T")


@overload
def loop(
    lower: jax_typing.ArrayLike,
    upper: jax_typing.ArrayLike,
    *,
    init_carry: None = ...,
    step: jax_typing.ArrayLike = ...,
    unroll: int | bool | None = ...,
) -> Callable[[Callable[[jax_typing.Array], None]], None]:
  ...


@overload
def loop(
    lower: jax_typing.ArrayLike,
    upper: jax_typing.ArrayLike,
    *,
    init_carry: _T = ...,
    step: jax_typing.ArrayLike = ...,
    unroll: int | bool | None = ...,
) -> Callable[[Callable[[jax_typing.Array, _T], _T]], _T]:
  ...


def loop(
    lower: jax_typing.ArrayLike,
    upper: jax_typing.ArrayLike,
    *,
    init_carry: _T | None = None,
    step: jax_typing.ArrayLike = 1,
    unroll: int | bool | None = None,
) -> Callable[[Callable[..., _T | None]], _T | None]:
  """Returns a decorator that calls the decorated function in a loop."""
  zero: jax_typing.ArrayLike
  if not all(map(jax_core.is_concrete, (lower, upper, step))):
    idx_type = jnp.result_type(lower, upper, step)
    lower = lax.convert_element_type(lower, idx_type)
    upper = lax.convert_element_type(upper, idx_type)
    step = lax.convert_element_type(step, idx_type)
    zero = jnp.array(0, dtype=idx_type)
  else:
    # Preserve concrete bounds to allow loop unrolling.
    lower = cast(int, lower)
    upper = cast(int, upper)
    step = cast(int, step)
    zero = 0

  def decorator(body):
    if init_carry is None:
      body_fn = lambda idx, _: body(lower + idx * step)
    else:
      body_fn = lambda idx, carry: body(lower + idx * step, carry)
    return lax.fori_loop(
        zero,
        pl_utils.cdiv(upper - lower, step),
        body_fn,
        init_val=init_carry,
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

def _get_empty_ref(out):
  aval = pl_core._convert_out_shape_to_aval(out)
  mem_space = (None if isinstance(aval.memory_space, jax_core.MemorySpace)  # type: ignore
               else aval.memory_space)  # type: ignore
  val = lax.empty(aval.shape, aval.dtype, out_sharding=aval.sharding,  # type: ignore
                  _manual_axis_type=aval.manual_axis_type)  # type: ignore
  return jax_core.new_ref(val, memory_space=mem_space)


def _make_kernel(body,
                 out_type: object,
                 mesh: pl_core.Mesh,
                 scratch_types: pl_core.ScratchShapeTree = (),
                 name: str | None = None,
                 **mesh_kwargs
                 ):
  if unwrap_out := not isinstance(out_type, (tuple, list)):
    out_type = (out_type,)

  @api.jit
  def wrapper(*operands):
    arg_refs = tree_util.tree_map(jax_core.new_ref, operands)
    out_refs = tree_util.tree_map(_get_empty_ref, out_type)

    @pl_core.core_map(
        mesh,
        scratch_shapes=scratch_types,
        **mesh_kwargs,
        name=name or util.fun_name(body),
    )
    def _(*scratch_refs, **scratch_kwrefs):
      return body(*arg_refs, *out_refs, *scratch_refs, **scratch_kwrefs)

    outs = tree_util.tree_map(lambda ref: ref[...], out_refs)
    return outs[0] if unwrap_out else outs
  return wrapper


def kernel(body: Callable | api.NotSpecified = api.NotSpecified(),
           out_type: object | None = None,
           *,
           mesh: pl_core.Mesh,
           scratch_types: pl_core.ScratchShapeTree = (),
           compiler_params: pl_core.CompilerParams | None = None,
           interpret: bool = False,
           cost_estimate: pl_core.CostEstimate | None = None,
           debug: bool = False,
           name: str | None = None,
           metadata: dict[str, str] | None = None,
):
  """Entry point for creating a Pallas kernel.

  This is a convenience wrapper around ``core_map`` for executing a kernel
  over a mesh and ``run_scoped`` for allocating scratch memory.

  If ``body`` is provided, this function behaves as a decorator:

  .. code-block:: python

    def kernel_body(in_ref, out_ref):
      ...
    kernel = pl.kernel(kernel_body, out_shape=...)

  If ``body`` is omitted, this function behaves as a decorator factory and
  will return a decorator that can be used to annotate a kernel body:

  .. code-block:: python

    @pl.kernel(out_shape=...)
    def kernel(in_ref, out_ref):
      ...

  Args:
    body: The body of the kernel. If provided, this function behaves as a
      decorator, and if omitted, this function behaves as a decorator factory.
    out_shape: The shape of the output. Should be a PyTree of
      ``jax.ShapeDtypeStruct`` or ``jax.Array`` s.
    mesh: The mesh to run the kernel on.
    scratch_shapes: The shapes of the scratch arrays.
    compiler_params: The compiler parameters to pass to the backend.
    interpret: Whether to run the function in interpret mode.
    debug: Whether or not to out helpful debugging information.
    cost_estimate: The cost estimate of the function.
    name: The (optional) name of the kernel.
    metadata: Optional dictionary of information about the kernel that will be
      serialized as JSON in the HLO. Can be used for debugging and analysis.

  Returns:
    If ``body`` is provided, returns a function that runs the kernel.
    It should take any number of input operands and returns an output with the
    same PyTree structure as `out_shape`.
    If ``body`` is omitted, returns a decorator that can be used to annotate
    a kernel body.
  """
  # Note we default out_shape to None to allow `body` to come before it
  # in the function signature, but `body` itself is optional.
  if out_type is None:
    raise ValueError('out_type must be provided.')
  kwds = dict(
      out_type=out_type,
      mesh=mesh,
      scratch_types=scratch_types,
      compiler_params=compiler_params,
      interpret=interpret,
      cost_estimate=cost_estimate,
      debug=debug,
      name=name,
      metadata=metadata)
  if isinstance(body, api.NotSpecified):
    return lambda fun: _make_kernel(fun, **kwds)
  else:
    return _make_kernel(body, **kwds)


def with_scoped(
    *types: Any,
    collective_axes: Hashable | tuple[Hashable, ...] = (),
    **kw_types: Any,
):
  """Returns a function decorator that runs a function with provided allocations.

  Example::

    @pl.with_scoped(pltpu.VMEM((8, 128), jnp.float32),
                    sem_ref=pltpu.SemaphoreType.DMA)
    def f(vmem_ref, sem_ref):
      ...

    f()

  The arguments to `f` will be forwarded to the decorated function as the
  initial arguments.

  Example::

    @pl.with_scoped(pltpu.VMEM((8, 128), jnp.float32),
                    sem_ref=pltpu.SemaphoreType.DMA)
    def f(outer_ref, vmem_ref, sem_ref):
      ...

    outer_ref = ...
    f(outer_ref)

  """
  def decorator(f):
    def inner(*args):
      return pl_primitives.run_scoped(
          functools.partial(f, *args),
          *types,
          collective_axes=collective_axes,
          **kw_types,
      )
    return inner
  return decorator
