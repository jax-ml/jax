# Copyright 2026 The JAX Authors.
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

"""`lax.pipeline` and its block reads and writes.

`pipeline` loops over a static grid. Its body reads tiles with `block_slice`
and writes carry tiles with `block_update`, and its value is that of a
`fori_loop`. A platform lowering may run the loop as a software pipeline, as
`compiler_params` configure.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import functools
import math
import operator
import threading
from typing import Any

from jax._src import api_util
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import flattree as ft
from jax._src import source_info_util
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import compiler_params as cp
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lax.control_flow import loops
from jax._src.traceback_util import api_boundary
from jax._src.typing import Array, ArrayLike  # pylint: disable=g-multiple-import
from jax._src.util import safe_map, safe_zip, split_list  # pylint: disable=g-multiple-import
import numpy as np

map, unsafe_map = safe_map, map  # pylint: disable=redefined-builtin
zip, unsafe_zip = safe_zip, zip  # pylint: disable=redefined-builtin


@contextlib.contextmanager
def _temporarily_set(
    state: threading.local, attr: str, value: Any
) -> Iterator[None]:
  """Sets `state.attr` to `value` within the context."""
  prev = getattr(state, attr)
  setattr(state, attr, value)
  try:
    yield
  finally:
    setattr(state, attr, prev)


### Compiler params


class PipelineParams:
  """Base class for `lax.pipeline` compiler parameters.

  Platform parameter types subclass `PipelineParams` as frozen dataclasses and
  register a platform lowering handler where needed.
  """


### Which trace is a pipeline body.


def body_only_placement_message(name: str) -> str:
  """The error for a body only primitive used outside a pipeline body."""
  return (
      f"{name} may only appear directly in the body of a lax.pipeline, not at"
      " the top level and not inside a jit, cond, loop, vmap or call within"
      " the body."
  )


class _BodyTraceState(threading.local):
  traces: tuple[core.Trace, ...] = ()


_body_trace_state = _BodyTraceState()


def _tracing_pipeline_body() -> contextlib.AbstractContextManager[None]:
  traces = (*_body_trace_state.traces, core.trace_ctx.trace)
  return _temporarily_set(_body_trace_state, "traces", traces)


def check_directly_in_pipeline_body(api_name: str) -> None:
  """Raises `TypeError` unless the current trace is a `lax.pipeline` body.

  A `jit`, `cond`, loop, or call within the body has a trace of its own.

  Args:
    api_name: Public API name used in the error message.
  """
  traces = _body_trace_state.traces
  if not traces or core.trace_ctx.trace is not traces[-1]:
    raise TypeError(body_only_placement_message(api_name))


### Primitives


def _check_bool(where: str, name: str, value: Any) -> None:
  if not isinstance(value, bool):
    raise TypeError(f"{where}: {name} must be a bool, got {value!r}.")


def _check_index_rank(prim: str, operand: Any, index: Sequence[Any]) -> None:
  if len(index) != operand.ndim:
    raise TypeError(
        f"{prim}: index must hold one scalar per dimension of the"
        f" {operand.ndim}D operand, got {len(index)} entries."
    )


def _block_slice_abstract_eval(src, *index, block_shape, revisits_tile):
  _check_bool("block_slice", "revisits_tile", revisits_tile)
  if not isinstance(block_shape, tuple) or len(block_shape) != src.ndim:
    raise TypeError(
        "block_slice: block_shape must be a tuple with one size per dimension"
        f" of the {src.ndim}D source, got {block_shape!r}."
    )
  _check_index_rank("block_slice", src, index)
  aval, _ = slicing.dynamic_slice_p.abstract_eval(
      src, *index, slice_sizes=block_shape
  )
  return aval


block_slice_p = core.Primitive("block_slice")
block_slice_p.def_impl(
    functools.partial(dispatch.apply_primitive, block_slice_p)
)
block_slice_p.def_abstract_eval(_block_slice_abstract_eval)


def _block_update_abstract_eval(dst, tile, *index, disjoint):
  _check_bool("block_update", "disjoint", disjoint)
  aval, _ = slicing.dynamic_update_slice_p.abstract_eval(dst, tile, *index)
  return aval


block_update_p = core.Primitive("block_update")
block_update_p.def_impl(
    functools.partial(dispatch.apply_primitive, block_update_p)
)
block_update_p.def_abstract_eval(_block_update_abstract_eval)


# Primitives legal only directly in a pipeline body. Register more with
# `register_pipeline_body_primitive`.
PIPELINE_BODY_ONLY_PRIMITIVES: set[core.Primitive] = set()

# `rule(*operands, **params)` evaluates a body only primitive with ordinary
# primitives and returns its outputs as a list, for the default lowering.
PLAIN_EVAL_RULES: dict[core.Primitive, Callable[..., list[Any]]] = {}


class _TypecheckState(threading.local):
  # While the typecheck of a pipeline_p checks its body: the operand lists,
  # as tuples of atom ids, of the body only equations directly in that body.
  direct_operands: frozenset[tuple[int, ...]] = frozenset()
  # Inside `checking_pipeline_bodies`.
  lenient: int = 0


_typecheck_state = _TypecheckState()


@contextlib.contextmanager
def checking_pipeline_bodies() -> Iterator[None]:
  """Accepts body only primitives anywhere in jaxprs checked in this context.

  For work that checks a body jaxpr on its own, such as tracing or dead code
  elimination. The `pipeline_p` typecheck and every pipeline lowering check
  placement again, and the lowering check also covers results cached here.
  """
  with _temporarily_set(
      _typecheck_state, "lenient", _typecheck_state.lenient + 1
  ):
    yield


def _typechecking_body(
    jaxpr: core.Jaxpr,
) -> contextlib.AbstractContextManager[None]:
  direct = frozenset(
      tuple(map(id, eqn.invars))
      for eqn in jaxpr.eqns
      if eqn.primitive in PIPELINE_BODY_ONLY_PRIMITIVES
  )
  return _temporarily_set(_typecheck_state, "direct_operands", direct)


def register_pipeline_body_primitive(
    prim: core.Primitive, plain_eval_rule: Callable[..., list[Any]]
) -> None:
  """Registers `prim` as legal directly in a `lax.pipeline` body.

  `core.check_jaxpr` then raises `JaxprTypeError` for `prim` elsewhere and
  checks its operands with `prim.abstract_eval`.

  Args:
    prim: Body only primitive to register.
    plain_eval_rule: Rule evaluating `prim` with ordinary primitives for the
      default lowering.
  """

  def typecheck(_, *in_atoms, **params):
    if (
        not _typecheck_state.lenient
        and tuple(map(id, in_atoms)) not in _typecheck_state.direct_operands
    ):
      raise core.JaxprTypeError(body_only_placement_message(prim.name))
    try:
      out = prim.abstract_eval(*(x.aval for x in in_atoms), **params)
    except (TypeError, ValueError) as e:
      raise core.JaxprTypeError(str(e)) from e
    out_avals, out_effects = out
    return (out_avals if prim.multiple_results else [out_avals]), out_effects

  PIPELINE_BODY_ONLY_PRIMITIVES.add(prim)
  PLAIN_EVAL_RULES[prim] = plain_eval_rule
  core.custom_typechecks[prim] = typecheck


register_pipeline_body_primitive(
    block_slice_p,
    lambda src, *index, block_shape, revisits_tile: [
        slicing.dynamic_slice_p.bind(src, *index, slice_sizes=block_shape)
    ],
)
register_pipeline_body_primitive(
    block_update_p,
    lambda dst, tile, *index, disjoint: [
        slicing.dynamic_update_slice_p.bind(dst, tile, *index)
    ],
)


# The loop of `lax.pipeline`.
#
# Operands: `num_consts` loop invariant constants, then one initial value per
# carry. Results: the final carries.
#
# Parameters:
#   grid: the static grid, a tuple of non negative Python ints. The body runs
#     once per grid point, in row major order.
#   num_consts: the number of leading operands that are loop invariant.
#   jaxpr: the body. Its inputs are the `num_consts` constants, one int32
#     scalar coordinate per grid dimension, then the carries. It returns the
#     next carries, typed as the carry inputs. `jaxpr.consts` are extra loop
#     invariants that precede all of these.
#   compiler_params: None or a `PipelineParams`, read by its platform's
#     lowering.
def _pipeline_is_effectful(params: Mapping[str, Any]) -> bool:
  return bool(params["jaxpr"].effects)


pipeline_p = core.Primitive("pipeline")
pipeline_p.multiple_results = True
pipeline_p.skip_canonicalization = True
pipeline_p.is_effectful = _pipeline_is_effectful  # type: ignore[assignment]


def _pipeline_abstract_eval(*avals, grid, num_consts, jaxpr, compiler_params):
  del avals, grid, num_consts, compiler_params
  _check_body_effects(jaxpr.effects)
  # Sound only because `_check_body_effects` rejects input effects.
  return jaxpr.out_avals, core.positional_effects(jaxpr)


pipeline_p.def_impl(functools.partial(dispatch.apply_primitive, pipeline_p))
pipeline_p.def_effectful_abstract_eval(_pipeline_abstract_eval)


### Public API


_MAX_TRIP_COUNT = int(np.iinfo(np.int32).max)


def _check_trip_count(where: str, grid: Any, sizes: tuple[int, ...]) -> None:
  if math.prod(sizes) > _MAX_TRIP_COUNT:
    raise ValueError(
        f"{where}: the grid {grid!r} has more steps than an int32 can count."
    )


def _canonicalize_grid(grid: Any) -> tuple[int, ...]:
  """Validates `grid` and returns it as a tuple of Python ints."""
  is_sequence = isinstance(grid, Sequence) and not isinstance(
      grid, (str, bytes)
  )

  def size(d: Any) -> int:
    try:
      if isinstance(d, (bool, np.bool_)):
        raise TypeError
      n = operator.index(d)
    except TypeError:
      raise TypeError(
          "lax.pipeline: grid must be a static non negative int or a sequence"
          f" of them, got {grid!r}."
      ) from None
    if n < 0:
      raise ValueError(
          f"lax.pipeline: grid sizes must be non negative, got {grid!r}."
      )
    return int(n)

  out = tuple(map(size, grid if is_sequence else (grid,)))
  _check_trip_count("lax.pipeline", grid, out)
  return out


COORD_AVAL = core.ShapedArray((), np.dtype(np.int32))


@api_boundary
def pipeline(
    body: Callable[..., Any],
    *,
    grid: int | Sequence[int],
    init: Any,
    compiler_params: Any = None,
) -> Any:
  """Runs ``body`` over a static grid, threading a carry.

  The value is that of this Python loop, with one int32 coordinate per grid
  dimension, in row major order::

    def pipeline(body, grid, init):
      acc = init
      for coords in itertools.product(*map(range, grid)):
        acc = body(*coords, acc)
      return acc

  In ``body``, :func:`block_slice` reads a tile of an array and
  :func:`block_update` writes a tile of the carry. A platform lowering may
  overlap one step's block reads and writes with another step's compute, as
  ``compiler_params`` configure for that platform.

  ``jax.grad`` and ``jax.jvp`` work when no tangent reaches the ``pipeline``,
  that is when ``init`` and the values ``body`` closes over are independent
  of the differentiated inputs, and raise ``NotImplementedError`` otherwise.
  To differentiate through it, wrap it in a ``jax.custom_vjp`` function.
  ``jax.vmap`` raises ``NotImplementedError``.

  Args:
    body: function ``(*coords, carry) -> carry``, returning a carry with the
      pytree structure and types of ``init``.
    grid: a static non negative int or sequence of them. The total number of
      steps fits in an int32.
    init: the initial carry, a pytree of arrays.
    compiler_params: None, or a frozen dataclass subclassing
      ``jax._src.lax.control_flow.pipeline.PipelineParams``.

  Returns:
    The carry after the last grid step.

  See Also:
    :func:`jax.experimental.pallas.tpu.emit_pipeline`, which pipelines copies
    between memory spaces inside a Pallas TPU kernel.
  """
  if not callable(body):
    raise TypeError("lax.pipeline: body must be callable.")
  grid_ = _canonicalize_grid(grid)
  params = cp.check_compiler_params(
      compiler_params, "lax.pipeline", PipelineParams
  )

  init_flat = ft.flatten(init)
  dbg = api_util.debug_info("pipeline", body, (*range(len(grid_)), init), {})
  del init
  coord_fts = tuple(ft.flatten(COORD_AVAL) for _ in grid_)

  def traced_body(*args):
    with _tracing_pipeline_body():
      return body(*args)

  def trace(carry_avals):
    in_avals = ft.pack(((*coord_fts, carry_avals), {}))
    with checking_pipeline_bodies():
      return pe.trace_to_jaxpr(traced_body, in_avals, dbg)

  # As in `scan`: retrace once if a weakly typed init value must be promoted
  # to the type the body returns.
  init_avals = init_flat.map(core.typeof)
  jaxpr, out_avals = trace(init_avals)
  if len(out_avals) != len(init_avals):
    loops._check_carry_type("pipeline body", None, init_avals, out_avals)  # pylint: disable=protected-access
  init_flat, changed = init_flat.map3(
      list(init_avals),
      list(out_avals),
      loops._promote_weak_typed_input,  # pylint: disable=protected-access
  ).unzip2()
  if any(changed):
    init_avals = init_flat.map(core.typeof)
    jaxpr, out_avals = trace(init_avals)
  loops._check_carry_type("pipeline body", None, init_avals, out_avals)  # pylint: disable=protected-access
  _check_body_effects(jaxpr.effects)

  jaxpr, consts = pe.separate_consts(jaxpr)
  outs = pipeline_p.bind(
      *consts,
      *init_flat.vals,
      grid=grid_,
      num_consts=len(consts),
      jaxpr=jaxpr,
      compiler_params=params,
  )
  return out_avals.update(outs).unflatten()


def _check_body_effects(body_effects: effects.Effects) -> None:
  """Rejects disallowed, ordered, and input effects in a pipeline body."""
  disallowed = effects.control_flow_allowed_effects.filter_not_in(body_effects)
  if disallowed:
    raise NotImplementedError(
        f"Effects not supported in lax.pipeline: {disallowed}"
    )
  ordered = effects.ordered_effects.filter_in(body_effects)
  if ordered:
    raise NotImplementedError(
        f"Ordered effects are not supported in lax.pipeline: {ordered}"
    )
  # Input effects index the body inputs (*consts, *coords, *carries), which
  # differ from the pipeline operands (*consts, *init).
  inputs = [e for e in body_effects if isinstance(e, effects.JaxprInputEffect)]
  if inputs:
    raise NotImplementedError(
        "Input effects (Ref effects) are not supported in lax.pipeline:"
        f" {inputs}"
    )


def _canonicalize_index(
    api_name: str, x: Any, index: Sequence[ArrayLike]
) -> list[Any]:
  """Validates and normalizes `index` for `x`."""
  if not isinstance(index, Sequence) or len(index) != np.ndim(x):
    raise ValueError(
        f"{api_name}: index must hold one scalar per dimension of the"
        f" {np.ndim(x)}D array, got {index!r}."
    )
  if not all(dtypes.issubdtype(dtypes.dtype(i), np.integer) for i in index):
    raise TypeError(
        f"{api_name}: index entries must be integers, got {index!r}."
    )
  # Python ints take the dtype of the traced entries, so `(i * 16, 0)` works
  # whatever the default int width.
  strong = [dtypes.dtype(i) for i in index if not dtypes.is_weakly_typed(i)]
  if len(set(strong)) > 1:
    raise TypeError(
        f"{api_name}: index entries must share one integer dtype, got"
        f" {', '.join(d.name for d in strong)}."
    )
  common = strong[0] if strong else np.dtype(np.int32)
  index = [lax.convert_element_type(i, common) for i in index]
  return slicing._dynamic_slice_indices(x, index, True)  # pylint: disable=protected-access


@api_boundary
def block_slice(
    src: ArrayLike,
    block_shape: Sequence[int],
    index: Sequence[ArrayLike],
    *,
    revisits_tile: bool = True,
) -> Array:
  """Reads the tile of ``src`` at ``index`` in a :func:`pipeline` body.

  The value is ``lax.dynamic_slice(src, index, block_shape)``, which clamps
  out of bounds starts and reads negative starts from the end. A platform
  lowering may start the read in an earlier grid step.

  Legal directly in a :func:`pipeline` body, where a lowering can overlap the
  read with other grid steps; raises ``TypeError`` elsewhere.

  Args:
    src: the array to read.
    block_shape: the static shape of the tile, one size per dimension.
    index: the start of the tile, one integer scalar per dimension.
    revisits_tile: False promises that consecutive grid steps read different
      tiles, so the lowering may reuse the tile's buffer at once.

  Returns:
    The tile.
  """
  api_name = "lax.block_slice"
  check_directly_in_pipeline_body(api_name)
  _check_bool(api_name, "revisits_tile", revisits_tile)
  block_shape = core.canonicalize_shape(block_shape)
  index = _canonicalize_index(api_name, src, index)
  src, *index = core.auto_insert_reshard(src, *index)
  return block_slice_p.bind(
      src,
      *index,
      block_shape=tuple(block_shape),
      revisits_tile=revisits_tile,
  )


@api_boundary
def block_update(
    dst: ArrayLike,
    tile: ArrayLike,
    index: Sequence[ArrayLike],
    *,
    disjoint: bool = True,
) -> Array:
  """Writes ``tile`` into ``dst`` at ``index`` in a :func:`pipeline` body.

  The value is ``lax.dynamic_update_slice(dst, tile, index)``. A platform
  lowering may finish the write in a later grid step.

  Legal directly in a :func:`pipeline` body; raises ``TypeError`` elsewhere.

  Args:
    dst: the array to write into, usually the carry.
    tile: the values to write; same dtype and rank as ``dst``.
    index: the start of the tile, one integer scalar per dimension.
    disjoint: True promises that each grid step reads only regions of ``dst``
      outside the pending writes of other steps, so the lowering may keep reads
      of ``dst`` in flight while the write completes.

  Returns:
    ``dst`` with the tile written.
  """
  api_name = "lax.block_update"
  check_directly_in_pipeline_body(api_name)
  _check_bool(api_name, "disjoint", disjoint)
  index = _canonicalize_index(api_name, dst, index)
  dst, tile, *index = core.auto_insert_reshard(dst, tile, *index)
  return block_update_p.bind(dst, tile, *index, disjoint=disjoint)


### Typecheck


@contextlib.contextmanager
def _as_jaxpr_type_error(*errors: type[Exception]) -> Iterator[None]:
  try:
    yield
  except errors as e:
    raise core.JaxprTypeError(str(e)) from e


def _pipeline_typecheck(_, *in_atoms, grid, num_consts, jaxpr, compiler_params):
  """Typechecks a `pipeline_p` equation and its body jaxpr."""
  if not isinstance(grid, tuple) or not all(
      type(d) is int and d >= 0  # pylint: disable=unidiomatic-typecheck
      for d in grid
  ):
    raise core.JaxprTypeError(
        f"pipeline grid must be a tuple of non negative ints, got {grid!r}"
    )
  with _as_jaxpr_type_error(TypeError, ValueError, NotImplementedError):
    cp.check_compiler_params(compiler_params, "pipeline", PipelineParams)
    _check_trip_count("pipeline", grid, grid)
    _check_body_effects(jaxpr.effects)
  num_carries = len(jaxpr.outvars)
  if len(in_atoms) != num_consts + num_carries:
    raise core.JaxprTypeError(
        f"pipeline has {len(in_atoms)} operands but num_consts={num_consts}"
        f" and the body returns {num_carries} carries"
    )
  if len(jaxpr.invars) != num_consts + len(grid) + num_carries:
    raise core.JaxprTypeError(
        f"pipeline body takes {len(jaxpr.invars)} inputs, expected"
        f" {num_consts} constants, {len(grid)} coordinates (one per grid"
        f" dimension) and {num_carries} carries"
    )
  const_avals, coord_avals, carry_avals = split_list(
      jaxpr.in_avals, [num_consts, len(grid)]
  )
  in_const_avals, in_carry_avals = split_list(
      [x.aval for x in in_atoms], [num_consts]
  )
  if not all(map(core.typecompat, const_avals, in_const_avals)):
    raise core.JaxprTypeError(
        "pipeline constant operands do not match the body constant inputs"
    )
  if not all(
      isinstance(a, core.ShapedArray) and not a.shape and a.dtype == np.int32
      for a in coord_avals
  ):
    raise core.JaxprTypeError(
        "pipeline body coordinate inputs must be int32 scalars, got"
        f" {coord_avals}"
    )
  if not all(map(core.typecompat, carry_avals, in_carry_avals)):
    raise core.JaxprTypeError(
        "pipeline init operands do not match the body carry inputs"
    )
  if not all(map(core.typematch, carry_avals, jaxpr.out_avals)):
    raise core.JaxprTypeError(
        f"pipeline body carry outputs {jaxpr.out_avals} do not match its carry"
        f" inputs {carry_avals}"
    )
  with _as_jaxpr_type_error(ValueError):
    check_body_only_placement(jaxpr)
  with _typechecking_body(jaxpr):
    core.check_jaxpr(jaxpr)
  # Sound only because `_check_body_effects` rejects input effects.
  return jaxpr.out_avals, core.positional_effects(jaxpr)


def _nested_non_pipeline_eqns(jaxpr: core.Jaxpr) -> Iterator[core.JaxprEqn]:
  """Yields equations in `jaxpr` and nested subjaxprs, skipping `pipeline_p`."""
  for eqn in jaxpr.eqns:
    yield eqn
    if eqn.primitive is pipeline_p:
      continue
    for sub in core.jaxprs_in_params(eqn.params):
      yield from _nested_non_pipeline_eqns(sub)


def check_body_only_placement(jaxpr: core.Jaxpr) -> None:
  """Raises `ValueError` for a body only primitive nested in `jaxpr`'s eqns.

  `jaxpr` is a pipeline body. The typecheck and the lowerings call this, since
  some typecheck rules, such as `cond`'s, skip their jaxprs, and lowering
  typechecks only with checks enabled.

  Args:
    jaxpr: Pipeline body jaxpr to inspect.
  """
  for eqn in jaxpr.eqns:
    if eqn.primitive is pipeline_p:
      continue  # Its own typecheck and lowering check its body.
    for sub in core.jaxprs_in_params(eqn.params):
      for e in _nested_non_pipeline_eqns(sub):
        if e.primitive in PIPELINE_BODY_ONLY_PRIMITIVES:
          raise ValueError(body_only_placement_message(e.primitive.name))


core.custom_typechecks[pipeline_p] = _pipeline_typecheck


### Default lowering: a plain loop of dynamic slices.


def unravel_grid_index(flat, grid, rem, div, const):
  """Returns the row major coordinates of the flat grid step `flat`."""
  if len(grid) <= 1:
    return [flat] * len(grid)
  coords = [None] * len(grid)
  for d in range(len(grid) - 1, 0, -1):
    extent = const(max(grid[d], 1))
    coords[d] = rem(flat, extent)
    flat = div(flat, extent)
  coords[0] = flat
  return coords


def _eval_body_plain(jaxpr: core.Jaxpr, *args):
  """Evaluates the body with its block ops as dynamic slices."""
  env: dict[core.Var, Any] = {}
  read = lambda v: v.val if isinstance(v, core.Literal) else env[v]
  map(env.__setitem__, jaxpr.all_invars, [*jaxpr.consts, *args])
  for eqn in jaxpr.eqns:
    invals = map(read, eqn.invars)
    name_stack = (
        source_info_util.current_name_stack() + eqn.source_info.name_stack
    )
    with (
        source_info_util.user_context(
            eqn.source_info.traceback, name_stack=name_stack
        ),
        eqn.ctx.manager,
    ):
      if (rule := PLAIN_EVAL_RULES.get(eqn.primitive)) is not None:
        outs = rule(*invals, **eqn.params)
      else:
        bind_params = eqn.primitive.get_bind_params(eqn.params)
        outs = eqn.primitive.bind(*invals, **bind_params)
        outs = outs if eqn.primitive.multiple_results else [outs]
    map(env.__setitem__, eqn.outvars, outs)
  return map(read, jaxpr.outvars)


def _pipeline_impl(*args, grid, num_consts, jaxpr, compiler_params):
  """Evaluates `pipeline_p` as a `while_loop` of plain dynamic slices."""
  del compiler_params
  consts, init = split_list(args, [num_consts])
  trip_count = math.prod(grid)

  def cond(state):
    return state[0] < np.int32(trip_count)

  def step(state):
    flat, carry = state
    coords = unravel_grid_index(flat, grid, lax.rem, lax.div, np.int32)
    return flat + np.int32(1), _eval_body_plain(jaxpr, *consts, *coords, *carry)

  _, out = loops.while_loop(cond, step, (np.int32(0), list(init)))
  return out


def _pipeline_lowering(ctx, *args, jaxpr, **params):
  check_body_only_placement(jaxpr)
  return mlir.lower_fun(_pipeline_impl)(ctx, *args, jaxpr=jaxpr, **params)


def _misplaced_lowering(prim: core.Primitive) -> Callable[..., Any]:
  """Returns a default lowering for `prim` that raises the placement error."""
  # The default pipeline lowering uses `PLAIN_EVAL_RULES` for its body.

  def lowering(ctx, *args, **params):
    del ctx, args, params
    raise ValueError(body_only_placement_message(prim.name))

  return lowering


mlir.register_lowering(block_slice_p, _misplaced_lowering(block_slice_p))
mlir.register_lowering(block_update_p, _misplaced_lowering(block_update_p))
# Caching is per primitive, and platform rules may read enclosing body state.
mlir.register_lowering(pipeline_p, _pipeline_lowering, cacheable=False)
