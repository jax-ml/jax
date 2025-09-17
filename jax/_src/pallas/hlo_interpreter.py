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
"""HLO interpreter for Pallas kernels.

The interpret mode for Pallas emulates the behavior of a Pallas kernel
by producing an equivalent HLO program. This involves several steps that
are carried out in stages:

1) Resolve Pallas-specific dtypes (e.g. Semaphores) to a suitable
 HLO type (e.g. int).
2) Discharge stateful operations.
3) Evaluate the body of the kernel in a loop.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import reduce, partial
import itertools
from typing import Any
from collections.abc import Callable

import jax
from jax import lax
from jax._src import core as jax_core
from jax._src import frozen_dict
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives
from jax._src import state
from jax._src.state import discharge as state_discharge
from jax._src import util
from jax._src.util import (
    foreach,
    safe_map,
    safe_zip,
    split_list,
)
import jax.numpy as jnp
import numpy as np

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

BlockMapping = pallas_core.BlockMapping
GridMapping = pallas_core.GridMapping
CostEstimate = pallas_core.CostEstimate


def _logical_to_interpret_mode_dtype(dtype):
  """Converts logical dtypes into JAX dtypes for interpret mode.

  Logical types are dtypes that exist as part of the Pallas API but
  do not have an corresponding backing type in HLO (for example,
  a Semaphore dtype).

  This function maps a logical dtype to a valid HLO dtype that can be
  used to emulate the behavior of the logical dtype (such as mapping a
  Semaphore to int).
  """
  if (hasattr(dtype, "_rules") and
      hasattr(dtype._rules, "pallas_interpret_element_aval")):
    return dtype._rules.pallas_interpret_element_aval(dtype).dtype
  return dtype


def _logical_aval_to_interpret_mode_aval(aval):
  if isinstance(aval, state.AbstractRef):
    inner_aval = _logical_aval_to_interpret_mode_aval(aval.inner_aval)
    return aval.update(inner_aval=inner_aval)
  if isinstance(aval, jax_core.ShapedArray):
    inner_dtype = _logical_to_interpret_mode_dtype(aval.dtype)
    return jax_core.ShapedArray(aval.shape, inner_dtype, weak_type=aval.weak_type)
  return aval


def _dynamic_slice(
    start_idx, block_shape: tuple[int, ...], value, is_squeeze,
):
  start_idx = tuple(jnp.asarray(s, dtype=jnp.int32) for s in start_idx)
  output = lax.dynamic_slice(value, start_idx, slice_sizes=block_shape)
  squeeze_dims = tuple(np.arange(len(is_squeeze))[np.array(is_squeeze,
                                                           dtype=np.bool_)])
  return lax.squeeze(output, squeeze_dims)  # type: ignore[arg-type]


def _dynamic_update_slice(start_idx, block_shape, value, update, is_squeeze):
  start_idx = tuple(jnp.asarray(s, dtype=jnp.int32) for s in start_idx)
  broadcast_dims = tuple(i for i, b in enumerate(is_squeeze)
                         if not b)
  update = lax.broadcast_in_dim(update, block_shape, broadcast_dims)
  assert update.shape == block_shape
  return lax.dynamic_update_slice(value, update, start_idx)


# TODO(justinfu): Move this to a common utility file.
def _get_next_indices(grid, indices):
  next_indices = []
  carry = True
  for dim_size, index in reversed(list(zip(grid, indices))):
    i = jnp.where(carry, index + 1, index)
    carry = dim_size == i
    next_indices.append(jnp.where(carry, 0, i))
  return tuple(reversed(next_indices))


def _pad_to_block_dimension(value, block_shape: tuple[int, ...]):
  """Pads values so the shape evenly divides into block dimensions.

  For example, if values has a shape of (33, 2, 5) with a block_shape of
  (32, 2, 4), this function will pad the value of shape to (64, 2, 8).

  Args:
    value: Array to be padded.
    block_shape: Block shapes to use for padding.

  Returns:
    A padded array.
  """
  padded_shape = tuple(
      ((v - 1) // b + 1) * b for v, b in zip(value.shape, block_shape)
  )
  if padded_shape != value.shape:
    pad_width = tuple((0, a-b) for a, b in zip(padded_shape, value.shape))
    pad_value = primitives.uninitialized_value(shape=(), dtype=value.dtype)
    value = jnp.pad(value, pad_width, constant_values=pad_value)
  return value


def _initialize_output_vals(
    block_mappings_output: Iterable[BlockMapping],
    input_args, input_output_aliases) -> Sequence[jax.Array]:
  oi_map = {v: k for k, v in input_output_aliases}
  output_vals = []
  for i, bm in enumerate(block_mappings_output):
    if i in oi_map:
      output_vals.append(input_args[oi_map[i]])
    else:
      output_vals.append(primitives.uninitialized_value(
          bm.array_shape_dtype.shape,
          bm.array_shape_dtype.dtype))
  return output_vals


def kernel_to_hlo_jaxpr(jaxpr: jax_core.Jaxpr,
                        consts: Sequence[Any],
                        grid_mapping: GridMapping,
                        backend: str | None,
    ) -> tuple[jax_core.Jaxpr, Sequence[Any], Sequence[Any]]:
  """Converts a Pallas kernel jaxpr to a valid HLO jaxpr."""
  del backend
  with grid_mapping.trace_env():
    # TODO(justinfu): Evaluate backend-specific primitives in a new pass.
    phys_jaxpr, phys_consts = resolve_physical_types(jaxpr, consts)
    # For now, we assume that physical types are 1:1 with logical types
    # so that the indexing of scratch vars is unchanged.
    assert len(phys_jaxpr.invars) == len(jaxpr.invars)
    scratch_invars = phys_jaxpr.invars[grid_mapping.slice_scratch_ops]
    scratch_avals = [v.aval for v in scratch_invars]
    discharged_jaxpr, discharged_consts = state_discharge.discharge_state(
        phys_jaxpr, phys_consts)
  return discharged_jaxpr, discharged_consts, scratch_avals


def eval_jaxpr_recursive(
    jaxpr: jax_core.Jaxpr,
    consts,
    *args,
    recurse_hop_rule: Callable[[jax_core.Jaxpr, Sequence[Any]],
                               tuple[jax_core.Jaxpr, Sequence[Any]]],
    propagate_source_info=True) -> list[Any]:
  """Evaluates a Jaxpr with recursion into higher-order primitives.

  ``recurse_hop_rule`` is a Jaxpr interpreter (translates a Jaxpr to a new
  Jaxpr) that will be called on sub-jaxprs of higher-order primitives, such
  as the body of a loop or branches of a conditional.

  Args:
    jaxpr: The Jaxpr to evaluate.
    consts: Consts that ``jaxpr`` closes over.
    *args: Input arguments to the ``jaxpr``.
    recurse_hop_rule: A Jaxpr interpreter to call on sub-jaxprs of
      higher-order primitives.
    propagate_source_info: Whether to propagate source info.
  """
  def read(v: jax_core.Atom) -> Any:
    return v.val if isinstance(v, jax_core.Literal) else env[v]

  def write(v: jax_core.Var, val: Any) -> None:
    env[v] = val

  env: dict[jax_core.Var, Any] = {}
  foreach(write, jaxpr.constvars, consts)
  foreach(write, jaxpr.invars, args)
  lu = jax_core.last_used(jaxpr)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    name_stack = source_info_util.current_name_stack()
    name_stack += eqn.source_info.name_stack
    traceback = eqn.source_info.traceback if propagate_source_info else None
    with source_info_util.user_context(
        traceback, name_stack=name_stack), eqn.ctx.manager:
      if eqn.primitive in _eval_jaxpr_hop_rules:
        ans = _eval_jaxpr_hop_rules[eqn.primitive](
            recurse_hop_rule, *in_vals, **eqn.params)
      else:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        ans = eqn.primitive.bind(*subfuns, *in_vals, **bind_params)
    if eqn.primitive.multiple_results:
      foreach(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
    jax_core.clean_up_dead_vars(eqn, env, lu)
  return map(read, jaxpr.outvars)

# Higher-order primitive rules.
_eval_jaxpr_hop_rules = {}

def pad_jaxpr_constvars(jaxpr: jax_core.Jaxpr,
                        i: int,
                        all_const_avals: Sequence[Any]
                        ) -> jax_core.ClosedJaxpr:
  """Pads a Jaxpr with constvars from all branches.

  For primitives that have multiple Jaxprs (e.g. cond_p), we need
  to pad each Jaxpr with all consts from all branches so the
  signatures match, but only use the consts for this branch.
  """
  unused_const_vars = [tuple(map(jax_core.Var, const_avals))
                       for const_avals in all_const_avals]
  const_prefix = util.concatenate(unused_const_vars[:i])
  const_suffix = util.concatenate(unused_const_vars[i + 1:])
  constvars = [*const_prefix, *jaxpr.constvars, *const_suffix]
  jaxpr = jaxpr.replace(constvars=constvars)
  effects = pe.make_jaxpr_effects(jaxpr.constvars, jaxpr.invars,
                                  jaxpr.outvars, jaxpr.eqns)
  jaxpr = jaxpr.replace(effects=effects)
  return jax_core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())


def make_hop_rule(primitive, *keys):
  """Makes a rule for higher-order ops by recursively applying the jaxpr pass.

  Args:
    primitive: A JAX primitive.
    keys: The names of parameters which correspond to Jaxprs that need
      to be recursed over.

  Returns:
    A primitive rule for the edtype Jaxpr pass. This should be registered
    using `register_edtype_rule`.
  """
  def _resolve_jaxpr(interpreter,
                     value: jax_core.Jaxpr | jax_core.ClosedJaxpr,
                     mapped_idx=None):
    extra_args = ()
    if isinstance(value, jax_core.Jaxpr):
      if len(value.constvars) > 0:
        raise ValueError(f"Cannot physicalize a jaxpr with constvars: {value}")
      physical_jaxpr, physical_consts = interpreter(value, ())
      if physical_consts:
        if mapped_idx is not None:
          new_jaxpr = pad_jaxpr_constvars(physical_jaxpr,
                                          mapped_idx,
                                          physical_consts)
          extra_args = tuple(physical_consts)
        else:
          new_jaxpr = pe.convert_constvars_jaxpr(physical_jaxpr)
          extra_args = tuple(physical_consts)
      else:
        new_jaxpr = physical_jaxpr
    elif isinstance(value, jax_core.ClosedJaxpr):
      jaxpr, new_consts = interpreter(value.jaxpr, value.consts)
      new_jaxpr = jax_core.ClosedJaxpr(jaxpr, new_consts)
    else:
      raise ValueError(f"Parameter of type {type(value)} is not a Jaxpr.")
    return new_jaxpr, extra_args

  def rule(interpreter, *args, **params):
    new_params = {}
    for key in keys:
      value = params[key]
      if isinstance(value, jax_core.Jaxpr) or isinstance(
          value, jax_core.ClosedJaxpr):
        new_jaxpr, extra_args = _resolve_jaxpr(interpreter, value)
        new_params[key] = new_jaxpr
        args = extra_args + args
      elif isinstance(value, tuple) or isinstance(value, list):
        mapped_jaxprs, mapped_args = zip(*map(
          lambda x, i: _resolve_jaxpr(interpreter, x, mapped_idx=i), value, range(len(value))))
        all_new_args = tuple(new_arg for _args in mapped_args for new_arg in _args)
        new_params[key] = tuple(mapped_jaxprs)
        args = all_new_args + args
      else:
        raise ValueError(f"Parameter {key} is not a Jaxpr or sequence of Jaxprs: {value}")
    params.update(new_params)
    return primitive.bind(*args, **params)
  return rule

_eval_jaxpr_hop_rules[lax.scan_p] = make_hop_rule(lax.scan_p, 'jaxpr')
_eval_jaxpr_hop_rules[lax.while_p] = make_hop_rule(
    lax.while_p, 'body_jaxpr', 'cond_jaxpr')
_eval_jaxpr_hop_rules[lax.cond_p] = make_hop_rule(lax.cond_p, 'branches')
def _run_scoped_physicalize_rule(
    interpreter, *consts, jaxpr: jax_core.Jaxpr, collective_axes):
  if collective_axes:
    raise NotImplementedError(
        "run_scoped interpret rule does not support collective axes"
    )
  physical_jaxpr, physical_consts = interpreter(jaxpr, consts)
  return primitives.run_scoped_p.bind(
      *physical_consts, jaxpr=physical_jaxpr, collective_axes=collective_axes
  )
_eval_jaxpr_hop_rules[primitives.run_scoped_p] = _run_scoped_physicalize_rule


# TODO(justinfu): Replace this with a standardized physicalize pass.
def resolve_physical_types(jaxpr: jax_core.Jaxpr, consts: Sequence[Any]):
  kernel_avals = jax_core.ClosedJaxpr(jaxpr, consts).in_avals
  kernel_avals = tuple(map(_logical_aval_to_interpret_mode_aval,
                             kernel_avals))
  interp_fun = partial(
      eval_jaxpr_recursive, jaxpr, consts,
      recurse_hop_rule=resolve_physical_types)
  wrapped = lu.wrap_init(interp_fun, debug_info=jaxpr.debug_info)
  new_jaxpr, _, new_consts = pe.trace_to_jaxpr_dynamic(
      wrapped, kernel_avals)
  return new_jaxpr, new_consts


def pallas_call_hlo_interpret(
    *args,
    backend: str | None,
    jaxpr: jax_core.Jaxpr,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: GridMapping,
    mesh: pallas_core.Mesh | None,
    compiler_params: Any,
    cost_estimate: CostEstimate,
    out_avals: tuple[jax_core.AbstractValue, ...],
    metadata: frozen_dict.FrozenDict[str, str] | None,
):
  del mesh, compiler_params, cost_estimate, out_avals, metadata
  debug_info = jaxpr.debug_info
  # If we're in interpret mode, we *scan* over the grid and eval the
  # discharged jaxpr.
  dynamic_grid_args, args = split_list(
      args, [grid_mapping.num_dynamic_grid_bounds]
  )
  dynamic_grid_args_iter = iter(dynamic_grid_args)
  grid = tuple(
      a if a is not pallas_core.dynamic_grid_dim
      else next(dynamic_grid_args_iter)
      for a in grid_mapping.grid
  )
  assert next(dynamic_grid_args_iter, None) is None
  discharged_jaxpr, discharged_consts, scratch_avals = kernel_to_hlo_jaxpr(
      jaxpr, (), grid_mapping, backend=backend)
  if debug:
    print(f"\nJaxpr of the the kernel in pallas_call {debug_info.func_src_info}:")
    print(discharged_jaxpr)
  out = _initialize_output_vals(grid_mapping.block_mappings_output,
                                args, input_output_aliases)
  # TODO(b/370563936): Fix correctness issue w/ io aliasing
  scalars = args[grid_mapping.slice_index_ops]
  block_args = args[len(scalars):]
  # invars: [*scalar_prefetch, *consts, *inputs, *outputs, *scratch]
  # block_args now contains: *consts, *inputs, *outputs
  scratch_values = tuple(
      primitives.uninitialized_value(a.shape, a.dtype) for a in scratch_avals
  )

  carry = []
  for x, bm in zip(itertools.chain(block_args, out), grid_mapping.block_mappings):
    padding = [bd.padding if isinstance(bd, pallas_core.Element) else (0, 0)
               for bd in bm.block_shape]
    if padding is not None and any(p != (0, 0) for p in padding):
      if input_output_aliases:
        raise NotImplementedError("Padding with aliasing not supported.")
      pad_value = primitives.uninitialized_value(shape=(), dtype=x.dtype)
      x = lax.pad(x, pad_value, [(*p, 0) for p in padding])
    carry.append(x)

  block_shapes = [pallas_core._get_block_shape(bm.block_shape)
                  for bm in grid_mapping.block_mappings]
  is_squeeze_dim = [
      tuple(isinstance(bd, pallas_core.Squeezed) for bd in bm.block_shape)
      for bm in grid_mapping.block_mappings
  ]

  # Pad values to evenly divide into block dimensions. This matches the
  # behavior of the non-interpret mode. We pad with NaN, to make it easier
  # to catch OOB accesses.
  for carry_element in carry:
    aval = carry_element.aval
    if isinstance(aval, jax_core.DShapedArray):
      aval = jax_core.ShapedArray(aval.shape, aval.dtype)
      carry_element.aval = aval

  carry = map(_pad_to_block_dimension, carry, block_shapes)
  carry.extend(scratch_values)

  num_inout_blocks = len(block_args) + len(out)
  grid_start_indices = (jnp.int32(0),) * len(grid)
  if grid:
    num_iterations = reduce(jnp.multiply, grid)  # type: ignore[arg-type]
  else:
    # Base case is always one iteration when grid is ()
    num_iterations = 1

  # The scan carry: (i, loop_idx, *consts, *ins, *outs, *scratch)
  # i:int32 is the iteration index
  # loop_idx: tuple[int32] are the program ids for each grid axis
  def cond(carry):
    i, *_ = carry
    return i < num_iterations
  def body(carry):
    i, loop_idx, *carry_blocks = carry

    if grid_mapping.local_grid_env is not None:
      local_grid_env = grid_mapping.local_grid_env(loop_idx, grid)
    else:
      local_grid_env = tuple(
          pallas_core.GridAxis(idx, b)
          for dim, (idx, b) in enumerate(zip(loop_idx, grid))
          if dim not in grid_mapping.vmapped_dims
      )

    carry_consts_ins, scratch = split_list(carry_blocks, [num_inout_blocks])
    with pallas_core.grid_env(local_grid_env):
      for s in scalars:
        if isinstance(s.dtype, jax_core.bint):
          aval = jax_core.get_aval(s)
          s.aval = aval.update(dtype=jnp.int32)
      start_indices = [
          bm.compute_start_indices_interpret(loop_idx, *scalars)
          for bm in grid_mapping.block_mappings
      ]
    blocks = map(_dynamic_slice, start_indices, block_shapes,
                 carry_consts_ins, is_squeeze_dim)
    with pallas_core.grid_env(local_grid_env):
      assert len(discharged_jaxpr.invars) == len(scalars) + len(blocks) + len(
          scratch_values
      ), (
          len(discharged_jaxpr.invars),
          len(scalars),
          len(blocks),
          len(scratch_values),
      )

      blocks = jax_core.eval_jaxpr(
          discharged_jaxpr, discharged_consts, *scalars, *blocks, *scratch
      )

    _, out_inout, out_scratch = split_list(
        blocks, [grid_mapping.num_index_operands, num_inout_blocks])
    out_carry = map(_dynamic_update_slice, start_indices, block_shapes,
                    carry_consts_ins, out_inout, is_squeeze_dim)
    return (i + 1, _get_next_indices(grid, loop_idx),
            *out_carry, *out_scratch)

  (_, _, *carry) = lax.while_loop(
      cond, body, (jnp.int32(0), grid_start_indices, *carry)
  )

  out_out = carry[len(block_args):len(block_args) + len(out)]
  out_nopad = []
  for o, bm in zip(out_out, grid_mapping.block_mappings_output):
    padding = [bd.padding if isinstance(bd, pallas_core.Element) else (0, 0)
               for bd in bm.block_shape]
    if padding is not None and any(p != (0, 0) for p in padding):
      if input_output_aliases:
        raise NotImplementedError("Padding with aliasing not supported.")
      pad_low, pad_high = zip(*padding)
      limit_indices = [s - p for s, p in zip(o.shape, pad_high)]
      o = lax.slice(o, pad_low, limit_indices)
    if o.shape != bm.array_shape_dtype.shape:
      o = lax.slice(o, (0,) * o.ndim, bm.array_shape_dtype.shape)
    out_nopad.append(o)
  return out_nopad
