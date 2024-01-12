# Copyright 2023 The JAX Authors.
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

"""Module for lowering JAX primitives to Triton IR."""
from __future__ import annotations

from collections.abc import Sequence
import contextlib
import dataclasses
import functools
import operator
from typing import Any, Callable
import zlib

import jax
from jax import lax
from jax import tree_util
from jax._src import ad_checkpoint
from jax._src import ad_util
from jax._src import api_util
from jax._src import core as jax_core
from jax._src import custom_derivatives
from jax._src import linear_util as lu
from jax._src import pjit
from jax._src import state
from jax._src import util
from jax._src.lax.control_flow import for_loop
from jax._src.lib import gpu_triton as triton_kernel_call_lib
from jax._src.lib import hlo_helpers
from jax._src.lib import triton_compat
from jax._src.lib import triton_dialect
from jax._src.lib.mlir import ir
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives
from jax._src.pallas import utils as pallas_utils
from jax._src.pallas.pallas_call import pallas_call_p
from jax._src.state import AbstractRef
from jax._src.state import discharge
from jax._src.state import indexing
from jax._src.state import primitives as sp
from jax._src.util import merge_lists
from jax._src.util import partition_list
from jax._src.util import split_list
from jax._src.util import weakref_lru_cache
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp
from jax_triton import triton_lib
from jax_triton.triton_lib import compile_ttir_to_ptx_inplace
from jax_triton.triton_lib import get_triton_type
import numpy as np
from triton.compiler import code_generator as code_gen
import triton.compiler.backends.cuda as cb
from triton._C.libtriton import ir as tl_ir

# TODO(sharadmv): Enable type checking.
# mypy: ignore-errors

if triton_compat is None or triton_dialect is None:
  raise RuntimeError(
      "Cannot import the Triton bindings. You may need a newer version of"
      " jaxlib. Try installing a nightly wheel from:"
      " https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda_releases.html"
      " or https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda12_releases.html"
  )

tc = triton_compat
tt_dialect = triton_dialect

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip
partial = functools.partial
Grid = tuple[int, ...]
NDIndexer = indexing.NDIndexer
GridMapping = pallas_core.GridMapping
BlockMapping = pallas_core.BlockMapping


# # General lowering logic
@dataclasses.dataclass
class TritonModuleContext:
  name: str
  grid_mapping: GridMapping
  program_ids: Sequence[tc.tensor]


@dataclasses.dataclass
class BlockInfo:
  full_shape_dtype: jax.ShapeDtypeStruct
  start_indices: Sequence[Any]
  block_shape: tuple[int, ...]


@dataclasses.dataclass
class TritonLoweringRuleContext:
  context: TritonModuleContext
  avals_in: Any
  avals_out: Any
  block_infos: Sequence[BlockInfo | None]

  @property
  def builder(self) -> tc.builder:
    return tc.builder.current

  replace = dataclasses.replace


@dataclasses.dataclass
class TritonLoweringResult:
  """Keeps pybind11 objects alive."""

  module: ir.Module
  grid: tuple[int, ...]


@dataclasses.dataclass
class TritonCompilationResult:
  kernel_name: str
  ttir: str
  ptx: str
  shared_mem_bytes: int
  compute_capability: int
  lowering_result: TritonLoweringResult


class TritonLoweringException(Exception):
  pass


def _eval_index_map(
    ctx: TritonModuleContext, idx, block_mapping: BlockMapping | None
):
  if block_mapping is None:
    return None
  block_indices = tuple(
      lower_jaxpr_to_triton_ir(
          ctx, block_mapping.index_map_jaxpr.jaxpr, None, *idx
      )
  )
  return tuple(
      i if b is pallas_core.mapped else i * b
      for i, b in zip(block_indices, block_mapping.block_shape)
  )


def _convert_dtype(dtype: jnp.dtype) -> tc.dtype:
  if dtype == jnp.float32:
    return tc.float32
  elif dtype == jnp.float64:
    return tc.float64
  elif dtype == jnp.float16:
    return tc.float16
  elif dtype == jnp.bfloat16:
    return tc.bfloat16
  elif dtype == jnp.int32:
    return tc.int32
  elif dtype == jnp.int64:
    return tc.int64
  raise ValueError(f"Unhandled dtype: {dtype}")


triton_lowering_rules = {}


def _process_grid_to_3d_grid(grid_mapping: GridMapping):
  launch_grid = []
  launch_grid_to_pallas_grid = []

  # Preserve grid order provided to pallas_call
  for i, s in enumerate(grid_mapping.grid):
    if i not in grid_mapping.mapped_dims:
      launch_grid.append(s)
      launch_grid_to_pallas_grid.append(i)

  # For mapped dims, iterate from inner to outer. This follows the pallas_call
  # batching rule that prepends the vmapped dimension.
  for dim in reversed(grid_mapping.mapped_dims):
    s = grid_mapping.grid[dim]
    launch_grid.append(s)
    launch_grid_to_pallas_grid.append(dim)

  num_collapse = len(launch_grid[:-2])

  cuda_yz_limit = 2**16 - 1

  # Check Z and then Y launch dims to make sure they're within CUDA bounds
  if (num_collapse + 1 < len(launch_grid) and
      launch_grid[num_collapse + 1] > cuda_yz_limit):
    num_collapse += 2
  elif (num_collapse < len(launch_grid) and
        launch_grid[num_collapse] > cuda_yz_limit):
    num_collapse += 1

  collapse_dims = launch_grid[:num_collapse]
  prog_id_dims = launch_grid[num_collapse:]

  if len(collapse_dims) == 0:
    prog_ids = [None] * len(prog_id_dims)
    for i in range(len(prog_id_dims)):
      out_idx = launch_grid_to_pallas_grid[i]
      prog_ids[out_idx] = tc.program_id(i)

    return prog_id_dims, prog_ids
  else:
    new_grid = [np.prod(collapse_dims), *prog_id_dims]

  assert new_grid[0] < 2**31 - 1, \
          "Cannot fix pallas kernel launch grid within CUDA limits"

  out_indices = [None] * len(grid_mapping.grid)

  grid0 = tc.program_id(0)
  for i, s in enumerate(collapse_dims):
    out_idx = launch_grid_to_pallas_grid[i]
    grid0, out_indices[out_idx] = grid0 // s, grid0 % s

  for i in range(len(prog_id_dims)):
    out_idx = launch_grid_to_pallas_grid[num_collapse + i]
    out_indices[out_idx] = tc.program_id(i + 1)

  assert len(out_indices) == len(grid_mapping.grid)
  return new_grid, out_indices


def lower_jaxpr_to_triton_module(
    jaxpr: jax_core.Jaxpr,
    in_shapes,
    grid_mapping: GridMapping,
    name: str,
    cuda_options: cb.CUDAOptions,
) -> TritonLoweringResult:
  jaxpr, _ = pe.dce_jaxpr(jaxpr, [True] * len(jaxpr.outvars), instantiate=True)
  with contextlib.ExitStack() as stack:
    builder = tc.builder(cuda_options)
    stack.enter_context(builder)
    module: ir.Module = ir.Module.create()
    stack.enter_context(ir.InsertionPoint.at_block_begin(module.body))
    param_types = [
        code_gen.str_to_ty(get_triton_type(var.aval)) for var in jaxpr.invars
    ]
    assert len(jaxpr.outvars) == 0
    fn_type = ir.FunctionType.get(
        [t.to_ir(builder) for t in param_types],
        [],
    )
    fn = tt_dialect.FuncOp(
        name,
        ir.TypeAttr.get(fn_type),
        sym_visibility="public",
        res_attrs=ir.DictAttr.get(dict(noinline=ir.BoolAttr.get(False))),
    )
    fn.arg_attrs = ir.ArrayAttr.get(
        [ir.DictAttr.get({"tt.divisibility": builder.get_int32_attr(32)})]
        * len(param_types)
    )
    fn.body.blocks.append(*fn_type.inputs)
    [entry] = fn.body.blocks
    with ir.InsertionPoint(entry):
      new_grid, program_ids = _process_grid_to_3d_grid(grid_mapping)
      local_program_ids = [
          pid
          for i, pid in enumerate(program_ids)
          if i not in grid_mapping.mapped_dims
      ]
      ctx = TritonModuleContext(name, grid_mapping, local_program_ids)
      if grid_mapping.num_index_operands:
        raise NotImplementedError(
            "Scalar prefetch not supported in Triton lowering."
        )
      start_indices = map(
          partial(_eval_index_map, ctx, program_ids),
          grid_mapping.block_mappings,
      )
      block_infos = [
          BlockInfo(
              jax.ShapeDtypeStruct(shape_dtype.shape, shape_dtype.dtype),
              start_idx,
              block_mapping.block_shape,
          )
          if block_mapping is not None
          else None
          for shape_dtype, block_mapping, start_idx in zip(
              in_shapes, grid_mapping.block_mappings, start_indices
          )
      ]
      args = map(tc.tensor, entry.arguments, param_types)
      () = lower_jaxpr_to_triton_ir(ctx, jaxpr, block_infos, *args)
      tt_dialect.return_([])
    return TritonLoweringResult(module, new_grid)


def lower_jaxpr_to_triton_ir(
    ctx: TritonModuleContext,
    jaxpr: jax_core.Jaxpr,
    block_infos: Sequence[BlockInfo | None] | None,
    *args
) -> Sequence[Any]:
  env = {}
  block_info_env = {}

  def read_env(var: jax_core.Atom):
    if type(var) is jax_core.Literal:
      t = tc._to_tensor(np.array(var.val).tolist())
      dst_ty = code_gen.str_to_ty(get_triton_type(var.aval)).element_ty
      if t.type.scalar != dst_ty:
        # _to_tensor(np.array(var.val).tolist()) can be lossy e.g. np.float64
        # comes out of .tolist() as list[float], which then comes out of
        # _to_tensor as a block of f32.
        t = tc.semantic.cast(t, dst_ty)
      return t
    return env[var]

  def read_block_info_env(var: jax_core.Atom):
    if type(var) is jax_core.Literal:
      return None
    return block_info_env.get(var, None)

  def write_env(var: jax_core.Var, val):
    env[var] = val

  if block_infos is None:
    block_infos = [None] * len(jaxpr.invars)
  for invar, block_info in zip(jaxpr.invars, block_infos):
    block_info_env[invar] = block_info
  map(write_env, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    invals = map(read_env, eqn.invars)
    if eqn.primitive not in triton_lowering_rules:
      raise NotImplementedError(
          "Unimplemented primitive in Pallas GPU lowering: "
          f"{eqn.primitive.name}. "
          "Please file an issue on https://github.com/google/jax/issues.")
    rule = triton_lowering_rules[eqn.primitive]
    avals_in = [v.aval for v in eqn.invars]
    avals_out = [v.aval for v in eqn.outvars]
    eqn_block_infos = map(read_block_info_env, eqn.invars)
    rule_ctx = TritonLoweringRuleContext(
        ctx, avals_in, avals_out, eqn_block_infos
    )
    try:
      outvals = rule(rule_ctx, *invals, **eqn.params)
    except TritonLoweringException:
      raise  # We only add the extra info to the innermost exception.
    except Exception as e:
      raise TritonLoweringException(
          f"Exception while lowering eqn:\n  {eqn}\n"
          f"With context:\n  {rule_ctx}\n"
          f"With inval shapes={map(lambda t: t.shape, invals)}\n"
          f"With inval types={map(lambda t: t.type, invals)}\n"
          f"In jaxpr:\n{jaxpr}") from e
    if eqn.primitive.multiple_results:
      map(write_env, eqn.outvars, outvals)
    else:
      write_env(eqn.outvars[0], outvals)
  return map(read_env, jaxpr.outvars)


# # Primitive lowering rules
# ## Programming model primitives
def _program_id_lowering_rule(ctx: TritonLoweringRuleContext, *, axis):
  return ctx.context.program_ids[axis]


triton_lowering_rules[primitives.program_id_p] = _program_id_lowering_rule
# ## Atomic op primitives
_ATOMIC_OP_MAPPING = {
    primitives.AtomicOpType.XCHG: tc.atomic_xchg,
    primitives.AtomicOpType.ADD: tc.atomic_add,
    primitives.AtomicOpType.MAX: tc.atomic_max,
    primitives.AtomicOpType.MIN: tc.atomic_min,
    primitives.AtomicOpType.AND: tc.atomic_and,
    primitives.AtomicOpType.OR: tc.atomic_or,
    primitives.AtomicOpType.XOR: tc.atomic_xor,
}


def _atomic_lowering_rule(
    ctx: TritonLoweringRuleContext,
    *args_flat,
    args_tree,
    atomic_type: primitives.AtomicOpType,
):
  ptr, indexers, value, mask = args_tree.unflatten(args_flat)
  if len(indexers) != 1:
    raise NotImplementedError("Only single indexer is supported.")
  idx = indexers[0]
  ptr = _compute_pointers_from_indices(
      ptr, ctx.block_infos[0], idx, ctx.avals_in[0].shape
  )
  op = _ATOMIC_OP_MAPPING.get(atomic_type)
  if op is None:
    raise NotImplementedError(atomic_type)
  return op(ptr, value, mask=mask)


triton_lowering_rules[primitives.atomic_rmw_p] = _atomic_lowering_rule


def _associative_scan_lowering(
    body, ctx: TritonLoweringRuleContext, args, axes
):
  flat_args = tree_util.tree_leaves(args)
  (axis,) = axes
  dtype = ctx.avals_in[0].dtype
  in_avals = [
      jax_core.ShapedArray((), dtype=dtype),
      jax_core.ShapedArray((), dtype=dtype),
  ]
  in_tree = tree_util.tree_structure((args, args))
  flat_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(body), in_tree
  )
  combine_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      flat_fun, in_avals
  )
  out_tree = out_tree_thunk()
  del out_tree  # Not needed
  if consts:
    raise NotImplementedError("Associative scan with constants not supported.")
  element_types = [arg.type.scalar for arg in flat_args]
  scan_op = ctx.builder.create_scan([t.handle for t in flat_args], axis)
  param_types = element_types * 2
  ir_param_types = [ty.to_ir(ctx.builder) for ty in param_types]
  entry = scan_op.regions[0].blocks.append(*ir_param_types)
  combine_args = [
      tc.tensor(entry.arguments[i], ty) for i, ty in enumerate(param_types)
  ]
  with ir.InsertionPoint.at_block_begin(entry):
    results = lower_jaxpr_to_triton_ir(
        ctx.context, combine_jaxpr, None, *combine_args
    )
    ctx.builder.create_scan_ret(*(r.handle for r in results))
  scan_op.verify()
  def wrap_tensor(x, scalar_ty):
    if ctx.avals_out[0].shape:
      res_ty = tc.block_type(scalar_ty, ctx.avals_out[0].shape)
    else:
      # 0d-tensor -> scalar
      res_ty = scalar_ty
    return tc.tensor(x, res_ty)
  return map(wrap_tensor, scan_op.result, element_types)


def _cumsum_lowering_rule(
    ctx: TritonLoweringRuleContext,
    x,
    *, axis: int, reverse: bool
):
  if reverse:
    raise NotImplementedError("Reverse cumsum is not supported.")
  return _associative_scan_lowering(jnp.add, ctx, x, (axis,))[0]

triton_lowering_rules[lax.cumsum_p] = _cumsum_lowering_rule


_TRITON_FN_MAPPING = {
    # Unary ops.
    lax.neg_p: tc.semantic.minus,
    lax.abs_p: tc.abs,
    lax.ceil_p: tc.math.ceil,
    lax.floor_p: tc.math.floor,
    lax.exp_p: tc.exp,
    lax.exp2_p: tc.math.exp2,
    lax.expm1_p: tc.math.expm1,
    lax.log_p: tc.log,
    lax.log1p_p: tc.math.log1p,
    lax.sqrt_p: tc.sqrt,
    lax.cbrt_p: tc.math.cbrt,
    lax.rsqrt_p: tc.math.rsqrt,
    lax.sin_p: tc.sin,
    lax.cos_p: tc.cos,
    lax.tan_p: tc.math.tan,
    lax.asin_p: tc.math.asin,
    lax.acos_p: tc.math.acos,
    lax.atan_p: tc.math.atan,
    lax.atan2_p: tc.math.atan2,
    lax.sinh_p: tc.math.sinh,
    lax.cosh_p: tc.math.cosh,
    lax.tanh_p: tc.math.tanh,
    lax.asinh_p: tc.math.asinh,
    lax.acosh_p: tc.math.acosh,
    lax.atanh_p: tc.math.atanh,
    lax.not_p: tc.semantic.invert,
    lax.population_count_p: tc.math.popc,
    lax.clz_p: tc.math.clz,
    # Binary ops.
    lax.add_p: tc.semantic.add,
    lax.sub_p: tc.semantic.sub,
    lax.mul_p: tc.semantic.mul,
    lax.pow_p: tc.math.pow,
    lax.rem_p: tc.semantic.mod,
    lax.and_p: tc.semantic.and_,
    lax.or_p: tc.semantic.or_,
    lax.xor_p: tc.semantic.xor_,
    lax.eq_p: tc.semantic.equal,
    lax.ne_p: tc.semantic.not_equal,
    lax.gt_p: tc.semantic.greater_than,
    lax.ge_p: tc.semantic.greater_equal,
    lax.lt_p: tc.semantic.less_than,
    lax.le_p: tc.semantic.less_equal,
    lax.shift_left_p: tc.semantic.shl,
    lax.shift_right_arithmetic_p: tc.semantic.ashr,
    lax.shift_right_logical_p: tc.semantic.lshr,
    lax.nextafter_p: tc.math.nextafter,
    ad_util.add_any_p: tc.semantic.add,
    # Other ops.
    sp.broadcast_to_p: tc.broadcast_to,
    primitives.atomic_cas_p: tc.atomic_cas,
    primitives.max_contiguous_p: tc.max_contiguous,
    primitives.multiple_of_p: tc.multiple_of,
}


for primitive, fn in _TRITON_FN_MAPPING.items():
  def rule(ctx, *args, fn=fn, **kwargs):
    kwargs = tree_util.tree_map(tc.constexpr, kwargs)
    return fn(*args, **kwargs)

  triton_lowering_rules[primitive] = rule


def _integer_pow(a, *, y):
  if y == 2:
    return a * a
  if y == 3:
    return a * a * a
  if y == -2:
    return 1.0 / (a * a)
  return jax.lax.pow(a, y)


def lower_fun(
    fun: Callable[..., Any], *, multiple_results: bool
) -> Callable[..., Any]:
  fn = fun if multiple_results else lambda *args, **kw: (fun(*args, **kw),)

  def f_lowered(ctx: TritonLoweringRuleContext, *args, **params):
    wrapped_fun = lu.wrap_init(fn, params)
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, ctx.avals_in)
    jaxpr = jax_core.ClosedJaxpr(jaxpr, consts)
    out = _closed_call_lowering_rule(ctx, *args, call_jaxpr=jaxpr)
    return out if multiple_results else out[0]

  return f_lowered


_JAX_FN_MAPPING = {
    lax.clamp_p: lambda min, a, max: jnp.minimum(jnp.maximum(min, a), max),
    lax.integer_pow_p: _integer_pow,
    lax.logistic_p: lambda a: 1 / (1 + jnp.exp(-a)),
}

for primitive, fn in _JAX_FN_MAPPING.items():
  triton_lowering_rules[primitive] = lower_fun(fn, multiple_results=False)


def _min_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return tc.math.min(a, b)


triton_lowering_rules[lax.min_p] = _min_lowering_rule


def _max_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  return tc.math.max(a, b)


triton_lowering_rules[lax.max_p] = _max_lowering_rule


def _div_lowering_rule(ctx: TritonLoweringRuleContext, a, b):
  if a.dtype.is_floating() or b.dtype.is_floating():
    return a / b
  return a // b


triton_lowering_rules[lax.div_p] = _div_lowering_rule


def _iota_lowering_rule(
    ctx: TritonLoweringRuleContext, *, dtype, shape, dimension
):
  if dimension != 0:
    raise NotImplementedError()
  return tc.arange(0, shape[0])


triton_lowering_rules[lax.iota_p] = _iota_lowering_rule


def _convert_element_type_lowering_rule(
    ctx: TritonLoweringRuleContext, a, *, new_dtype, weak_type
):
  if new_dtype == ctx.avals_in[0].dtype:
    return a
  return tc.semantic.cast(a, _convert_dtype(new_dtype))


triton_lowering_rules[lax.convert_element_type_p] = (
    _convert_element_type_lowering_rule
)


def select_n_lowering_rule(ctx: TritonLoweringRuleContext, pred, a, b):
  return tc.semantic.where(pred, b, a)


triton_lowering_rules[lax.select_n_p] = select_n_lowering_rule


def _broadcast_in_dim_lowering_rule(
    ctx: TritonLoweringRuleContext, a, *, broadcast_dimensions, shape
):
  shape = map(tc.constexpr, shape)
  if not a.type.is_block():
    return tc.broadcast_to(a, shape)
  expand_dims = [i for i in range(len(shape)) if i not in broadcast_dimensions]
  for dim in expand_dims:
    a = tc.expand_dims(a, dim)
  return tc.broadcast_to(a, shape)


triton_lowering_rules[jax.lax.broadcast_in_dim_p] = (
    _broadcast_in_dim_lowering_rule
)


def _squeeze_lowering_rule(ctx: TritonLoweringRuleContext, a, *, dimensions):
  del dimensions
  return _reshape_lowering_rule(ctx, a, new_sizes=None, dimensions=None)


triton_lowering_rules[lax.squeeze_p] = _squeeze_lowering_rule


def _reshape_lowering_rule(
    ctx: TritonLoweringRuleContext, a, *, new_sizes, dimensions
):
  del new_sizes  # Unused.
  if dimensions is not None:
    return ValueError("`dimensions` is not supported.")

  dst_shape = map(tc.constexpr, ctx.avals_out[0].shape)
  if not a.type.is_block():
    assert all(dim_size.value == 1 for dim_size in dst_shape)
    return tc.broadcast_to(a, dst_shape)

  # Expand-dims or reduce-sum to handle singleton dims as `tl.reshape` is not
  # currently implemented.
  i = 0
  while a.shape != dst_shape:
    dim_size = a.shape[i].value if i < len(a.shape) else None
    dst_dim_size = dst_shape[i].value if i < len(dst_shape) else None
    if dim_size == dst_dim_size:
      i += 1
    elif dst_dim_size == 1:
      a = tc.expand_dims(a, axis=i)
      i += 1
    elif dim_size == 1:
      in_shape = tuple(d.value for d in a.shape)
      out_shape = tuple(d.value for di, d in enumerate(a.shape) if di != i)
      reduce_ctx = ctx.replace(
          avals_in=[ctx.avals_in[0].update(shape=in_shape)],
          avals_out=[ctx.avals_in[0].update(shape=out_shape)],
      )
      a = _reduce_lowering(jnp.add, reduce_ctx, a, axes=(i,))
    else:  # We expect this to fail.
      return tc.reshape(a, dst_shape)
  return a


triton_lowering_rules[jax.lax.reshape_p] = _reshape_lowering_rule


def _compute_pointers_from_indices(
    root_ptr: tc.tensor,
    block_info: BlockInfo | None,
    nd_indexer: NDIndexer,
    array_shape: tuple[int, ...],
) -> tc.tensor:
  if block_info is None:
    full_shape = array_shape
    num_mapped_dims = 0
    block_shape = array_shape
  else:
    full_shape = block_info.full_shape_dtype.shape
    num_mapped_dims = sum(
        b is pallas_core.mapped for b in block_info.block_shape
    )
    block_shape = block_info.block_shape
  strides = pallas_utils.strides_from_shape(full_shape)
  indexer_shape = nd_indexer.get_indexer_shape()
  int_indexer_shape = nd_indexer.int_indexer_shape
  indices = nd_indexer.indices
  other_shape = indexer_shape[len(int_indexer_shape) :]
  bcast_indices = []
  other_shape_idx = 0
  if block_info is None:
    start_index_offsets = [None] * len(indices)
  else:
    start_index_offsets = block_info.start_indices
  assert len(indices) + num_mapped_dims == len(full_shape)
  assert len(start_index_offsets) == len(full_shape)
  indexer_iter = iter(indices)
  for dim_stride, dim_block_size, start_offset in zip(
      strides, block_shape, start_index_offsets
  ):
    if dim_block_size is pallas_core.mapped:
      index = tc._to_tensor(0)
    else:
      index = next(indexer_iter)
    if isinstance(index, primitives.Slice):
      # Handle slices with static and dynamic indices and static sizes
      if isinstance(index.start, int):
        ptr_dim_offset = tc.arange(index.start, index.start + index.size)
      else:
        ptr_dim_offset = index.start + tc.arange(0, index.size)
      # We need to add broadcastable dimensions for the advanced int indexing
      # and for previous slices
      num_left_expand_dims = len(int_indexer_shape) + other_shape_idx
      num_right_expand_dims = len(other_shape) - other_shape_idx - 1
      other_shape_idx += 1
    elif isinstance(index, slice):
      if index != slice(None):
        raise NotImplementedError("Only `slice(None)` allowed.")
      ptr_dim_offset = tc.arange(0, dim_block_size)
      num_left_expand_dims = len(int_indexer_shape) + other_shape_idx
      num_right_expand_dims = len(other_shape) - other_shape_idx - 1
      other_shape_idx += 1
    else:
      # indexer is either a *scalar* or an array of size `int_indexer_shape`
      ptr_dim_offset = index
      num_left_expand_dims = 0
      num_right_expand_dims = len(other_shape)
      if not ptr_dim_offset.type.is_block():
        num_left_expand_dims = max(len(indexer_shape) - 1, 0)
      else:
        num_right_expand_dims = len(other_shape)
    if not ptr_dim_offset.type.is_block() and indexer_shape:
      ptr_dim_offset = tc.broadcast_to(
          ptr_dim_offset,
          [tc.constexpr(1)] * len(indexer_shape),
      )
    else:
      for _ in range(num_left_expand_dims):
        ptr_dim_offset = tc.expand_dims(ptr_dim_offset, 0)
      for _ in range(num_right_expand_dims):
        ndim = len(ptr_dim_offset.shape)
        ptr_dim_offset = tc.expand_dims(ptr_dim_offset, ndim)
    if start_offset is not None:
      ptr_dim_offset += start_offset
    stride_size = tc._to_tensor(int(dim_stride))
    bcast_indices.append(ptr_dim_offset * stride_size)
  block_shapes = [
      () if not index.type.is_block() else tuple(index.type.get_block_shapes())
      for index in bcast_indices
  ]
  bcast_indices = [
      tc.broadcast_to(index, map(tc.constexpr, indexer_shape))
      if indexer_shape != block_shape
      else index
      for index, block_shape in zip(bcast_indices, block_shapes)
  ]
  return sum(bcast_indices, root_ptr)


def _pack_indices(non_slice_idx, indexed_dims):
  non_slice_idx_iter = iter(non_slice_idx)
  return tuple(
      next(non_slice_idx_iter) if indexed else slice(None)
      for indexed in indexed_dims
  )


def _get_lowering_rule(
    ctx: TritonLoweringRuleContext, ptr, *idx, tree
):
  indexers = tree_util.tree_unflatten(tree, idx)
  if not isinstance(ptr.type, tc.pointer_type):
    assert len(indexers) == 0
    return ptr
  if len(indexers) > 1:
    raise NotImplementedError("No support for multiple indexers yet.")
  indexer = indexers[0]
  args_flat, args_tree = tree_util.tree_flatten((ptr, (indexer,), None, None))
  return _masked_load_lowering_rule(
      ctx,
      *args_flat,
      args_tree=args_tree,
      eviction_policy=None,
      cache_modifier=None,
      is_volatile=False,
  )


triton_lowering_rules[sp.get_p] = _get_lowering_rule


def _masked_load_lowering_rule(
    ctx: TritonLoweringRuleContext,
    *args_flat,
    args_tree,
    eviction_policy,
    cache_modifier,
    is_volatile,
):
  ptr, indexers, mask, other = args_tree.unflatten(args_flat)
  if len(indexers) > 1:
    raise NotImplementedError("No support for multiple indexers yet.")
  idx = indexers[0]
  if not isinstance(ptr.type, tc.pointer_type):
    assert len(ctx.avals_in) == 1
    return ptr
  ptr = _compute_pointers_from_indices(
      ptr, ctx.block_infos[0], idx, ctx.avals_in[0].shape
  )
  val = tc.load(
      ptr,
      mask=mask,
      other=other,
      cache_modifier=cache_modifier,
      volatile=is_volatile,
      eviction_policy=eviction_policy,
  )
  # `tl.load` of a `*int1` returns a tensor with type `int8`, so fix the type.
  return val.to(ptr.dtype.element_ty)


triton_lowering_rules[primitives.load_p] = _masked_load_lowering_rule


def _swap_lowering_rule(
    ctx: TritonLoweringRuleContext, ptr, value, *idx, tree
):
  indexers = tree_util.tree_unflatten(tree, idx)
  if not isinstance(ptr.type, tc.pointer_type):
    assert len(indexers) == 0
    return ptr
  if len(indexers) > 1:
    raise NotImplementedError("No support for multiple indexers yet.")
  indexer = indexers[0]
  args_flat, args_tree = tree_util.tree_flatten((ptr, (indexer,), value, None))
  return _masked_swap_lowering_rule(
      ctx, *args_flat, args_tree=args_tree, eviction_policy=None
  )


triton_lowering_rules[sp.swap_p] = _swap_lowering_rule


def _masked_swap_lowering_rule(
    ctx: TritonLoweringRuleContext, *args_flat, args_tree, eviction_policy
):
  ptr, indexers, value, mask = args_tree.unflatten(args_flat)
  if len(indexers) > 1:
    raise NotImplementedError("No support for multiple indexers yet.")
  idx = indexers[0]
  ptr_type = (
      ptr.type.element_ty.element_ty
      if ptr.type.is_block()
      else ptr.type.element_ty
  )
  value_type = value.type.element_ty if value.type.is_block() else value.type
  assert ptr_type == value_type, (ptr_type, value_type)
  ptr = _compute_pointers_from_indices(
      ptr, ctx.block_infos[0], idx, ctx.avals_in[0].shape
  )
  other = None if mask is None else value
  old_value = tc.load(ptr, mask=mask, other=other)
  tc.store(
      ptr,
      value,
      mask=mask,
      eviction_policy=eviction_policy,
  )
  return old_value


triton_lowering_rules[primitives.swap_p] = _masked_swap_lowering_rule


def _addupdate_lowering_rule(
    ctx: TritonLoweringRuleContext, ptr, value, *idx, tree
):
  indexers = tree_util.tree_unflatten(tree, idx)
  if not isinstance(ptr.type, tc.pointer_type):
    assert len(indexers) == 0
    return ptr
  if len(indexers) > 1:
    raise NotImplementedError("No support for multiple indexers yet.")
  indexer = indexers[0]
  ptr = _compute_pointers_from_indices(
      ptr, ctx.block_infos[0], indexer, ctx.avals_in[0].shape,
  )
  tc.atomic_add(ptr, value)
  return []


triton_lowering_rules[sp.addupdate_p] = _addupdate_lowering_rule


def _transpose_lowering(ctx: TritonLoweringRuleContext, a, *, permutation):
  if permutation != (1, 0):
    raise NotImplementedError(permutation)
  return tc.semantic.trans(a)


triton_lowering_rules[lax.transpose_p] = _transpose_lowering


_TF32_PRECISIONS = (lax.Precision.HIGH, lax.Precision.DEFAULT)


def _dot_general_lowering(
    ctx: TritonLoweringRuleContext,
    a,
    b,
    *,
    dimension_numbers,
    precision,
    preferred_element_type
):
  del preferred_element_type  # Unused.
  ((a_contract_dim,), (b_contract_dim,)), batch_dims = dimension_numbers
  assert batch_dims == ((), ())

  if a_contract_dim == 0:
    a = tc.semantic.trans(a)
  if b_contract_dim == 1:
    b = tc.semantic.trans(b)

  if precision is None:
    allow_tf32 = True
  else:
    prec_a, prec_b = precision
    allow_tf32 = prec_a in _TF32_PRECISIONS or prec_b in _TF32_PRECISIONS

  out_dtype = acc_dtype = _convert_dtype(ctx.avals_out[0].dtype)
  if acc_dtype not in (tc.int32, tc.float16):
    acc_dtype = tc.float32

  return tc.dot(
      a,
      b,
      allow_tf32=allow_tf32,
      out_dtype=acc_dtype,
  ).to(out_dtype)


triton_lowering_rules[lax.dot_general_p] = _dot_general_lowering


def _reduction_lowering(body, ctx: TritonLoweringRuleContext, args, axes):
  flat_args = tree_util.tree_leaves(args)
  (axis,) = axes
  mapped_avals = [
      jax_core.mapped_aval(aval.shape[axis], axis, aval)
      for aval in ctx.avals_in
  ]
  in_tree = tree_util.tree_structure((args, args))
  flat_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(body), in_tree
  )
  combine_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      flat_fun, [*mapped_avals, *mapped_avals]
  )
  out_tree = out_tree_thunk()
  del out_tree  # Not needed
  if consts:
    raise NotImplementedError("Reductions with constants not supported.")
  element_types = [arg.type.scalar for arg in flat_args]
  reduce_op = ctx.builder.create_reduce([t.handle for t in flat_args], axis)
  param_types = element_types * 2
  ir_param_types = [ty.to_ir(ctx.builder) for ty in param_types]
  entry = reduce_op.regions[0].blocks.append(*ir_param_types)
  combine_args = [
      tc.tensor(entry.arguments[i], ty) for i, ty in enumerate(param_types)
  ]
  with ir.InsertionPoint.at_block_begin(entry):
    results = lower_jaxpr_to_triton_ir(
        ctx.context, combine_jaxpr, None, *combine_args
    )
    ctx.builder.create_reduce_ret(*(r.handle for r in results))
  reduce_op.verify()

  def wrap_tensor(x, scalar_ty):
    if ctx.avals_out[0].shape:
      res_ty = tc.block_type(scalar_ty, ctx.avals_out[0].shape)
    else:
      # 0d-tensor -> scalar
      res_ty = scalar_ty
    return tc.tensor(x, res_ty)

  return map(wrap_tensor, reduce_op.result, element_types)


def _reduce_lowering(body, ctx: TritonLoweringRuleContext, a, *, axes):
  assert isinstance(axes, tuple)
  if not axes:
    return a
  while len(axes) > 1:
    axis = max(axes)
    dst_avals = tuple(v.update(shape=v.shape[:axis] + v.shape[axis + 1:])
                      for v in ctx.avals_in)
    a = _reduce_lowering(
        body, ctx.replace(avals_out=dst_avals), a, axes=(axis,))
    # Adding an intervening -(-reduce(.)) introduces a convert_layout between
    # reduces, which seems necessary for correctness.
    # TODO(bjp): Get rid of the double negation.
    #     https://github.com/openai/triton/issues/1776
    a = -(-a)
    ctx = ctx.replace(avals_in=dst_avals)
    axes = tuple(ax for ax in axes if ax != axis)
  return _reduction_lowering(body, ctx, a, axes=axes)[0]


triton_lowering_rules[lax.reduce_max_p] = functools.partial(
    _reduce_lowering, jnp.maximum
)
triton_lowering_rules[lax.reduce_min_p] = functools.partial(
    _reduce_lowering, jnp.minimum
)
triton_lowering_rules[lax.reduce_sum_p] = functools.partial(
    _reduce_lowering, jnp.add
)


def _argreduce_lowering(
    body, ctx: TritonLoweringRuleContext, a, *, axes, index_dtype
):
  if index_dtype != jnp.int32:
    raise ValueError("`index_type` must be i32.")
  if len(axes) != 1:
    raise ValueError("`pallas` reduce operations only support one reduce axis.")
  (axis,) = axes
  n = ctx.avals_in[0].shape[axis]
  index = tc.arange(0, n)
  if len(a.shape) > 1:
    # Broadcast index across the non-reduced axes
    expand_dims_index = [tc.constexpr(None)] * len(a.shape)
    expand_dims_index[axis] = slice(None)
    index = index[expand_dims_index]
    index = tc.broadcast_to(index, a.shape)
  ctx = ctx.replace(
      avals_in=[
          ctx.avals_in[0],
          ctx.avals_in[0].update(dtype=jnp.dtype("int32")),
      ]
  )
  _, indices = _reduction_lowering(body, ctx, (a, index), axes=axes)
  return indices


def _reduce_argmax_combine(left, right):
  value1, index1 = left
  value2, index2 = right
  gt = value1 > value2
  lt = value1 < value2
  index_min = jnp.minimum(index1, index2)
  index_ret = jnp.where(gt, index1, jnp.where(lt, index2, index_min))
  value_ret = jnp.maximum(value1, value2)
  return value_ret, index_ret


triton_lowering_rules[lax.argmax_p] = functools.partial(
    _argreduce_lowering, _reduce_argmax_combine
)


def _reduce_argmin_combine(left, right):
  value1, index1 = left
  value2, index2 = right
  gt = value1 > value2
  lt = value1 < value2
  index_min = jnp.minimum(index1, index2)
  index_ret = jnp.where(lt, index1, jnp.where(gt, index2, index_min))
  value_ret = jnp.minimum(value1, value2)
  return value_ret, index_ret


triton_lowering_rules[lax.argmin_p] = functools.partial(
    _argreduce_lowering, _reduce_argmin_combine
)


def _pjit_lowering_rule(ctx: TritonLoweringRuleContext, *args, jaxpr, **_):
  if jaxpr.consts:
    raise NotImplementedError
  return lower_jaxpr_to_triton_ir(
      ctx.context, jaxpr.jaxpr, ctx.block_infos, *args
  )


triton_lowering_rules[pjit.pjit_p] = _pjit_lowering_rule


def _closed_call_lowering_rule(
    ctx: TritonLoweringRuleContext, *args, call_jaxpr, **_
):
  jaxpr, consts = call_jaxpr.jaxpr, call_jaxpr.consts
  if consts:
    raise NotImplementedError
  return lower_jaxpr_to_triton_ir(ctx.context, jaxpr, ctx.block_infos, *args)


triton_lowering_rules[jax_core.closed_call_p] = _closed_call_lowering_rule
triton_lowering_rules[custom_derivatives.custom_jvp_call_p] = (
    _closed_call_lowering_rule
)


def _remat_lowering_rule(ctx: TritonLoweringRuleContext, *args, jaxpr, **_):
  return lower_jaxpr_to_triton_ir(ctx.context, jaxpr, ctx.block_infos, *args)


triton_lowering_rules[ad_checkpoint.remat_p] = _remat_lowering_rule
triton_lowering_rules[ad_util.stop_gradient_p] = lambda _, x: x


def _is_read_only(ref_effects) -> bool:
  if len(ref_effects) == 0:
    return True
  if len(ref_effects) > 1:
    # Means we must have a write or accum effect so not read-only
    return False
  (eff,) = ref_effects
  return isinstance(eff, state.ReadEffect)


def _for_lowering_rule(
    ctx: TritonLoweringRuleContext,
    *args,
    jaxpr,
    which_linear,
    nsteps,
    reverse,
    unroll
):
  del which_linear
  if reverse or unroll != 1:
    raise NotImplementedError
  lower_bound = ctx.builder.get_int32(0)
  upper_bound = ctx.builder.get_int32(nsteps)
  step = ctx.builder.get_int32(1)
  init_args = args
  # Partially discharge state from jaxpr for non-pointers
  should_discharge = [not isinstance(a, AbstractRef) for a in ctx.avals_in]
  discharged_jaxpr, () = discharge.discharge_state(
      jaxpr, (), should_discharge=[True, *should_discharge]
  )
  in_avals = [v.aval for v in jaxpr.invars]
  state_effects = state.get_ref_state_effects(in_avals, jaxpr.effects)[1:]
  # Read-only `Ref`s don't need to be passed in explicitly as loop arguments so
  # we can filter them out.
  read_only = map(_is_read_only, state_effects)
  is_loop_arg = map(
      operator.and_, map(operator.not_, read_only), should_discharge
  )
  ptrs, _ = partition_list(should_discharge, init_args)
  non_loop_args, loop_args = partition_list(is_loop_arg, init_args)
  for_op = ctx.builder.create_for_op(
      lower_bound, upper_bound, step, [arg.handle for arg in loop_args]
  )
  with ir.InsertionPoint(for_op.body):
    loop_index = tc.tensor(for_op.induction_variable, tc.int32)
    # Emit loop body
    for_body_args = [
        tc.tensor(for_op.body.arguments[i + 1], arg.type)
        for i, arg in enumerate(loop_args)
    ]
    loop_body_args = merge_lists(is_loop_arg, non_loop_args, for_body_args)
    out_discharged = lower_jaxpr_to_triton_ir(
        ctx.context,
        discharged_jaxpr,
        [None, *ctx.block_infos],
        loop_index,
        *loop_body_args,
    )
    all_out = merge_lists(should_discharge, ptrs, out_discharged)
    _, loop_out = partition_list(is_loop_arg, all_out)
    ctx.builder.create_yield_op([arg.handle for arg in loop_out])
  for_out = [tc.tensor(r, a.type) for r, a in zip(for_op.results_, loop_args)]
  return merge_lists(is_loop_arg, non_loop_args, for_out)


triton_lowering_rules[for_loop.for_p] = _for_lowering_rule

def _lower_jaxpr_to_for_loop(ctx: TritonLoweringRuleContext, jaxpr: jax_core.Jaxpr,
                             lower_bound, upper_bound, consts, *args,
                             has_loop_index: bool,
                             step: int = 1,
                             bound_type: tc.dtype = tc.int32):
  if step != 1:
    raise NotImplementedError
  if bound_type == tc.int64:
    step = ctx.builder.get_int64(step)
  else:
    step = ctx.builder.get_int32(step)

  for_op = ctx.builder.create_for_op(
      lower_bound, upper_bound, step, [arg.handle for arg in args]
  )
  with ir.InsertionPoint.at_block_begin(for_op.body):
    loop_index = tc.tensor(for_op.induction_variable, tc.int32)
    for_body_args = [
        tc.tensor(for_op.body.arguments[i + 1], arg.type) for i, arg in enumerate(args)
    ]
    if has_loop_index:
      jaxpr_args = [*consts, loop_index, *for_body_args]
    else:
      jaxpr_args = [*consts, *for_body_args]
    all_out = lower_jaxpr_to_triton_ir(
        ctx.context,
        jaxpr,
        ctx.block_infos,
        *jaxpr_args)
    ctx.builder.create_yield_op([arg.handle for arg in all_out])

  return [tc.tensor(r, a.type) for r, a in zip(for_op.results_, args)]


def _scan_lowering_rule(
    ctx: TritonLoweringRuleContext,
    *args,
    jaxpr,
    linear,
    length,
    reverse,
    unroll,
    num_consts,
    num_carry,
):
  # Only implements fori_loop-like scans
  num_extensive = len(args) - num_consts - num_carry
  if num_extensive: raise NotImplementedError
  if reverse: raise NotImplementedError
  if unroll != 1: raise NotImplementedError
  del linear, num_extensive, unroll, reverse

  jaxpr, jaxpr_consts = jaxpr.jaxpr, jaxpr.consts
  if jaxpr_consts: raise NotImplementedError
  del jaxpr_consts

  jaxpr, has_loop_index = (
      pallas_utils.pattern_match_scan_to_fori_loop(jaxpr, num_consts, num_carry)
      )
  consts, args = util.split_list(args, [num_consts])
  if has_loop_index:
    lb, *args = args
    lower_bound = lb.handle
    ub = lb + length
    upper_bound = ub.handle
    bound_type = ub.type
  else:
    lower_bound = ctx.builder.get_int32(0)
    upper_bound = ctx.builder.get_int32(length)
    bound_type = tc.int32
  for_out = _lower_jaxpr_to_for_loop(
      ctx, jaxpr, lower_bound, upper_bound, consts, *args,
      has_loop_index=has_loop_index, step=1, bound_type=bound_type)
  if has_loop_index:
    # Need to return the final loop index value if the outer scan expects
    # it as an output
    return [tc.tensor(upper_bound, bound_type), *for_out]
  return for_out

triton_lowering_rules[lax.scan_p] = _scan_lowering_rule

def _maybe_pattern_match_fori_loop(ctx: TritonLoweringRuleContext, *args,
                                   cond_nconsts, cond_jaxpr, body_nconsts, body_jaxpr
                                   ):
  if cond_nconsts:
    return None
  _, cond_invars = split_list(cond_jaxpr.jaxpr.invars, [cond_nconsts])
  cond_in_avals = [v.aval for v in cond_invars]
  if len(cond_in_avals) < 2:
    return None
  # Check that the first two carry values are scalar ints
  a1, a2 = cond_in_avals[:2]
  if a1.shape != () or a1.dtype not in (jnp.int32, jnp.int64):
    return None
  if a2.shape != () or a2.dtype not in (jnp.int32, jnp.int64):
    return None
  # Check that the only eqn in the cond checks the loop index condition
  v1, v2 = cond_invars[:2]
  outvar = cond_jaxpr.jaxpr.outvars[0]
  assert outvar.aval.dtype == jnp.bool_
  if len(cond_jaxpr.jaxpr.eqns) != 1:
    return None
  eqn = cond_jaxpr.jaxpr.eqns[0]
  if eqn.primitive != lax.lt_p:
    return None
  if eqn.outvars != [outvar]:
    return None
  if eqn.invars != [v1, v2]:
    return None
  # Check that the carry is updated in the body appropriately
  _, body_invars = split_list(body_jaxpr.jaxpr.invars, [body_nconsts])
  v1, v2 = body_invars[:2]
  vo1, vo2 = body_jaxpr.jaxpr.outvars[:2]
  # Upper bound should be constant
  if v2 is not vo2:
    return None
  # Check that we increment the loop index in the body
  for i, eqn in enumerate(body_jaxpr.jaxpr.eqns):
    if eqn.primitive is lax.add_p:
      if eqn.invars[0] is v1:
        if isinstance(eqn.invars[1], jax_core.Literal):
          if eqn.invars[1].val == 1:
            if eqn.outvars[0] == vo1:
              eqn_index = i
              break
  else:
    return None
  jaxpr = body_jaxpr.jaxpr
  new_invars = (*jaxpr.invars[:body_nconsts],
                jaxpr.invars[body_nconsts],
                *jaxpr.invars[body_nconsts + 2:])
  new_outvars = tuple(jaxpr.outvars[2:])
  jaxpr = jaxpr.replace(
      eqns=jaxpr.eqns[:eqn_index] + jaxpr.eqns[eqn_index + 1:],
      invars=new_invars,
      outvars=new_outvars)
  _, body_consts, carry = split_list(args, [cond_nconsts, body_nconsts])
  (lb, ub), args = carry[:2], carry[2:]
  const_block_infos, args_block_infos = split_list(ctx.block_infos,
                                                   [body_nconsts])
  ctx = ctx.replace(block_infos=[*const_block_infos, None,
                                 *args_block_infos[2:]])
  for_out = _lower_jaxpr_to_for_loop(ctx, jaxpr, lb.handle, ub.handle,
                                     body_consts, *args, has_loop_index=True,
                                     step=1, bound_type=lb.type)
  return [ub, ub, *for_out]

def _while_lowering_rule(
    ctx: TritonLoweringRuleContext,
    *args,
    cond_nconsts,
    cond_jaxpr,
    body_nconsts,
    body_jaxpr
):
  # First, try to pattern match to fori_loop and lower to scf.for if possible
  result = _maybe_pattern_match_fori_loop(ctx, *args, cond_nconsts=cond_nconsts,
                                          body_nconsts=body_nconsts, cond_jaxpr=cond_jaxpr,
                                          body_jaxpr=body_jaxpr)
  if result is not None:
    return result
  # Fall back to default while lowering
  num_args = len(args)
  cond_consts, body_consts, carry = util.split_list(
      args, [cond_nconsts, body_nconsts]
  )
  cond_const_block_infos, body_const_block_infos, carry_block_infos = (
      util.split_list(ctx.block_infos, [cond_nconsts, body_nconsts])
  )
  cond_const_types = [a.type.to_ir(ctx.builder) for a in cond_consts]
  body_const_types = [a.type.to_ir(ctx.builder) for a in body_consts]
  carry_types = [a.type.to_ir(ctx.builder) for a in carry]
  all_types = [*cond_const_types, *body_const_types, *carry_types]
  while_op = ctx.builder.create_while_op(
      all_types,
      [arg.handle for arg in args],
  )

  before_block = while_op.before.blocks.append(*all_types)
  cond_consts_, _, carry_ = util.split_list(
      before_block.arguments,
      [cond_nconsts, body_nconsts],
  )
  cond_args = [
      tc.tensor(a, b.type)
      for a, b in zip([*cond_consts_, *carry_], [*cond_consts, *carry])
  ]
  with ir.InsertionPoint.at_block_begin(before_block):
    [cond] = lower_jaxpr_to_triton_ir(
        ctx.context,
        cond_jaxpr.jaxpr,
        [*cond_const_block_infos, *carry_block_infos],
        *cond_args,
    )
    ctx.builder.create_condition_op(cond.handle, before_block.arguments)

  after_block = while_op.after.blocks.append(*all_types)
  cond_consts_, body_consts_, carry_ = util.split_list(
      after_block.arguments,
      [cond_nconsts, body_nconsts],
  )
  all_args = [
      tc.tensor(a, b.type)
      for a, b in zip(
          [*cond_consts_, *body_consts_, *carry_],
          [*cond_consts, *body_consts, *carry],
      )
  ]
  cond_const_args, body_const_args, carry_args = util.split_list(
      all_args, [cond_nconsts, body_nconsts]
  )
  with ir.InsertionPoint.at_block_begin(after_block):
    loop_out = lower_jaxpr_to_triton_ir(
        ctx.context,
        body_jaxpr.jaxpr,
        [*body_const_block_infos, *carry_block_infos],
        *body_const_args,
        *carry_args
    )
    cond_consts_handles = [a.handle for a in cond_const_args]
    body_consts_handles = [a.handle for a in body_const_args]
    loop_out_handles = [a.handle for a in loop_out]
    all_handles = [*cond_consts_handles, *body_consts_handles, *loop_out_handles]
    if all_handles:
      ctx.builder.create_yield_op(all_handles)

  all_out =  [
      tc.tensor(r, a.type) for r, a in zip(while_op.results_, args)
  ]
  return all_out[cond_nconsts + body_nconsts :]


triton_lowering_rules[lax.while_p] = _while_lowering_rule


def _cond_lowering_rule(
    ctx: TritonLoweringRuleContext,
    index,
    *args,  # *consts, *ops
    branches,  # tuple(jaxprs)
    linear,
):
  block_infos = ctx.block_infos

  def to_type(out_aval):
    elt_type = code_gen.str_to_ty(get_triton_type(out_aval)).element_ty
    if not out_aval.shape:
      # Scalar types get special handling.
      return elt_type
    return tc.block_type(elt_type, out_aval.shape)

  out_types = [to_type(out) for out in ctx.avals_out]
  out_ir_types = [t.to_ir(ctx.builder) for t in out_types]

  use_branch0 = index == 0
  # TODO(bjp): Switch to scf.index_switch once exposed in triton.cc
  if_op = ctx.builder.create_if_op(out_ir_types, use_branch0.handle, with_else=True)
  with ir.InsertionPoint.at_block_begin(if_op.then_block):
    outs0 = lower_jaxpr_to_triton_ir(
        ctx.context,
        branches[0].jaxpr,
        block_infos[1:],
        *args)
    ctx.builder.create_yield_op([out.handle for out in outs0])
  with ir.InsertionPoint.at_block_begin(if_op.else_block):
    # TODO(bjp): Instead of linear nest of 'if's, partition into halves.
    if len(branches) > 2:
      outs1 = _cond_lowering_rule(
          ctx, index - 1, *args, branches=branches[1:], linear=linear
      )
    else:
      outs1 = lower_jaxpr_to_triton_ir(
          ctx.context,
          branches[1].jaxpr,
          block_infos[1:],
          *args)
    ctx.builder.create_yield_op([out.handle for out in outs1])

  return map(tc.tensor, if_op.results_, out_types)


triton_lowering_rules[lax.cond_p] = _cond_lowering_rule


@weakref_lru_cache
def compile_jaxpr(
    jaxpr: jax_core.Jaxpr,
    in_shapes,
    grid_mapping: GridMapping,
    name: str,
    num_warps: int,
    num_stages: int,
    debug: bool,
) -> TritonCompilationResult:
  # TODO(sharadmv): handle multiple devices, right now we assume device 0
  # which is fine when we have multiple of the same GPU but this won't work in
  # general.
  device = 0
  arch = triton_kernel_call_lib.get_compute_capability(device)
  target = ("cuda", arch)
  cuda_backend = cb.CUDABackend(target)
  cuda_options = cuda_backend.parse_options(
      dict(
          num_warps=num_warps,
          num_stages=num_stages,
          debug=debug,
      )
  )
  context = tl_ir.context()
  tl_ir.load_dialects(context)
  cuda_backend.load_dialects(context)

  lowering_result = lower_jaxpr_to_triton_module(
      jaxpr, in_shapes, grid_mapping, name, cuda_options
  )

  ttir = str(lowering_result.module)
  ptx, name, shared_mem_bytes, compute_capability = compile_ttir_to_ptx_inplace(
      lowering_result.module,
      context,
      cuda_backend,
      cuda_options,
      device=device,
  )
  return TritonCompilationResult(
      name, ttir, ptx, shared_mem_bytes, compute_capability, lowering_result
  )


def pallas_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    jaxpr: jax_core.Jaxpr,
    name: str,
    in_shapes: tuple[jax.ShapeDtypeStruct, ...],
    out_shapes: tuple[jax.ShapeDtypeStruct, ...],
    which_linear: tuple[bool, ...],
    interpret: bool,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: GridMapping,
    triton_params: dict[str, Any] | None = None,
    **compiler_params: Any,
):
  if interpret:
    return mlir.lower_fun(pallas_call_p.impl, multiple_results=True)(
        ctx,
        *in_nodes,
        jaxpr=jaxpr,
        name=name,
        out_shapes=out_shapes,
        in_shapes=in_shapes,
        which_linear=which_linear,
        interpret=interpret,
        debug=debug,
        input_output_aliases=input_output_aliases,
        grid_mapping=grid_mapping,
        **compiler_params
    )
  num_warps = compiler_params.get("num_warps", 4)
  if len(ctx.module_context.platforms) > 1:
    raise NotImplementedError("multi-platform lowering for Pallas kernels")
  if ctx.module_context.platforms[0] == 'rocm':
    num_stages = compiler_params.get("num_stages", 1)
  else:
    num_stages = compiler_params.get("num_stages", 3)

  if debug:
    print(jaxpr)
    print(grid_mapping)
  compilation_result = compile_jaxpr(
      jaxpr,
      (*in_shapes, *out_shapes),
      grid_mapping,
      name,
      num_warps,
      num_stages,
      debug=debug,
  )
  #Triton returns a tuple for ROCm. We just want file path to be passed
  if ctx.module_context.platforms[0] == 'rocm':
      compilation_result.ptx = compilation_result.ptx[1]

  if debug:
    compilation_result.lowering_result.module.dump()

  kernel = triton_kernel_call_lib.TritonKernel(
      compilation_result.kernel_name,
      num_warps,
      compilation_result.shared_mem_bytes,
      compilation_result.ptx,
      compilation_result.ttir,
      compilation_result.compute_capability,
  )

  grid = triton_lib.normalize_grid(
      compilation_result.lowering_result.grid, metaparams={}
  )

  kernel_params = []
  for _ in range(len(in_shapes) + len(out_shapes)):
    kernel_params.append(
        triton_kernel_call_lib.create_array_parameter(
            0,  # bytes to zero  # TODO(cjfj): Expose through user API.
            16,  # divisible by 16
        )
    )

  kernel_call = triton_kernel_call_lib.TritonKernelCall(
      kernel, grid[0], grid[1], grid[2], kernel_params
  )

  out_types = [
      ir.RankedTensorType.get(shape.shape, mlir.dtype_to_ir_type(shape.dtype))
      for shape in out_shapes
  ]

  if triton_params is None:
    triton_params = {}
  serialized_metadata = triton_params.get("serialized_metadata", b"")
  kernel_call_proto = kernel_call.to_proto(name, serialized_metadata)
  return hlo_helpers.custom_call(
      call_target_name="triton_kernel_call",
      result_types=out_types,
      operands=in_nodes,
      backend_config=zlib.compress(kernel_call_proto),
      operand_layouts=triton_lib.avals_to_layouts(ctx.avals_in),
      result_layouts=triton_lib.avals_to_layouts(ctx.avals_out),
      operand_output_aliases=dict(input_output_aliases),
  ).results


mlir.register_lowering(pallas_call_p, pallas_call_lowering, platform="gpu")
