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

"""Python interface for programming the XLA:GPU Thunk Executor directly."""

from collections.abc import Callable, Mapping, Sequence
import functools
import math
import struct
from typing import Any

import jax
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import flattree as ft
from jax._src import linear_util as lu
from jax._src import tree_util
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import pxla
from jax._src.lib import jaxlib_extension_version
from jax._src.lib import thunky_dialect
from jax._src.lib.mlir import ir
from jax._src.lib.mlir import passmanager
from jax._src.lib.mlir.dialects import builtin as builtin_dialect  # noqa: F401
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state.primitives import get_p
from jax._src.state.primitives import swap_p
from jax._src.state.types import (
    AbstractRef,
    AccumEffect,
    BitcastTransform,
    ReadEffect,
    ReshapeTransform,
    TransformedRef,
    WriteEffect,
)
import numpy as np

_thunky: Any = thunky_dialect


def _check_thunky_available():
  if _thunky is None:
    raise RuntimeError(
        "jax.experimental.thunky requires jaxlib_extension_version >= 503, got"
        f" {jaxlib_extension_version}."
    )


def make_ir_context() -> ir.Context:
  """Creates an MLIR Context with the `thunky` dialect registered."""
  _check_thunky_available()
  ctx = mlir.make_ir_context()
  _thunky.register_dialect(ctx)
  return ctx


def jax_executable_to_mlir(
    compiled: jax.stages.Compiled, ctx: ir.Context | None = None
) -> ir.Module:
  """Converts a compiled JAX executable directly into an `ir.Module` in the `thunky` dialect."""
  _check_thunky_available()
  runtime_exe = compiled.runtime_executable()
  if runtime_exe is None:
    raise ValueError("Compiled executable has no runtime_executable()")
  serialized = runtime_exe.serialize()
  return _thunky.jax_executable_to_mlir(serialized, ctx)


def splice_thunk_module(
    target_block: ir.Block,
    callee_module: ir.Module,
    arg_mapping: Sequence[ir.Value],
) -> None:
  """Merges `callee_module` into `target_block`'s module and emits a call to it.

  If the callee's entry function takes additional trailing buffer arguments
  beyond `arg_mapping` (such as scratch or constant buffers allocated by XLA),
  corresponding buffer parameters are appended to the enclosing `func.func` and
  forwarded to the call.
  """
  callee_func = None
  for op in callee_module.body.operations:
    if (
        op.operation.name == "func.func"
        and ir.StringAttr(op.operation.attributes["sym_name"]).value == "main"
    ):
      callee_func = op.operation
      break
  if callee_func is None:
    raise ValueError("callee_module has no func.func @main")

  callee_block = callee_func.regions[0].blocks[0]
  if len(callee_block.arguments) < len(arg_mapping):
    raise ValueError(
        f"Expected at least {len(arg_mapping)} arguments in callee, got"
        f" {len(callee_block.arguments)}"
    )

  curr_op = target_block.owner.operation
  while curr_op is not None and curr_op.name != "func.func":
    parent = curr_op.parent
    curr_op = parent.operation if parent is not None else None
  if curr_op is None or curr_op.parent is None:
    raise ValueError("target_block is not inside a module func.func operation")

  call_args = list(arg_mapping)
  if len(callee_block.arguments) > len(arg_mapping):
    main_entry_block = curr_op.regions[0].blocks[0]
    for extra_callee_arg in callee_block.arguments[len(arg_mapping) :]:
      new_outer_arg = main_entry_block.add_argument(
          extra_callee_arg.type, extra_callee_arg.location
      )
      call_args.append(new_outer_arg)
    curr_op.attributes["function_type"] = ir.TypeAttr.get(
        ir.FunctionType.get(
            [arg.type for arg in main_entry_block.arguments], []
        )
    )

  dst_module: Any = curr_op.parent.opview
  sym_name = mlir.merge_mlir_modules(dst_module, "callee", callee_module)
  with ir.InsertionPoint(target_block):
    func_dialect.CallOp([], ir.FlatSymbolRefAttr.get(sym_name), call_args)


# ==============================================================================
# JAX Jaxpr Embedding & Jaxpr-to-MLIR Translation
# ==============================================================================


def _is_buffer_aval(aval: Any) -> bool:
  return isinstance(aval, AbstractRef) and isinstance(
      aval.inner_aval, core.ShapedArray
  )


def _shape_dtype_nbytes(shape: Sequence[int], dtype: Any) -> int:
  total_bits = int(math.prod(shape)) * dtypes.itemsize_bits(dtype)
  if total_bits % 8 != 0:
    raise ValueError(
        f"Buffer shape {tuple(shape)} with dtype {dtype} has {total_bits} "
        "bits, which is not a whole number of bytes"
    )
  return total_bits // 8


def _buffer_nbytes(x: Any) -> int:
  return _shape_dtype_nbytes(x.shape, x.dtype)


class ThunkEffect(core.effects.Effect):
  """Ordered effect preserving execution order of side-effecting thunk ops."""

  def __str__(self) -> str:
    return "ThunkEffect"


thunk_effect = ThunkEffect()
core.effects.ordered_effects.add_type(ThunkEffect)
core.effects.lowerable_effects.add_type(ThunkEffect)
core.effects.control_flow_allowed_effects.add_type(ThunkEffect)
core.effects.remat_allowed_effects.add_type(ThunkEffect)


def _extract_concrete_int(x: Any) -> int:
  if isinstance(x, core.Literal):
    return int(x.val)
  if isinstance(x, (int, np.integer)):
    return int(x)
  c = core.to_concrete_value(x)
  if c is not None:
    return int(c)
  raise ValueError(f"Expected static integer slice index, got {x}")


SliceSpec = tuple[int, tuple[int, ...], np.dtype]


def _spec_nbytes(spec: SliceSpec) -> int:
  return _shape_dtype_nbytes(spec[1], spec[2])


def _transforms_to_slice_params(
    base_aval: Any, transforms: Sequence[Any]
) -> SliceSpec:
  """Computes (offset_bytes, new_shape, new_dtype) for a sequence of Ref transforms."""
  curr_shape = tuple(int(d) for d in base_aval.shape)
  curr_dtype = np.dtype(base_aval.dtype)
  offset_bytes = 0

  for t in transforms:
    if isinstance(t, indexing.NDIndexer):
      idx_tuple = t.indices
      if not idx_tuple:
        continue
      first = idx_tuple[0]
      for dim_offset, rem in enumerate(idx_tuple[1:], start=1):
        if isinstance(rem, indexing.Slice):
          rem_start = _extract_concrete_int(rem.start)
          rem_size = _extract_concrete_int(rem.size)
          rem_stride = _extract_concrete_int(rem.stride)
          if (
              rem_start != 0
              or rem_stride != 1
              or rem_size != curr_shape[dim_offset]
          ):
            raise ValueError(
                "Only full slices (:) are supported on trailing dimensions for"
                " contiguous buffer slicing"
            )
        elif rem is not None:
          raise ValueError(
              "Non-contiguous slicing on trailing dimensions is not supported;"
              " only axis-0 slicing is contiguous"
          )

      row_stride_bits = int(math.prod(curr_shape[1:])) * dtypes.itemsize_bits(
          curr_dtype
      )
      if isinstance(first, indexing.Slice):
        start = _extract_concrete_int(first.start)
        size = _extract_concrete_int(first.size)
        stride = _extract_concrete_int(first.stride)
        if stride != 1:
          raise ValueError(f"Ref slice stride must be 1, got {stride}")
        if start < 0:
          start += curr_shape[0]
        if start < 0 or start + size > curr_shape[0]:
          raise ValueError(
              f"Ref slice [{start}:{start + size}] out of bounds for axis 0 of"
              f" size {curr_shape[0]}"
          )
        offset_bits = start * row_stride_bits
        if offset_bits % 8 != 0:
          raise ValueError(
              f"Ref slice offset {start} with dtype {curr_dtype} is"
              f" {offset_bits} bits, which is not byte-aligned"
          )
        offset_bytes += offset_bits // 8
        curr_shape = (size, *curr_shape[1:])
        _shape_dtype_nbytes(curr_shape, curr_dtype)
      else:
        i = _extract_concrete_int(first)
        if i < 0:
          i += curr_shape[0]
        if i < 0 or i >= curr_shape[0]:
          raise IndexError(
              f"Ref index {i} out of bounds for axis 0 of size {curr_shape[0]}"
          )
        offset_bits = i * row_stride_bits
        if offset_bits % 8 != 0:
          raise ValueError(
              f"Ref index {i} with dtype {curr_dtype} is {offset_bits} bits,"
              " which is not byte-aligned"
          )
        offset_bytes += offset_bits // 8
        curr_shape = curr_shape[1:]
        _shape_dtype_nbytes(curr_shape, curr_dtype)
    elif isinstance(t, BitcastTransform):
      curr_dtype = np.dtype(t.dtype)
      transformed_aval = t.transform_type(
          core.ShapedArray(curr_shape, curr_dtype)
      )
      curr_shape = tuple(int(d) for d in transformed_aval.shape)
    elif isinstance(t, ReshapeTransform):
      curr_shape = tuple(int(d) for d in t.shape)
    else:
      raise TypeError(f"Unsupported Ref transform in thunky: {t}")

  return offset_bytes, curr_shape, curr_dtype


def _unwrap_ref_and_spec(ref_or_view: Any) -> tuple[Any, SliceSpec]:
  if isinstance(ref_or_view, TransformedRef):
    base_ref = ref_or_view.ref
    base_aval = core.typeof(base_ref)
    spec = _transforms_to_slice_params(base_aval, ref_or_view.transforms)
    return base_ref, spec
  aval = core.typeof(ref_or_view)
  if not isinstance(aval, AbstractRef):
    raise TypeError(f"Expected AbstractRef buffer, got {aval}")
  return ref_or_view, (0, aval.shape, aval.dtype)


# --- memzero ---

memzero_p = core.Primitive("thunky.memzero")
memzero_p.multiple_results = True


def _memzero_abstract_eval(
    buf: AbstractRef, *, slice_spec: SliceSpec
) -> tuple[list[Any], set[Any]]:
  del slice_spec
  if not _is_buffer_aval(buf):
    raise TypeError(f"memzero expects an AbstractRef buffer, got {buf}")
  return [], {WriteEffect(0)}


memzero_p.def_effectful_abstract_eval(_memzero_abstract_eval)


def memzero(buf: Any) -> None:
  """Zeroes out a device buffer in a thunky Jaxpr."""
  base_ref, spec = _unwrap_ref_and_spec(buf)
  memzero_p.bind(base_ref, slice_spec=spec)


# --- memset ---

memset_p = core.Primitive("thunky.memset")
memset_p.multiple_results = True


def _memset_abstract_eval(
    buf: AbstractRef, *, value: int, slice_spec: SliceSpec
) -> tuple[list[Any], set[Any]]:
  del value, slice_spec
  if not _is_buffer_aval(buf):
    raise TypeError(f"memset expects an AbstractRef buffer, got {buf}")
  return [], {WriteEffect(0)}


memset_p.def_effectful_abstract_eval(_memset_abstract_eval)


def memset(buf: Any, value: np.generic) -> None:
  """Fills a device buffer with a 32-bit numpy scalar value in a thunky Jaxpr."""
  if not isinstance(value, np.generic):
    raise TypeError(
        "thunky.memset expects a numpy scalar (np.generic) value, got "
        f"{type(value).__name__}"
    )
  if dtypes.itemsize_bits(value.dtype) != 32:
    raise NotImplementedError(
        "thunky.memset currently only supports 32-bit scalar values, got"
        f" {value.dtype}"
    )
  base_ref, spec = _unwrap_ref_and_spec(buf)
  dtype = spec[2]
  if dtype is not None and dtypes.itemsize_bits(dtype) != 32:
    raise NotImplementedError(
        "thunky.memset currently only supports 32-bit element types, got"
        f" {dtype}"
    )
  bits = int(value.view(np.uint32))
  memset_p.bind(base_ref, value=bits, slice_spec=spec)


# --- copy ---

copy_p = core.Primitive("thunky.copy")
copy_p.multiple_results = True


def _copy_abstract_eval(
    src: AbstractRef,
    dst: AbstractRef,
    *,
    src_spec: SliceSpec,
    dst_spec: SliceSpec,
) -> tuple[list[Any], set[Any]]:
  if not _is_buffer_aval(src) or not _is_buffer_aval(dst):
    raise TypeError(f"copy expects AbstractRef buffers, got {src}, {dst}")
  if _spec_nbytes(src_spec) != _spec_nbytes(dst_spec):
    raise ValueError(
        f"copy buffer size mismatch: {_spec_nbytes(src_spec)} vs"
        f" {_spec_nbytes(dst_spec)}"
    )
  return [], {ReadEffect(0), WriteEffect(1)}


copy_p.def_effectful_abstract_eval(_copy_abstract_eval)


def copy(src: Any, dst: Any) -> None:
  """Copies from src device buffer to dst device buffer in a thunky Jaxpr."""
  src_ref, src_spec = _unwrap_ref_and_spec(src)
  dst_ref, dst_spec = _unwrap_ref_and_spec(dst)
  copy_p.bind(src_ref, dst_ref, src_spec=src_spec, dst_spec=dst_spec)


# --- ptx_kernel ---

ptx_kernel_p = core.Primitive("thunky.ptx_kernel")
ptx_kernel_p.multiple_results = True


def _ptx_kernel_abstract_eval(
    *buffers: AbstractRef,
    slice_specs: tuple[SliceSpec, ...],
    written: tuple[bool, ...],
    kernel_name: str,
    ptx: str,
    grid_dim: tuple[int, int, int],
    block_dim: tuple[int, int, int],
    shmem_bytes: int,
) -> tuple[list[Any], set[Any]]:
  del slice_specs, kernel_name, ptx, grid_dim, block_dim, shmem_bytes
  for b in buffers:
    if not _is_buffer_aval(b):
      raise TypeError(f"ptx_kernel expects AbstractRef arguments, got {b}")
  effs = {WriteEffect(i) if w else ReadEffect(i) for i, w in enumerate(written)}
  return [], effs | {thunk_effect}


ptx_kernel_p.def_effectful_abstract_eval(_ptx_kernel_abstract_eval)


def ptx_kernel(
    *buffers: Any,
    written: Sequence[bool],
    kernel_name: str,
    ptx: str,
    grid_dim: tuple[int, int, int] = (1, 1, 1),
    block_dim: tuple[int, int, int] = (1, 1, 1),
    shmem_bytes: int = 0,
) -> None:
  """Launches a PTX kernel on the given device buffers in a thunky Jaxpr."""
  pairs = [_unwrap_ref_and_spec(b) for b in buffers]
  base_refs = [p[0] for p in pairs]
  slice_specs = tuple(p[1] for p in pairs)
  ptx_kernel_p.bind(
      *base_refs,
      slice_specs=slice_specs,
      written=tuple(written),
      kernel_name=str(kernel_name),
      ptx=str(ptx),
      grid_dim=tuple(int(d) for d in grid_dim),
      block_dim=tuple(int(d) for d in block_dim),
      shmem_bytes=int(shmem_bytes),
  )


# --- custom_call ---

custom_call_p = core.Primitive("thunky.custom_call")
custom_call_p.multiple_results = True


def _custom_call_abstract_eval(
    *buffers: AbstractRef,
    slice_specs: tuple[SliceSpec, ...],
    num_operands: int,
    num_results: int,
    target_name: str,
    backend_config: tuple[tuple[str, Any], ...],
) -> tuple[list[Any], set[Any]]:
  del slice_specs, target_name, backend_config
  for b in buffers:
    if not _is_buffer_aval(b):
      raise TypeError(f"custom_call expects AbstractRef arguments, got {b}")
  effs = (
      {ReadEffect(i) for i in range(num_operands)}
      | {WriteEffect(num_operands + j) for j in range(num_results)}
      | {thunk_effect}
  )
  return [], effs


custom_call_p.def_effectful_abstract_eval(_custom_call_abstract_eval)


def custom_call(
    target_name: str,
    operands: Sequence[Any],
    results: Sequence[Any],
    backend_config: Mapping[str, Any] | None = None,
) -> None:
  """Invokes an XLA:GPU CustomCallThunk on operand and result buffers in a thunky Jaxpr."""
  op_pairs = [_unwrap_ref_and_spec(o) for o in operands]
  res_pairs = [_unwrap_ref_and_spec(r) for r in results]
  slice_specs = tuple(p[1] for p in op_pairs + res_pairs)
  config_tuple = tuple(sorted((backend_config or {}).items()))
  custom_call_p.bind(
      *[p[0] for p in op_pairs],
      *[p[0] for p in res_pairs],
      slice_specs=slice_specs,
      num_operands=len(op_pairs),
      num_results=len(res_pairs),
      target_name=str(target_name),
      backend_config=config_tuple,
  )


# --- call_jax (Thunk Splicing Primitive) ---

call_jax_p = core.Primitive("thunky.call_jax")
call_jax_p.multiple_results = True


def _check_manual_mesh_environment() -> None:
  abstract_mesh = jax.sharding.get_abstract_mesh()
  if not abstract_mesh.empty and not abstract_mesh.are_all_axes_manual:
    raise ValueError(
        "thunky requires all mesh axes to be in manual mode (e.g. inside"
        f" shard_map); got mesh with non-manual axes: {abstract_mesh}"
    )


def _unwrap_call_jax_arg(arg: Any) -> tuple[Any, SliceSpec, bool]:
  if isinstance(arg, TransformedRef):
    base_ref = arg.ref
    base_aval = core.typeof(base_ref)
    spec = _transforms_to_slice_params(base_aval, arg.transforms)
    return base_ref, spec, True
  aval = core.typeof(arg)
  if _is_buffer_aval(aval):
    return arg, (0, aval.shape, aval.dtype), True
  if isinstance(aval, core.ShapedArray):
    return (
        arg,
        (0, tuple(int(d) for d in aval.shape), np.dtype(aval.dtype)),
        False,
    )
  raise TypeError(
      f"call_jax expects Ref, TransformedRef, or jax.Array arguments, got {arg}"
  )


def _call_jax_abstract_eval(
    *args_avals: Any,
    slice_specs: tuple[SliceSpec, ...],
    is_ref: tuple[bool, ...],
    fn: Any,
) -> tuple[list[Any], set[Any]]:
  _check_manual_mesh_environment()
  in_fn_avals = [
      AbstractRef(core.ShapedArray(s[1], s[2]))
      if ref_flag
      else core.ShapedArray(s[1], s[2])
      for s, ref_flag in zip(slice_specs, is_ref)
  ]
  flat_avals, in_tree = jax.tree.flatten(in_fn_avals)
  dbg = jax.api_util.debug_info("thunky_call_jax", fn, in_fn_avals, {})
  flat_fn, _ = jax.api_util.flatten_fun_nokwargs(
      lu.wrap_init(fn, debug_info=dbg), in_tree
  )
  jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(flat_fn, flat_avals)
  closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
  var_to_idx = {v: i for i, v in enumerate(closed_jaxpr.invars)}
  effs: set[Any] = {thunk_effect}
  for eff in closed_jaxpr.effects:
    if isinstance(eff, (ReadEffect, WriteEffect, AccumEffect)):
      arg_idx = (
          eff.input if isinstance(eff.input, int) else var_to_idx.get(eff.input)
      )
      if arg_idx is not None:
        if isinstance(eff, ReadEffect):
          effs.add(ReadEffect(arg_idx))
        else:
          effs.add(WriteEffect(arg_idx))
  return [], effs


call_jax_p.def_effectful_abstract_eval(_call_jax_abstract_eval)


def call_jax(fn: Any, *args: Any) -> Any:
  """Calls a JAX function on device buffer Refs or Array values via thunk splicing."""
  unwrapped = [_unwrap_call_jax_arg(a) for a in args]
  base_args = [u[0] for u in unwrapped]
  slice_specs = tuple(u[1] for u in unwrapped)
  is_ref = tuple(u[2] for u in unwrapped)
  in_fn_avals = [
      AbstractRef(core.ShapedArray(s[1], s[2]))
      if ref_flag
      else core.ShapedArray(s[1], s[2])
      for s, ref_flag in zip(slice_specs, is_ref)
  ]
  _, in_tree = jax.tree.flatten(in_fn_avals)
  dbg = jax.api_util.debug_info("thunky_call_jax", fn, in_fn_avals, {})
  flat_fn, out_tree_thunk = jax.api_util.flatten_fun_nokwargs(
      lu.wrap_init(fn, debug_info=dbg), in_tree
  )
  _, out_avals, _ = pe.trace_to_jaxpr_dynamic(flat_fn, in_fn_avals)
  out_refs = [
      core.empty_ref(core.ShapedArray(a.shape, a.dtype)) for a in out_avals
  ]
  out_specs = tuple(
      (0, tuple(int(d) for d in a.shape), np.dtype(a.dtype)) for a in out_avals
  )
  num_in = len(args)

  def _fn_with_out_refs(*all_args: Any) -> None:
    in_args = all_args[:num_in]
    out_ref_args = all_args[num_in:]
    res = fn(*in_args)
    flat_res, _ = jax.tree.flatten(res)
    for out_r, val in zip(out_ref_args, flat_res):
      out_r[...] = val

  call_jax_p.bind(
      *base_args,
      *out_refs,
      slice_specs=slice_specs + out_specs,
      is_ref=is_ref + tuple(True for _ in out_refs),
      fn=_fn_with_out_refs,
  )
  out_tree = out_tree_thunk()
  if not out_avals:
    return None
  return jax.tree.unflatten(out_tree, out_refs)


# --- call_thunky_p (Nested @thunky.jit Splicing Primitive) ---

call_thunky_p = core.Primitive("thunky.call_thunky_nested")
call_thunky_p.multiple_results = True
call_thunky_p.def_impl(functools.partial(dispatch.apply_primitive, call_thunky_p))


def _call_thunky_nested_abstract_eval(
    *unique_buffers: AbstractRef,
    arg_to_unique_idx: tuple[int, ...],
    slice_specs: tuple[SliceSpec, ...],
    jaxpr: core.ClosedJaxpr,
) -> tuple[list[Any], set[Any]]:
  for b in unique_buffers:
    if not _is_buffer_aval(b):
      raise TypeError(
          f"@thunky.jit expects AbstractRef arguments, got {b}"
      )
  body_jaxpr = jaxpr.jaxpr
  var_to_idx = {
      v: i for i, v in enumerate(body_jaxpr.invars[: len(slice_specs)])
  }
  effs: set[Any] = {thunk_effect}
  for eff in body_jaxpr.effects:
    if isinstance(eff, (ReadEffect, WriteEffect)):
      param_idx = (
          eff.input if isinstance(eff.input, int) else var_to_idx.get(eff.input)
      )
      if param_idx is not None and 0 <= param_idx < len(slice_specs):
        unique_idx = arg_to_unique_idx[param_idx]
        effs.add(type(eff)(unique_idx))
  return [], effs


call_thunky_p.def_effectful_abstract_eval(_call_thunky_nested_abstract_eval)


# --- cond (Conditional Thunk Primitive) ---

cond_p = core.Primitive("thunky.cond")
cond_p.multiple_results = True


def _map_jaxpr_effect_to_outer(
    eff: Any, jaxpr: core.Jaxpr, outer_offset: int
) -> Any:
  if isinstance(eff, (ReadEffect, WriteEffect, AccumEffect)):
    all_vars = jaxpr.constvars + jaxpr.invars
    if isinstance(eff.input, int):
      idx = eff.input
    elif eff.input in all_vars:
      idx = all_vars.index(eff.input)
    else:
      return None
    return type(eff)(outer_offset + idx)
  return eff


def _cond_abstract_eval(
    pred_or_index_buf: AbstractRef,
    *const_bufs: AbstractRef,
    index_spec: SliceSpec,
    branch_jaxprs: tuple[core.Jaxpr, ...],
    branch_const_counts: tuple[int, ...],
) -> tuple[list[Any], set[Any]]:
  del const_bufs
  if not _is_buffer_aval(pred_or_index_buf):
    raise TypeError(
        "cond/switch expects an AbstractRef predicate/index, got"
        f" {pred_or_index_buf}"
    )
  if index_spec[1] != ():
    raise ValueError(
        "cond/switch branch index buffer must have scalar shape (), got"
        f" {index_spec[1]}"
    )
  if index_spec[2] == np.dtype(bool):
    if len(branch_jaxprs) != 2:
      raise ValueError(
          f"bool predicate requires 2 branches, got {len(branch_jaxprs)}"
      )
  elif index_spec[2] == np.dtype(np.int32):
    if len(branch_jaxprs) < 1:
      raise ValueError("switch requires at least 1 branch")
  else:
    raise ValueError(
        "cond/switch branch index buffer must have dtype bool or int32, got"
        f" {index_spec[2]}"
    )
  effs: set[Any] = {ReadEffect(0), thunk_effect}
  offset = 1
  for jaxpr, count in zip(branch_jaxprs, branch_const_counts):
    for eff in jaxpr.effects:
      mapped = _map_jaxpr_effect_to_outer(eff, jaxpr, offset)
      if mapped is not None:
        effs.add(mapped)
    offset += count
  return [], effs


cond_p.def_effectful_abstract_eval(_cond_abstract_eval)


def _trace_zero_arg_fn(
    fn: Callable[[], None], name: str
) -> tuple[core.Jaxpr, list[Any]]:
  _, in_tree = jax.tree.flatten(())
  dbg = jax.api_util.debug_info(name, fn, (), {})
  flat_fn, _ = jax.api_util.flatten_fun_nokwargs(
      lu.wrap_init(fn, debug_info=dbg), in_tree
  )
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fn, [])
  return jaxpr, consts


def switch(
    index_buf: Any,
    branches: Sequence[Callable[[], None]],
) -> None:
  """Executes branches[index_buf] via XLA:GPU multi-branch ConditionalThunk."""
  index_ref, index_spec = _unwrap_ref_and_spec(index_buf)
  branch_jaxprs = []
  all_consts = []
  branch_const_counts = []
  for i, fn in enumerate(branches):
    jaxpr, consts = _trace_zero_arg_fn(fn, f"thunky_switch_branch_{i}")
    branch_jaxprs.append(jaxpr)
    all_consts.extend(consts)
    branch_const_counts.append(len(consts))
  cond_p.bind(
      index_ref,
      *all_consts,
      index_spec=index_spec,
      branch_jaxprs=tuple(branch_jaxprs),
      branch_const_counts=tuple(branch_const_counts),
  )


def cond(
    pred_buf: Any,
    true_fn: Callable[[], None],
    false_fn: Callable[[], None],
) -> None:
  """Executes true_fn if pred_buf is true, else false_fn, via XLA:GPU ConditionalThunk."""
  switch(pred_buf, (true_fn, false_fn))


# --- while_loop (While Thunk Primitive) ---

while_p = core.Primitive("thunky.while")
while_p.multiple_results = True


def _while_abstract_eval(
    cond_buf: AbstractRef,
    *const_bufs: AbstractRef,
    cond_spec: SliceSpec,
    cond_jaxpr: core.Jaxpr,
    body_jaxpr: core.Jaxpr,
    num_cond_consts: int,
) -> tuple[list[Any], set[Any]]:
  del const_bufs
  if not _is_buffer_aval(cond_buf):
    raise TypeError(
        f"while_loop expects an AbstractRef condition, got {cond_buf}"
    )
  if cond_spec[2] != np.dtype(bool) or cond_spec[1] != ():
    raise ValueError(
        "while_loop condition buffer must have shape () and dtype bool, got"
        f" {cond_spec[1]} {cond_spec[2]}"
    )
  effs: set[Any] = {ReadEffect(0), thunk_effect}
  for eff in cond_jaxpr.effects:
    mapped = _map_jaxpr_effect_to_outer(eff, cond_jaxpr, 1)
    if mapped is not None:
      effs.add(mapped)
  for eff in body_jaxpr.effects:
    mapped = _map_jaxpr_effect_to_outer(eff, body_jaxpr, 1 + num_cond_consts)
    if mapped is not None:
      effs.add(mapped)
  return [], effs


while_p.def_effectful_abstract_eval(_while_abstract_eval)


def while_loop(
    cond_buf: Any,
    cond_fn: Callable[[], None],
    body_fn: Callable[[], None],
) -> None:
  """Executes body_fn while cond_fn writes true to cond_buf, via XLA:GPU WhileThunk."""
  cond_ref, cond_spec = _unwrap_ref_and_spec(cond_buf)
  cond_jaxpr, cond_consts = _trace_zero_arg_fn(cond_fn, "thunky_while_cond")
  body_jaxpr, body_consts = _trace_zero_arg_fn(body_fn, "thunky_while_body")
  while_p.bind(
      cond_ref,
      *cond_consts,
      *body_consts,
      cond_spec=cond_spec,
      cond_jaxpr=cond_jaxpr,
      body_jaxpr=body_jaxpr,
      num_cond_consts=len(cond_consts),
  )


# --- async_start & async_done (Multi-Stream Execution Primitives) ---

async_start_p = core.Primitive("thunky.async_start")


def _async_start_abstract_eval(
    *const_bufs: AbstractRef,
    body_jaxpr: core.Jaxpr,
    stream_id: int,
    stream_kind: str,
) -> tuple[core.AbstractToken, set[Any]]:
  del const_bufs, stream_id, stream_kind
  effs: set[Any] = {thunk_effect}
  for eff in body_jaxpr.effects:
    mapped = _map_jaxpr_effect_to_outer(eff, body_jaxpr, 0)
    if mapped is not None:
      effs.add(mapped)
  return core.abstract_token, effs


async_start_p.def_effectful_abstract_eval(_async_start_abstract_eval)


def async_start(
    fn: Callable[[], None],
    *,
    stream_id: int = 0,
    stream_kind: str = "computation",
    is_communication: bool | None = None,
) -> Any:
  """Launches fn asynchronously on a computation or communication stream and returns a token."""
  if is_communication is not None:
    stream_kind = "communication" if is_communication else "computation"
  if stream_kind not in ("computation", "communication"):
    raise ValueError(
        "stream_kind must be 'computation' or 'communication', got"
        f" {stream_kind!r}"
    )
  body_jaxpr, body_consts = _trace_zero_arg_fn(fn, "thunky_async_start")
  return async_start_p.bind(
      *body_consts,
      body_jaxpr=body_jaxpr,
      stream_id=int(stream_id),
      stream_kind=stream_kind,
  )


async_done_p = core.Primitive("thunky.async_done")
async_done_p.multiple_results = True


def _async_done_abstract_eval(
    token: core.AbstractToken,
) -> tuple[list[Any], set[Any]]:
  if not isinstance(token, core.AbstractToken):
    raise TypeError(f"async_done expects an AbstractToken, got {token}")
  return [], {thunk_effect}


async_done_p.def_effectful_abstract_eval(_async_done_abstract_eval)


def async_done(token: Any) -> None:
  """Waits for the async execution associated with token to complete on the current stream."""
  async_done_p.bind(token)


# --- Collectives (NCCL Multi-GPU Primitives) ---


def _resolve_replica_groups(
    axis_name: Any,
    axis_index_groups: Sequence[Sequence[int]] | None = None,
) -> tuple[tuple[int, ...], ...]:
  _check_manual_mesh_environment()
  mesh = jax.sharding.get_abstract_mesh()
  if mesh.empty:
    raise NameError(
        f"unbound axis name: {axis_name}. Collectives in thunky require an"
        " active manual mesh context (e.g. inside jax.shard_map or"
        " jax.sharding.use_abstract_mesh)."
    )
  axes = (
      tuple(axis_name)
      if isinstance(axis_name, (tuple, list))
      else (axis_name,)
  )
  mesh_axes = tuple(pxla._axis_read(mesh.axis_names, a) for a in axes)
  trailing_size, ragged = divmod(mesh.size, math.prod(mesh.axis_sizes))
  assert not ragged
  mesh_spec = mesh.axis_sizes + (trailing_size,)
  replica_groups = pxla._axis_groups(mesh_spec, mesh_axes)
  if axis_index_groups is not None:
    axis_space = range(sum(len(group) for group in axis_index_groups))
    if {i for g in axis_index_groups for i in g} != set(axis_space):
      raise ValueError("axis_index_groups must cover all indices exactly once")
    if len(replica_groups[0]) != len(axis_space):
      raise ValueError(
          f"axis_index_groups total size ({len(axis_space)}) must match axis"
          f" size ({len(replica_groups[0])})"
      )
    return tuple(
        tuple(int(axis_group[i]) for i in axis_index_group)
        for axis_group in replica_groups
        for axis_index_group in axis_index_groups
    )
  return tuple(
      tuple(int(x) for x in axis_group) for axis_group in replica_groups
  )


all_reduce_p = core.Primitive("thunky.all_reduce")
all_reduce_p.multiple_results = True


def _all_reduce_abstract_eval(
    src: AbstractRef,
    dst: AbstractRef,
    *,
    src_spec: SliceSpec,
    dst_spec: SliceSpec,
    reduction_kind: str,
    replica_groups: tuple[tuple[int, ...], ...],
) -> tuple[list[Any], set[Any]]:
  del reduction_kind, replica_groups
  if not _is_buffer_aval(src) or not _is_buffer_aval(dst):
    raise TypeError(
        f"all_reduce expects AbstractRef arguments, got {src}, {dst}"
    )
  if src_spec[1] != dst_spec[1] or src_spec[2] != dst_spec[2]:
    raise ValueError(
        f"all_reduce shape/dtype mismatch: {src_spec[1]} {src_spec[2]} vs"
        f" {dst_spec[1]} {dst_spec[2]}"
    )
  return [], {ReadEffect(0), WriteEffect(1)}


all_reduce_p.def_effectful_abstract_eval(_all_reduce_abstract_eval)


def all_reduce(
    src: Any,
    dst: Any,
    axis_name: Any,
    *,
    reduction: str = "sum",
    axis_index_groups: Sequence[Sequence[int]] | None = None,
) -> None:
  """Executes an XLA:GPU AllReduceThunk across participating devices."""
  src_ref, src_spec = _unwrap_ref_and_spec(src)
  dst_ref, dst_spec = _unwrap_ref_and_spec(dst)
  groups = _resolve_replica_groups(axis_name, axis_index_groups)
  all_reduce_p.bind(
      src_ref,
      dst_ref,
      src_spec=src_spec,
      dst_spec=dst_spec,
      reduction_kind=str(reduction),
      replica_groups=groups,
  )


all_gather_p = core.Primitive("thunky.all_gather")
all_gather_p.multiple_results = True


def _all_gather_abstract_eval(
    src: AbstractRef,
    dst: AbstractRef,
    *,
    src_spec: SliceSpec,
    dst_spec: SliceSpec,
    replica_groups: tuple[tuple[int, ...], ...],
) -> tuple[list[Any], set[Any]]:
  del replica_groups
  if not _is_buffer_aval(src) or not _is_buffer_aval(dst):
    raise TypeError(
        f"all_gather expects AbstractRef arguments, got {src}, {dst}"
    )
  if src_spec[2] != dst_spec[2]:
    raise ValueError(
        f"all_gather dtype mismatch: {src_spec[2]} vs {dst_spec[2]}"
    )
  return [], {ReadEffect(0), WriteEffect(1)}


all_gather_p.def_effectful_abstract_eval(_all_gather_abstract_eval)


def all_gather(
    src: Any,
    dst: Any,
    axis_name: Any,
    *,
    axis_index_groups: Sequence[Sequence[int]] | None = None,
) -> None:
  """Executes an XLA:GPU AllGatherThunk across participating devices."""
  src_ref, src_spec = _unwrap_ref_and_spec(src)
  dst_ref, dst_spec = _unwrap_ref_and_spec(dst)
  groups = _resolve_replica_groups(axis_name, axis_index_groups)
  all_gather_p.bind(
      src_ref,
      dst_ref,
      src_spec=src_spec,
      dst_spec=dst_spec,
      replica_groups=groups,
  )


reduce_scatter_p = core.Primitive("thunky.reduce_scatter")
reduce_scatter_p.multiple_results = True


def _reduce_scatter_abstract_eval(
    src: AbstractRef,
    dst: AbstractRef,
    *,
    src_spec: SliceSpec,
    dst_spec: SliceSpec,
    reduction_kind: str,
    replica_groups: tuple[tuple[int, ...], ...],
) -> tuple[list[Any], set[Any]]:
  del reduction_kind, replica_groups
  if not _is_buffer_aval(src) or not _is_buffer_aval(dst):
    raise TypeError(
        f"reduce_scatter expects AbstractRef arguments, got {src}, {dst}"
    )
  if src_spec[2] != dst_spec[2]:
    raise ValueError(
        f"reduce_scatter dtype mismatch: {src_spec[2]} vs {dst_spec[2]}"
    )
  return [], {ReadEffect(0), WriteEffect(1)}


reduce_scatter_p.def_effectful_abstract_eval(_reduce_scatter_abstract_eval)


def reduce_scatter(
    src: Any,
    dst: Any,
    axis_name: Any,
    *,
    reduction: str = "sum",
    axis_index_groups: Sequence[Sequence[int]] | None = None,
) -> None:
  """Executes an XLA:GPU ReduceScatterThunk across participating devices."""
  src_ref, src_spec = _unwrap_ref_and_spec(src)
  dst_ref, dst_spec = _unwrap_ref_and_spec(dst)
  groups = _resolve_replica_groups(axis_name, axis_index_groups)
  reduce_scatter_p.bind(
      src_ref,
      dst_ref,
      src_spec=src_spec,
      dst_spec=dst_spec,
      reduction_kind=str(reduction),
      replica_groups=groups,
  )


all_to_all_p = core.Primitive("thunky.all_to_all")
all_to_all_p.multiple_results = True


def _all_to_all_abstract_eval(
    *all_refs: AbstractRef,
    src_specs: tuple[SliceSpec, ...],
    dst_specs: tuple[SliceSpec, ...],
    replica_groups: tuple[tuple[int, ...], ...],
    has_split_dimension: bool,
) -> tuple[list[Any], set[Any]]:
  del replica_groups, has_split_dimension
  n = len(src_specs)
  srcs = all_refs[:n]
  dsts = all_refs[n:]
  for s, d, s_spec, d_spec in zip(srcs, dsts, src_specs, dst_specs):
    if not _is_buffer_aval(s) or not _is_buffer_aval(d):
      raise TypeError(f"all_to_all expects AbstractRef arguments, got {s}, {d}")
    if _spec_nbytes(s_spec) != _spec_nbytes(d_spec) or s_spec[2] != d_spec[2]:
      raise ValueError(f"all_to_all buffer mismatch: {s_spec} vs {d_spec}")
  effs = {ReadEffect(i) for i in range(n)} | {
      WriteEffect(n + i) for i in range(n)
  }
  return [], effs


all_to_all_p.def_effectful_abstract_eval(_all_to_all_abstract_eval)


def all_to_all(
    src: Any,
    dst: Any,
    axis_name: Any,
    *,
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    has_split_dimension: bool = True,
) -> None:
  """Executes an XLA:GPU AllToAllThunk across participating devices."""
  groups = _resolve_replica_groups(axis_name, axis_index_groups)
  num_ranks = len(groups[0]) if groups else 0
  has_split = bool(has_split_dimension)

  if has_split:
    if isinstance(src, (tuple, list)):
      if len(src) != 1 or len(dst) != 1:
        raise ValueError(
            "all_to_all with has_split_dimension=True requires 1 src and 1 dst"
            f" Ref, got {len(src)} and {len(dst)}"
        )
      srcs, dsts = tuple(src), tuple(dst)
    else:
      srcs, dsts = (src,), (dst,)
  else:
    if not isinstance(src, (tuple, list)) or not isinstance(dst, (tuple, list)):
      raise ValueError(
          "all_to_all with has_split_dimension=False requires sequences of"
          f" {num_ranks} src and dst Refs (one per peer rank), got single"
          f" buffer {src} and {dst}"
      )
    if len(src) != num_ranks or len(dst) != num_ranks:
      raise ValueError(
          f"all_to_all with has_split_dimension=False requires {num_ranks} src"
          f" and dst Refs (matching replica_groups size), got {len(src)} and"
          f" {len(dst)}"
      )
    srcs, dsts = tuple(src), tuple(dst)

  src_unwrapped = [_unwrap_ref_and_spec(s) for s in srcs]
  dst_unwrapped = [_unwrap_ref_and_spec(d) for d in dsts]
  src_refs = [u[0] for u in src_unwrapped]
  dst_refs = [u[0] for u in dst_unwrapped]
  src_specs = tuple(u[1] for u in src_unwrapped)
  dst_specs = tuple(u[1] for u in dst_unwrapped)

  all_to_all_p.bind(
      *src_refs,
      *dst_refs,
      src_specs=src_specs,
      dst_specs=dst_specs,
      replica_groups=groups,
      has_split_dimension=has_split,
  )


collective_permute_p = core.Primitive("thunky.collective_permute")
collective_permute_p.multiple_results = True


def _collective_permute_abstract_eval(
    src: AbstractRef,
    dst: AbstractRef,
    *,
    src_spec: SliceSpec,
    dst_spec: SliceSpec,
    source_target_pairs: tuple[tuple[int, int], ...],
    replica_groups: tuple[tuple[int, ...], ...],
) -> tuple[list[Any], set[Any]]:
  del source_target_pairs, replica_groups
  if not _is_buffer_aval(src) or not _is_buffer_aval(dst):
    raise TypeError(
        f"collective_permute expects AbstractRef arguments, got {src}, {dst}"
    )
  if (
      _spec_nbytes(src_spec) != _spec_nbytes(dst_spec)
      or src_spec[2] != dst_spec[2]
  ):
    raise ValueError(
        f"collective_permute buffer mismatch: {src_spec} vs {dst_spec}"
    )
  return [], {ReadEffect(0), WriteEffect(1)}


collective_permute_p.def_effectful_abstract_eval(
    _collective_permute_abstract_eval
)


def collective_permute(
    src: Any,
    dst: Any,
    axis_name: Any,
    source_target_pairs: Sequence[tuple[int, int]] = ((0, 1), (1, 0)),
    *,
    axis_index_groups: Sequence[Sequence[int]] | None = None,
) -> None:
  """Executes an XLA:GPU CollectivePermuteThunk across participating devices."""
  src_ref, src_spec = _unwrap_ref_and_spec(src)
  dst_ref, dst_spec = _unwrap_ref_and_spec(dst)
  groups = _resolve_replica_groups(axis_name, axis_index_groups)
  group_size = len(groups[0]) if groups else 0
  pairs = tuple((int(s), int(t)) for s, t in source_target_pairs)
  srcs = [s % group_size for s, _ in pairs]
  dsts = [t % group_size for _, t in pairs]
  if len(srcs) != len(set(srcs)) or len(dsts) != len(set(dsts)):
    raise ValueError(
        "collective_permute sources and destinations must be unique, got"
        f" {source_target_pairs}."
    )
  expanded_pairs = tuple(
      (grp[s % group_size], grp[t % group_size])
      for grp in groups
      for s, t in pairs
  )
  collective_permute_p.bind(
      src_ref,
      dst_ref,
      src_spec=src_spec,
      dst_spec=dst_spec,
      source_target_pairs=expanded_pairs,
      replica_groups=groups,
  )


collective_group_p = core.Primitive("thunky.collective_group")
collective_group_p.multiple_results = True


def _collective_group_abstract_eval(
    *const_bufs: AbstractRef,
    body_jaxpr: core.Jaxpr,
) -> tuple[list[Any], set[Any]]:
  del const_bufs
  effs: set[Any] = {thunk_effect}
  for eff in body_jaxpr.effects:
    mapped = _map_jaxpr_effect_to_outer(eff, body_jaxpr, 0)
    if mapped is not None:
      effs.add(mapped)
  return [], effs


collective_group_p.def_effectful_abstract_eval(_collective_group_abstract_eval)


_ALLOWED_COLLECTIVE_GROUP_PRIMITIVES = {
    all_reduce_p,
    all_gather_p,
    reduce_scatter_p,
    collective_permute_p,
    all_to_all_p,
}


def collective_group(fn: Callable[[], None]) -> None:
  """Fuses nested collective operations into a single NCCL group dispatch (CollectiveGroupThunk)."""
  body_jaxpr, body_consts = _trace_zero_arg_fn(fn, "thunky_collective_group")
  for eqn in body_jaxpr.eqns:
    if eqn.primitive not in _ALLOWED_COLLECTIVE_GROUP_PRIMITIVES:
      allowed = sorted(p.name for p in _ALLOWED_COLLECTIVE_GROUP_PRIMITIVES)
      raise ValueError(
          "thunky.collective_group can only contain collective operations "
          f"({allowed}), got {eqn.primitive.name}"
      )
  collective_group_p.bind(
      *body_consts,
      body_jaxpr=body_jaxpr,
  )


# --- Jaxpr -> MLIR Translation Rules ---


def _make_replica_groups_attr(
    replica_groups: Sequence[Sequence[int]],
) -> ir.ArrayAttr:
  return ir.ArrayAttr.get(
      [ir.DenseI64ArrayAttr.get(list(g)) for g in replica_groups]
  )


def _mlir_get_slice(
    ctx: ir.Context,
    env: dict[core.Atom, ir.Value],
    var: core.Atom,
    spec: SliceSpec,
) -> ir.Value:
  base_val = env[var]
  offset_bytes = spec[0]
  slice_nbytes = _spec_nbytes(spec)
  if offset_bytes == 0 and slice_nbytes == _buffer_nbytes(var.aval):
    return base_val
  buf_ty = _thunky.mlir_buffer_type(slice_nbytes, ctx)
  return _thunky.slice_buffer(buf_ty, base_val, offset_bytes)


_thunky_mlir_rules: dict[core.Primitive, Callable[..., None]] = {}


def _lower_memzero(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  _thunky.memzero(
      _mlir_get_slice(ctx, env, eqn.invars[0], eqn.params["slice_spec"])
  )
_thunky_mlir_rules[memzero_p] = _lower_memzero


def _lower_memset(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  signed_val = struct.unpack(
      "<i", struct.pack("<I", eqn.params["value"] & 0xFFFFFFFF)
  )[0]
  _thunky.memset32(
      _mlir_get_slice(ctx, env, eqn.invars[0], eqn.params["slice_spec"]),
      signed_val,
  )
_thunky_mlir_rules[memset_p] = _lower_memset


def _lower_copy(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  src_val = _mlir_get_slice(ctx, env, eqn.invars[0], eqn.params["src_spec"])
  dst_val = _mlir_get_slice(ctx, env, eqn.invars[1], eqn.params["dst_spec"])
  _thunky.copy(src_val, dst_val)
_thunky_mlir_rules[copy_p] = _lower_copy


def _lower_ptx_kernel(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  slice_specs = eqn.params["slice_specs"]
  ir_bufs = [
      _mlir_get_slice(ctx, env, v, s) for v, s in zip(eqn.invars, slice_specs)
  ]
  _thunky.ptx_kernel(
      buffers=ir_bufs,
      kernel_name=eqn.params["kernel_name"],
      ptx=eqn.params["ptx"],
      grid_dim=list(eqn.params["grid_dim"]),
      block_dim=list(eqn.params["block_dim"]),
      written=list(eqn.params["written"]),
      shmem_bytes=eqn.params["shmem_bytes"],
  )
_thunky_mlir_rules[ptx_kernel_p] = _lower_ptx_kernel


def _to_mlir_attr(val: Any) -> ir.Attribute:
  if isinstance(val, ir.Attribute):
    return val
  if isinstance(val, (bool, np.bool_)):
    return ir.BoolAttr.get(bool(val))
  if isinstance(val, (int, np.integer)):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), int(val))
  if isinstance(val, (float, np.floating)):
    return ir.FloatAttr.get(ir.F64Type.get(), float(val))
  if isinstance(val, (str, bytes)):
    return ir.StringAttr.get(val)
  if isinstance(val, (list, tuple, np.ndarray)):
    arr = np.asarray(val)
    if np.issubdtype(arr.dtype, np.integer):
      return ir.DenseI64ArrayAttr.get([int(x) for x in arr.flat])
    if np.issubdtype(arr.dtype, np.floating):
      return ir.DenseF64ArrayAttr.get([float(x) for x in arr.flat])
  if isinstance(val, Mapping):
    return ir.DictAttr.get({k: _to_mlir_attr(v) for k, v in val.items()})
  raise TypeError(f"Unsupported backend_config attribute type: {type(val)}")


def _lower_custom_call(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  num_operands = eqn.params["num_operands"]
  slice_specs = eqn.params["slice_specs"]
  op_specs = slice_specs[:num_operands]
  res_specs = slice_specs[num_operands:]
  operands = [
      _mlir_get_slice(ctx, env, v, s)
      for v, s in zip(eqn.invars[:num_operands], op_specs)
  ]
  results = [
      _mlir_get_slice(ctx, env, v, s)
      for v, s in zip(eqn.invars[num_operands:], res_specs)
  ]
  op_shapes = [
      ir.TypeAttr.get(
          ir.RankedTensorType.get(s[1], mlir.dtype_to_ir_type(s[2]))
      )
      for s in op_specs
  ]
  res_shapes = [
      ir.TypeAttr.get(
          ir.RankedTensorType.get(s[1], mlir.dtype_to_ir_type(s[2]))
      )
      for s in res_specs
  ]
  backend_config = {
      k: _to_mlir_attr(v) for k, v in eqn.params["backend_config"]
  }
  _thunky.custom_call(
      buffers=operands + results,
      num_operands=len(operands),
      num_results=len(results),
      target_name=eqn.params["target_name"],
      backend_config=backend_config,
      operand_shapes=op_shapes,
      result_shapes=res_shapes,
  )
_thunky_mlir_rules[custom_call_p] = _lower_custom_call


def _add_main_buffer_arg(
    ctx: ir.Context, target_block: ir.Block, size_bytes: int
) -> ir.Value:
  curr_op = target_block.owner.operation
  while curr_op is not None and curr_op.name != "func.func":
    parent = curr_op.parent
    curr_op = parent.operation if parent is not None else None
  if curr_op is None:
    raise ValueError("target_block is not inside a func.func operation")
  main_entry_block = curr_op.regions[0].blocks[0]
  buf_ty = _thunky.mlir_buffer_type(size_bytes, ctx)
  new_arg = main_entry_block.add_argument(buf_ty, ir.Location.unknown(ctx))
  curr_op.attributes["function_type"] = ir.TypeAttr.get(
      ir.FunctionType.get([arg.type for arg in main_entry_block.arguments], [])
  )
  return new_arg


def _lower_call_jax(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, Any],
    eqn: core.JaxprEqn,
) -> None:
  fn = eqn.params["fn"]
  slice_specs = eqn.params["slice_specs"]
  is_ref = eqn.params.get("is_ref", tuple(True for _ in slice_specs))
  unique_ir_vals: list[ir.Value] = []
  unique_avals: list[Any] = []
  key_to_idx: dict[tuple[ir.Value, bool], int] = {}
  arg_specs: list[tuple[bool, Any]] = []
  for v, s, ref_flag in zip(eqn.invars, slice_specs, is_ref):
    if isinstance(v, core.Literal):
      arg_specs.append((False, v.val))
    elif v in env and not isinstance(env[v], ir.Value):
      arg_specs.append((False, env[v]))
    else:
      ir_val = _mlir_get_slice(ctx, env, v, s)
      key = (ir_val, ref_flag)
      if key not in key_to_idx:
        key_to_idx[key] = len(unique_ir_vals)
        unique_ir_vals.append(ir_val)
        shaped_aval = core.ShapedArray(s[1], s[2])
        unique_avals.append(
            AbstractRef(shaped_aval) if ref_flag else shaped_aval
        )
      arg_specs.append((True, key_to_idx[key]))

  def _wrapped_fn(*unique_args: Any) -> Any:
    actual_args = [
        unique_args[val] if is_dyn else val for is_dyn, val in arg_specs
    ]
    return fn(*actual_args)

  active_mesh = jax.sharding.get_abstract_mesh()
  if not active_mesh.empty:
    _wrapped_fn = jax.shard_map(
        _wrapped_fn,
        mesh=active_mesh,
        in_specs=tuple(jax.sharding.PartitionSpec() for _ in unique_avals),
        out_specs=jax.sharding.PartitionSpec(),
        check_vma=False,
    )

  compiled = jax.jit(_wrapped_fn).lower(*unique_avals).compile()
  callee_module = jax_executable_to_mlir(compiled, ctx=ctx)

  out_ir_vals: list[ir.Value] = []
  for out_var in eqn.outvars:
    new_out_ir = _add_main_buffer_arg(
        ctx, target_block, _buffer_nbytes(out_var.aval)
    )
    env[out_var] = new_out_ir
    out_ir_vals.append(new_out_ir)

  splice_thunk_module(target_block, callee_module, unique_ir_vals + out_ir_vals)
_thunky_mlir_rules[call_jax_p] = _lower_call_jax


def _lower_call_thunky_nested(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, Any],
    eqn: core.JaxprEqn,
) -> None:
  closed_jaxpr = eqn.params["jaxpr"]
  arg_to_unique_idx = eqn.params["arg_to_unique_idx"]
  slice_specs = eqn.params["slice_specs"]
  in_vals = [
      _mlir_get_slice(ctx, env, eqn.invars[u], s)
      for u, s in zip(arg_to_unique_idx, slice_specs)
  ]
  callee_module = jaxpr_to_mlir(closed_jaxpr, ctx=ctx)
  splice_thunk_module(target_block, callee_module, in_vals)
_thunky_mlir_rules[call_thunky_p] = _lower_call_thunky_nested


def _emit_sliced_ref_ir(
    ctx: ir.Context,
    base_ir_val: ir.Value,
    base_aval: Any,
    transforms: Sequence[Any],
) -> tuple[ir.Value, core.ShapedArray]:
  offset_bytes, shape, dtype = _transforms_to_slice_params(
      base_aval, transforms
  )
  slice_nbytes = _shape_dtype_nbytes(shape, dtype)
  if offset_bytes == 0 and slice_nbytes == _buffer_nbytes(base_aval):
    return base_ir_val, core.ShapedArray(shape, dtype)
  buf_ty = _thunky.mlir_buffer_type(slice_nbytes, ctx)
  sliced = _thunky.slice_buffer(buf_ty, base_ir_val, offset_bytes)
  return sliced, core.ShapedArray(shape, dtype)


def _lower_get(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, Any],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  ref_var = eqn.invars[0]
  if "tree" in eqn.params:
    transforms = tree_util.tree_unflatten(eqn.params["tree"], eqn.invars[1:])
  else:
    transforms = eqn.params.get("transforms", ())
  sliced_ir, _ = _emit_sliced_ref_ir(
      ctx, env[ref_var], ref_var.aval, transforms
  )
  env[eqn.outvars[0]] = sliced_ir
_thunky_mlir_rules[get_p] = _lower_get


def _lower_swap(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, Any],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  ref_var = eqn.invars[0]
  val_atom = eqn.invars[1]
  if (
      not isinstance(val_atom, core.Var)
      or val_atom not in env
      or not isinstance(env[val_atom], ir.Value)
  ):
    raise ValueError(
        f"Cannot lower Ref assignment of {val_atom} inside @thunky.jit: only"
        " direct Ref-to-Ref copies (e.g. dst[...] = src[...]) are supported"
        " outside thunky.call_jax."
    )
  if "tree" in eqn.params:
    transforms = tree_util.tree_unflatten(eqn.params["tree"], eqn.invars[2:])
  else:
    transforms = eqn.params.get("transforms", ())
  dst_ir, _ = _emit_sliced_ref_ir(ctx, env[ref_var], ref_var.aval, transforms)
  _thunky.copy(env[val_atom], dst_ir)
_thunky_mlir_rules[swap_p] = _lower_swap


def _lower_jaxpr_eqns(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, Any],
    eqns: Sequence[core.JaxprEqn],
) -> None:
  for eqn in eqns:
    rule = _thunky_mlir_rules.get(eqn.primitive)
    if rule is None:
      raise ValueError(
          f"Unsupported primitive {eqn.primitive.name!r} inside @thunky.jit."
          " Wrap JAX array operations in thunky.call_jax."
      )
    rule(ctx, target_block, env, eqn)


def _lower_cond(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  index_val = _mlir_get_slice(ctx, env, eqn.invars[0], eqn.params["index_spec"])
  branch_jaxprs: tuple[core.Jaxpr, ...] = eqn.params["branch_jaxprs"]
  branch_const_counts: tuple[int, ...] = eqn.params["branch_const_counts"]

  cond_op = _thunky.cond(
      branch_index=index_val, num_branches=len(branch_jaxprs)
  )
  offset = 1
  for i, (jaxpr, count) in enumerate(zip(branch_jaxprs, branch_const_counts)):
    consts = eqn.invars[offset : offset + count]
    offset += count
    branch_env = dict(env)
    for cv, outer_v in zip(jaxpr.constvars + jaxpr.invars, consts):
      branch_env[cv] = env[outer_v]
    block = cond_op.branches[i].blocks.append()
    with ir.InsertionPoint(block):
      _lower_jaxpr_eqns(ctx, block, branch_env, jaxpr.eqns)
      _thunky.yield_()
_thunky_mlir_rules[cond_p] = _lower_cond


def _lower_while(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  cond_val = _mlir_get_slice(ctx, env, eqn.invars[0], eqn.params["cond_spec"])
  num_cond = eqn.params["num_cond_consts"]
  cond_consts = eqn.invars[1 : 1 + num_cond]
  body_consts = eqn.invars[1 + num_cond :]
  cond_jaxpr = eqn.params["cond_jaxpr"]
  body_jaxpr = eqn.params["body_jaxpr"]

  cond_env = dict(env)
  for cv, outer_v in zip(cond_jaxpr.constvars + cond_jaxpr.invars, cond_consts):
    cond_env[cv] = env[outer_v]

  body_env = dict(env)
  for cv, outer_v in zip(body_jaxpr.constvars + body_jaxpr.invars, body_consts):
    body_env[cv] = env[outer_v]

  while_op = _thunky.while_(condition_buffer=cond_val)
  cond_block = while_op.cond_region.blocks.append()
  with ir.InsertionPoint(cond_block):
    _lower_jaxpr_eqns(ctx, cond_block, cond_env, cond_jaxpr.eqns)
    _thunky.yield_()

  body_block = while_op.body_region.blocks.append()
  with ir.InsertionPoint(body_block):
    _lower_jaxpr_eqns(ctx, body_block, body_env, body_jaxpr.eqns)
    _thunky.yield_()
_thunky_mlir_rules[while_p] = _lower_while


def _lower_async_start(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  body_jaxpr = eqn.params["body_jaxpr"]
  body_consts = eqn.invars
  body_env = dict(env)
  for cv, outer_v in zip(body_jaxpr.constvars + body_jaxpr.invars, body_consts):
    body_env[cv] = env[outer_v]

  stream_kind = eqn.params.get("stream_kind")
  if stream_kind is not None:
    is_comm = stream_kind == "communication"
  else:
    is_comm = bool(eqn.params.get("is_communication", False))

  i64_type = ir.IntegerType.get_signless(64)
  start_op = _thunky.AsyncStartOp(
      token=_thunky.mlir_token_type(ctx),
      stream_id=ir.IntegerAttr.get(i64_type, eqn.params["stream_id"]),
      is_communication=ir.BoolAttr.get(is_comm),
  )
  body_block = start_op.body_region.blocks.append()
  with ir.InsertionPoint(body_block):
    _lower_jaxpr_eqns(ctx, body_block, body_env, body_jaxpr.eqns)
    _thunky.yield_()
  env[eqn.outvars[0]] = start_op.token
_thunky_mlir_rules[async_start_p] = _lower_async_start


def _lower_async_done(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  del ctx, target_block
  _thunky.async_done(token=env[eqn.invars[0]])
_thunky_mlir_rules[async_done_p] = _lower_async_done


def _lower_all_reduce(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  src_var, dst_var = eqn.invars
  src_spec = eqn.params["src_spec"]
  dst_spec = eqn.params["dst_spec"]
  elem_type = ir.TypeAttr.get(mlir.dtype_to_ir_type(src_spec[2]))
  _thunky.all_reduce(
      src=_mlir_get_slice(ctx, env, src_var, src_spec),
      dst=_mlir_get_slice(ctx, env, dst_var, dst_spec),
      reduction_kind=eqn.params["reduction_kind"],
      element_type=elem_type,
      replica_groups=_make_replica_groups_attr(eqn.params["replica_groups"]),
      group_mode="flattened_id",
  )
_thunky_mlir_rules[all_reduce_p] = _lower_all_reduce


def _lower_all_gather(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  src_var, dst_var = eqn.invars
  src_spec = eqn.params["src_spec"]
  dst_spec = eqn.params["dst_spec"]
  elem_type = ir.TypeAttr.get(mlir.dtype_to_ir_type(src_spec[2]))
  _thunky.all_gather(
      src=_mlir_get_slice(ctx, env, src_var, src_spec),
      dst=_mlir_get_slice(ctx, env, dst_var, dst_spec),
      element_type=elem_type,
      replica_groups=_make_replica_groups_attr(eqn.params["replica_groups"]),
      group_mode="flattened_id",
  )
_thunky_mlir_rules[all_gather_p] = _lower_all_gather


def _lower_reduce_scatter(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  src_var, dst_var = eqn.invars
  src_spec = eqn.params["src_spec"]
  dst_spec = eqn.params["dst_spec"]
  elem_type = ir.TypeAttr.get(mlir.dtype_to_ir_type(src_spec[2]))
  _thunky.reduce_scatter(
      src=_mlir_get_slice(ctx, env, src_var, src_spec),
      dst=_mlir_get_slice(ctx, env, dst_var, dst_spec),
      reduction_kind=eqn.params["reduction_kind"],
      element_type=elem_type,
      replica_groups=_make_replica_groups_attr(eqn.params["replica_groups"]),
      group_mode="flattened_id",
  )
_thunky_mlir_rules[reduce_scatter_p] = _lower_reduce_scatter


def _lower_all_to_all(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, Any],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  src_specs = eqn.params["src_specs"]
  dst_specs = eqn.params["dst_specs"]
  n = len(src_specs)
  src_vars = eqn.invars[:n]
  dst_vars = eqn.invars[n:]
  elem_type = ir.TypeAttr.get(mlir.dtype_to_ir_type(src_specs[0][2]))
  srcs = [_mlir_get_slice(ctx, env, v, s) for v, s in zip(src_vars, src_specs)]
  dsts = [_mlir_get_slice(ctx, env, v, s) for v, s in zip(dst_vars, dst_specs)]
  _thunky.all_to_all(
      src=srcs,
      dst=dsts,
      element_type=elem_type,
      replica_groups=_make_replica_groups_attr(eqn.params["replica_groups"]),
      group_mode="flattened_id",
      has_split_dimension=eqn.params["has_split_dimension"],
  )
_thunky_mlir_rules[all_to_all_p] = _lower_all_to_all


def _lower_collective_permute(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  src_var, dst_var = eqn.invars
  src_spec = eqn.params["src_spec"]
  dst_spec = eqn.params["dst_spec"]
  elem_type = ir.TypeAttr.get(mlir.dtype_to_ir_type(src_spec[2]))
  _thunky.collective_permute(
      src=_mlir_get_slice(ctx, env, src_var, src_spec),
      dst=_mlir_get_slice(ctx, env, dst_var, dst_spec),
      element_type=elem_type,
      source_target_pairs=_make_replica_groups_attr(
          eqn.params["source_target_pairs"]
      ),
      group_mode="flattened_id",
      replica_groups=_make_replica_groups_attr(eqn.params["replica_groups"]),
  )
_thunky_mlir_rules[collective_permute_p] = _lower_collective_permute


def _lower_collective_group(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  del target_block
  body_jaxpr = eqn.params["body_jaxpr"]
  body_consts = eqn.invars
  body_env = dict(env)
  for cv, outer_v in zip(body_jaxpr.constvars + body_jaxpr.invars, body_consts):
    body_env[cv] = env[outer_v]

  group_op = _thunky.collective_group()
  body_block = group_op.body_region.blocks.append()
  with ir.InsertionPoint(body_block):
    _lower_jaxpr_eqns(ctx, body_block, body_env, body_jaxpr.eqns)
    _thunky.yield_()
_thunky_mlir_rules[collective_group_p] = _lower_collective_group


def _lower_empty_ref(
    ctx: ir.Context,
    target_block: ir.Block,
    env: dict[core.Atom, ir.Value],
    eqn: core.JaxprEqn,
) -> None:
  out_var = eqn.outvars[0]
  env[out_var] = _add_main_buffer_arg(
      ctx, target_block, _buffer_nbytes(out_var.aval)
  )
_thunky_mlir_rules[core.empty_ref_p] = _lower_empty_ref


def _trace_thunk_jaxpr(
    fn: Callable[..., Any],
    in_avals: Sequence[AbstractRef],
    scratch_shapes: Any,
) -> core.ClosedJaxpr:
  def _to_ref_aval(s: Any) -> AbstractRef:
    if _is_buffer_aval(s):
      return s
    return AbstractRef(
        core.ShapedArray(tuple(int(d) for d in s.shape), np.dtype(s.dtype))
    )

  user_avals = tuple(_to_ref_aval(arg) for arg in in_avals)

  if scratch_shapes is None or scratch_shapes == ():
    all_args = user_avals
    all_kwargs: dict[str, Any] = {}
  elif isinstance(scratch_shapes, dict):
    all_args = user_avals
    all_kwargs = jax.tree.map(_to_ref_aval, scratch_shapes)
  elif isinstance(scratch_shapes, (tuple, list)):
    all_args = user_avals + tuple(jax.tree.map(_to_ref_aval, scratch_shapes))
    all_kwargs = {}
  else:
    all_args = user_avals + (jax.tree.map(_to_ref_aval, scratch_shapes),)
    all_kwargs = {}

  in_avals_ft = ft.flatten((all_args, all_kwargs))
  dbg = jax.api_util.debug_info("thunky_jit", fn, all_args, all_kwargs)
  active_mesh = jax.sharding.get_abstract_mesh()
  if not active_mesh.empty:
    with core.extend_axis_env_nd(active_mesh.shape.items()):
      closed_jaxpr, _ = pe.trace_to_jaxpr(fn, in_avals_ft, dbg)
  else:
    closed_jaxpr, _ = pe.trace_to_jaxpr(fn, in_avals_ft, dbg)
  return closed_jaxpr


def jaxpr_to_mlir(
    closed_jaxpr: core.ClosedJaxpr, ctx: ir.Context | None = None
) -> ir.Module:
  """Translates a thunky ClosedJaxpr into an ir.Module in the thunky dialect."""
  if ctx is None:
    ctx = make_ir_context()
  with ctx, ir.Location.unknown():
    module = ir.Module.create()
    arg_types = [
        _thunky.mlir_buffer_type(_buffer_nbytes(v.aval), ctx)
        for v in closed_jaxpr.jaxpr.invars
    ]
    func_type = ir.FunctionType.get(arg_types, [])
    with ir.InsertionPoint(module.body):
      func = func_dialect.FuncOp("main", func_type)
      entry_block = func.add_entry_block()
      env: dict[core.Atom, Any] = dict(
          zip(closed_jaxpr.jaxpr.invars, entry_block.arguments)
      )
      for cv, c_val in zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts):
        env[cv] = c_val
      with ir.InsertionPoint(entry_block):
        _lower_jaxpr_eqns(ctx, entry_block, env, closed_jaxpr.jaxpr.eqns)
        func_dialect.ReturnOp([])
    passmanager.PassManager.parse("builtin.module(inline)").run(
        module.operation
    )
    module.operation.verify()
    return module


def _build_call_thunky_mlir_module(
    unique_avals: Sequence[Any],
    arg_to_unique_idx: tuple[int, ...],
    slice_specs: tuple[SliceSpec, ...],
    closed_jaxpr: core.ClosedJaxpr,
    ctx: ir.Context | None = None,
) -> ir.Module:
  if ctx is None:
    ctx = make_ir_context()
  with ctx, ir.Location.unknown():
    is_identity = arg_to_unique_idx == tuple(range(len(unique_avals))) and all(
        s[0] == 0 and _spec_nbytes(s) == _buffer_nbytes(a)
        for s, a in zip(slice_specs, unique_avals)
    )
    if is_identity:
      return jaxpr_to_mlir(closed_jaxpr, ctx=ctx)

    module = ir.Module.create()
    arg_types = [
        _thunky.mlir_buffer_type(_buffer_nbytes(a), ctx) for a in unique_avals
    ]
    func_type = ir.FunctionType.get(arg_types, [])
    with ir.InsertionPoint(module.body):
      func = func_dialect.FuncOp("main", func_type)
      entry_block = func.add_entry_block()
      with ir.InsertionPoint(entry_block):
        in_vals = []
        for u, spec in zip(arg_to_unique_idx, slice_specs):
          block_arg = entry_block.arguments[u]
          base_aval = unique_avals[u]
          offset_bytes = spec[0]
          slice_nbytes = _spec_nbytes(spec)
          if offset_bytes == 0 and slice_nbytes == _buffer_nbytes(base_aval):
            in_vals.append(block_arg)
          else:
            buf_ty = _thunky.mlir_buffer_type(slice_nbytes, ctx)
            in_vals.append(
                _thunky.slice_buffer(buf_ty, block_arg, offset_bytes)
            )
        callee_module = jaxpr_to_mlir(closed_jaxpr, ctx=ctx)
        splice_thunk_module(entry_block, callee_module, in_vals)
        func_dialect.ReturnOp([])
    passmanager.PassManager.parse("builtin.module(inline)").run(
        module.operation
    )
    module.operation.verify()
    return module


_inline_constant_p = core.Primitive("thunky.inline_constant")


@_inline_constant_p.def_abstract_eval
def _inline_constant_abstract_eval(*, value: bytes) -> core.ShapedArray:
  return core.ShapedArray((len(value),), np.dtype(np.uint8))


@_inline_constant_p.def_impl
def _inline_constant_impl(*, value: bytes) -> np.ndarray:
  return np.frombuffer(value, dtype=np.uint8).copy()


def _inline_constant_lowering(
    ctx: mlir.LoweringRuleContext, *, value: bytes
) -> Sequence[ir.Value]:
  del ctx
  return [mlir.ir_constant(np.frombuffer(value, dtype=np.uint8))]


mlir.register_lowering(_inline_constant_p, _inline_constant_lowering)


def _collect_constant_bytes(op: ir.Operation) -> list[bytes]:
  result: list[bytes] = []
  if op.name == "thunky.constant":
    result.append(ir.StringAttr(op.attributes["value"]).value_bytes)
  for region in op.regions:
    for block in region.blocks:
      for child in block.operations:
        result.extend(_collect_constant_bytes(child.operation))
  return result


@state_discharge.register_discharge_rule(call_thunky_p)
def _call_thunky_discharge_rule(
    ctx: state_discharge.DischargeContext,
    *args: Any,
    arg_to_unique_idx: tuple[int, ...],
    slice_specs: tuple[SliceSpec, ...],
    jaxpr: core.ClosedJaxpr,
) -> tuple[Sequence[Any | None], Sequence[Any]]:
  body_jaxpr = jaxpr.jaxpr
  var_to_idx = {
      v: i for i, v in enumerate(body_jaxpr.invars[: len(slice_specs)])
  }
  written_unique_set: set[int] = set()
  for eff in body_jaxpr.effects:
    if isinstance(eff, WriteEffect):
      param_idx = (
          eff.input if isinstance(eff.input, int) else var_to_idx.get(eff.input)
      )
      if param_idx is not None and 0 <= param_idx < len(slice_specs):
        written_unique_set.add(arg_to_unique_idx[param_idx])
  written_indices = sorted(written_unique_set)

  with make_ir_context() as ir_ctx:
    module = _build_call_thunky_mlir_module(
        ctx.in_avals, arg_to_unique_idx, slice_specs, jaxpr, ctx=ir_ctx
    )
    module_proto_bytes = _thunky.mlir_module_to_gpu_executable_proto(module)
    main_func = None
    for op in module.body.operations:
      if (
          op.operation.name == "func.func"
          and ir.StringAttr(op.operation.attributes["sym_name"]).value == "main"
      ):
        main_func = op.operation
        break
    if main_func is None:
      raise ValueError("thunky MLIR module missing func.func @main")
    main_args = main_func.regions[0].blocks[0].arguments
    scratch_result_shapes = []
    for arg in main_args[len(args) :]:
      type_str = str(arg.type)
      size_bytes = int(type_str.split("<")[1].split(">")[0])
      scratch_result_shapes.append(
          jax.ShapeDtypeStruct((size_bytes,), np.uint8)
      )
    constant_bytes_list = _collect_constant_bytes(main_func)

  const_vals = [
      _inline_constant_p.bind(value=b) for b in constant_bytes_list
  ]
  aliased_result_shapes = [
      jax.ShapeDtypeStruct(args[i].shape, args[i].dtype)
      for i in written_indices
  ]
  input_output_aliases = {
      input_idx: output_idx
      for output_idx, input_idx in enumerate(written_indices)
  }
  all_result_shapes = aliased_result_shapes + scratch_result_shapes
  if not all_result_shapes:
    all_result_shapes = [jax.ShapeDtypeStruct((1,), np.uint8)]
  results = jax.ffi.ffi_call(
      "thunky.inline_module",
      all_result_shapes,
      input_output_aliases=input_output_aliases,
      has_side_effect=True,
  )(
      *args,
      *const_vals,
      module=module_proto_bytes,
      num_aliased_outputs=np.int64(len(written_indices)),
      num_scratch=np.int64(len(scratch_result_shapes)),
  )
  written_map = {
      input_idx: results[output_idx]
      for output_idx, input_idx in enumerate(written_indices)
  }
  new_invals = [
      written_map.get(i) if should else None
      for i, should in enumerate(ctx.should_discharge)
  ]
  return new_invals, []


def jit(
    fn: Callable[..., Any] | None = None,
    *,
    scratch_shapes: Any = (),
) -> Any:
  """Compiles a thunky DSL function into an XLA:GPU ThunkExecutor via Jaxpr -> MLIR."""
  _check_thunky_available()
  if fn is None:
    return functools.partial(jit, scratch_shapes=scratch_shapes)

  @functools.wraps(fn)
  def wrapped(*args: Any) -> None:
    _check_manual_mesh_environment()

    def _is_non_ref_arg(b: Any) -> bool:
      if isinstance(b, TransformedRef):
        return False
      return not isinstance(core.typeof(b), AbstractRef)

    def _arg_aval(b: Any) -> AbstractRef:
      if _is_non_ref_arg(b):
        aval = core.typeof(b)
        if not isinstance(aval, core.ShapedArray):
          raise TypeError(f"Expected AbstractRef or ShapedArray, got {aval}")
        return AbstractRef(core.ShapedArray(aval.shape, aval.dtype))
      _, spec = _unwrap_ref_and_spec(b)
      return AbstractRef(core.ShapedArray(spec[1], spec[2]))

    in_avals = [_arg_aval(b) for b in args]
    closed_jaxpr = _trace_thunk_jaxpr(fn, in_avals, scratch_shapes)

    if any(_is_non_ref_arg(b) for b in args):
      jaxpr = closed_jaxpr.jaxpr
      var_to_idx = {v: i for i, v in enumerate(jaxpr.invars[: len(args)])}
      written_params: set[int] = set()
      for eff in jaxpr.effects:
        if isinstance(eff, WriteEffect):
          idx = (
              eff.input
              if isinstance(eff.input, int)
              else var_to_idx.get(eff.input)
          )
          if idx is not None and 0 <= idx < len(args):
            written_params.add(idx)

      wrapped_args = []
      for i, b in enumerate(args):
        if _is_non_ref_arg(b):
          if i in written_params:
            raise TypeError(
                f"Argument {i} to {fn.__name__} is mutated in-place and"
                " must be passed as a Ref (e.g. jax.new_ref(...)), got"
                f" {core.typeof(b)}"
            )
          wrapped_args.append(jax.new_ref(b))
        else:
          wrapped_args.append(b)
      args = tuple(wrapped_args)

    pairs = [_unwrap_ref_and_spec(b) for b in args]
    unique_refs: list[Any] = []
    ref_id_to_idx: dict[int, int] = {}
    arg_to_unique_idx: list[int] = []
    for base_ref, _ in pairs:
      rid = id(base_ref)
      if rid not in ref_id_to_idx:
        ref_id_to_idx[rid] = len(unique_refs)
        unique_refs.append(base_ref)
      arg_to_unique_idx.append(ref_id_to_idx[rid])
    slice_specs = tuple(p[1] for p in pairs)
    call_thunky_p.bind(
        *unique_refs,
        arg_to_unique_idx=tuple(arg_to_unique_idx),
        slice_specs=slice_specs,
        jaxpr=closed_jaxpr,
    )

  return wrapped
