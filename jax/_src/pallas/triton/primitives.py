# Copyright 2024 The JAX Authors.
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

"""Module for GPU-specific JAX primitives."""

from __future__ import annotations

from collections.abc import Sequence
import enum
from typing import Any, TypeAlias

import jax
from jax._src import core as jax_core
from jax._src import effects
from jax._src import lax
from jax._src import state
from jax._src import tree_util
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith as arith_dialect
from jax._src.lib.mlir.dialects import gpu as gpu_dialect
from jax._src.lib.triton import dialect as tt_dialect
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives as pallas_primitives
from jax._src.pallas.triton import lowering
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax.interpreters import mlir
import jax.numpy as jnp
import numpy as np


Ref: TypeAlias = state.AbstractRef | state.TransformedRef

Slice = indexing.Slice


def approx_tanh(x: jax.Array) -> jax.Array:
  r"""Elementwise approximate hyperbolic tangent: :math:`\mathrm{tanh}(x)`.

  See
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh.
  """
  if x.dtype == jnp.float16:
    asm = "tanh.approx.f16 $0, $1;"
    constraint = "h"
  elif x.dtype == jnp.bfloat16:
    asm = "tanh.approx.bf16 $0, $1;"
    constraint = "h"
  elif x.dtype == jnp.float32:
    asm = "tanh.approx.f32 $0, $1;"
    constraint = "f"
  elif x.dtype == jnp.float64:
    # f64 tanh.approx is only supported on ROCm (uses __ocml_tanh_f64)
    # CUDA does not have a PTX instruction for f64 approximate tanh
    asm = "tanh.approx.f64 $0, $1;"
    constraint = "d"
  else:
    raise TypeError(f"approx_tanh does not accept {x.dtype} arrays")

  [result] = elementwise_inline_asm(
      asm,
      args=[x],
      constraints=f"={constraint},{constraint}",
      pack=1,
      result_shape_dtypes=[jax.ShapeDtypeStruct(x.shape, x.dtype)],
  )
  return result


def elementwise_inline_asm(
    asm: str,
    *,
    args: Sequence[jax.Array],
    constraints: str,
    pack: int,
    result_shape_dtypes: Sequence[jax.ShapeDtypeStruct],
) -> Sequence[jax.Array]:
  """Inline assembly applying an elementwise operation.

  Args:
    asm: The assembly code to run.
    args: The arguments to pass to the assembly code.
    constraints: LLVM inline assembly `constraints
      <https://llvm.org/docs/LangRef.html#inline-asm-constraint-string>`_.
    pack: The number of elements from each argument expected by a single
      instance of the assembly code.
    result_shape_dtypes: The shapes and dtypes of the results produced by the
      assembly code.

  Returns:
    The results produced by the assembly code.
  """
  return elementwise_inline_asm_p.bind(
      *args,
      asm=asm,
      constraints=constraints,
      pack=pack,
      result_shape_dtypes=tuple(result_shape_dtypes),
  )


elementwise_inline_asm_p = jax_core.Primitive("elementwise_inline_asm_p")
elementwise_inline_asm_p.multiple_results = True


@elementwise_inline_asm_p.def_abstract_eval
def _elementwise_inline_asm_abstract_eval(
    *avals: jax_core.ShapedArray, result_shape_dtypes, **kwargs
) -> Sequence[jax_core.ShapedArray]:
  del kwargs  # Unused.
  if not all(x.shape == y.shape for x, y in zip(avals, avals[1:])):
    raise ValueError(
        "All arguments of elementwise_inline_asm must have the same shape"
    )
  return [jax_core.ShapedArray(s.shape, s.dtype) for s in result_shape_dtypes]


@lowering.register_lowering(elementwise_inline_asm_p)
def _elementwise_inline_asm_lowering(
    ctx: lowering.LoweringRuleContext,
    *args,
    asm,
    constraints,
    pack,
    result_shape_dtypes,
):
  del result_shape_dtypes  # Unused.

  if "tanh.approx" in asm:
    if ctx.context.platform == "rocm":
      return _approx_tanh_rocm_lowering(ctx, *args)
    if ctx.avals_in[0].dtype == jnp.float64:
      raise TypeError(
          "approx_tanh does not support float64 on CUDA; it is only"
          " supported on ROCm"
      )

  return tt_dialect.ElementwiseInlineAsmOp(
      [*map(mlir.aval_to_ir_type, ctx.avals_out)],
      asm,
      constraints=constraints,
      pure=True,
      packed_element=pack,
      args=args,
  ).result


def _approx_tanh_rocm_lowering(
    ctx: lowering.LoweringRuleContext,
    *args,
):
  """Lower approx_tanh for ROCm.

  AMD CDNA3 (MI300X/gfx942) does not have a hardware tanh instruction.
  See: https://github.com/triton-lang/triton/pull/7780
  """
  [arg] = args
  [out_aval] = ctx.avals_out
  in_dtype = ctx.avals_in[0].dtype

  if in_dtype == jnp.float64:
    result_type = mlir.aval_to_ir_type(out_aval)
    result = tt_dialect.extern_elementwise(
        result_type,
        list(args),
        libname="",
        libpath="",
        symbol="__ocml_tanh_f64",
        pure=True,
    )
    return [result]

  needs_cast = in_dtype in (jnp.float16, jnp.bfloat16)

  if needs_cast:
    f32_type = mlir.dtype_to_ir_type(jnp.dtype(jnp.float32))
    if out_aval.shape:
      result_type = ir.RankedTensorType.get(out_aval.shape, f32_type)
    else:
      result_type = f32_type
    arg = arith_dialect.extf(result_type, arg)
  else:
    result_type = mlir.aval_to_ir_type(out_aval)
  result = tt_dialect.extern_elementwise(
      result_type,
      [arg],
      libname="libdevice",
      libpath="",
      symbol="__triton_hip_fast_tanhf",
      pure=True,
  )

  if needs_cast:
    out_type = mlir.aval_to_ir_type(out_aval)
    result = arith_dialect.truncf(out_type, result)

  return [result]


def debug_barrier() -> None:
  """Synchronizes all kernel executions in the grid."""
  return debug_barrier_p.bind()


class BarrierEffect(jax_core.Effect):
  pass

barrier_effect = BarrierEffect()

pallas_core.kernel_local_effects.add_type(BarrierEffect)
effects.control_flow_allowed_effects.add_type(BarrierEffect)


debug_barrier_p = jax_core.Primitive("debug_barrier_p")
debug_barrier_p.multiple_results = True


@debug_barrier_p.def_effectful_abstract_eval
def _debug_barrier_abstract_eval():
  return (), {barrier_effect}


@lowering.register_lowering(debug_barrier_p)
def _debug_barrier_lowering(ctx: lowering.LoweringRuleContext):
  del ctx  # Unused.
  gpu_dialect.barrier()
  return []


def load(
    ref: Ref,
    *,
    mask: jax.Array | None = None,
    other: jax.typing.ArrayLike | None = None,
    cache_modifier: str | None = None,
    eviction_policy: str | None = None,
    volatile: bool = False,
) -> jax.Array:
  """Loads an array from the given ref.

  If neither ``mask`` nor ``other`` is specified, this function has the same
  semantics as ``ref[idx]`` in JAX.

  Args:
    ref: The ref to load from.
    mask: An optional boolean mask specifying which indices to load. If mask is
      ``False`` and ``other`` is not given, no assumptions can be made about the
      value in the resulting array.
    other: An optional value to use for indices where mask is ``False``.
    cache_modifier: TO BE DOCUMENTED.
    eviction_policy: TO BE DOCUMENTED.
    volatile: TO BE DOCUMENTED.
  """
  return pallas_primitives.load(
      ref,
      None,
      mask=mask,
      other=other,
      cache_modifier=cache_modifier,
      eviction_policy=eviction_policy,
      volatile=volatile,
  )


def store(
    ref: Ref,
    val: jax.Array,
    *,
    mask: jax.Array | None = None,
    eviction_policy: str | None = None,
) -> None:
  """Stores a value to the given ref.

  See :func:`~jax.experimental.pallas.load` for the meaning of the arguments.
  """
  return pallas_primitives.store(
      ref,
      None,
      val,
      mask=mask,
      eviction_policy=eviction_policy,
  )


class AtomicOpType(enum.Enum):
  XCHG = "xchg"
  ADD = "add"
  MAX = "max"
  MIN = "min"
  AND = "and"
  OR = "or"
  XOR = "xor"


atomic_rmw_p = jax_core.Primitive("atomic_rmw")


def _atomic_rmw_discharge_rule(
    in_avals, out_avals, *args_flat, args_tree, atomic_type: AtomicOpType
):
  del out_avals  # Unused.
  ref, transforms, val, mask = args_tree.unflatten(args_flat)
  *prev_transforms, idx = transforms
  ref = state_discharge.transform_array(ref, prev_transforms)

  if mask is not None:
    raise NotImplementedError

  if atomic_type == AtomicOpType.ADD:
    monoid = lambda x, y: x + y
  elif atomic_type == AtomicOpType.MAX:
    monoid = jnp.maximum
  elif atomic_type == AtomicOpType.MIN:
    monoid = jnp.minimum
  else:
    raise NotImplementedError(atomic_type)

  if all((isinstance(s, Slice) or not s.shape) for s in idx.indices):
    indices = idx.indices
    scalar_dims = [not isinstance(s, Slice) and not s.shape for s in indices]
    slice_starts = [s.start if isinstance(s, Slice) else s for s in indices]
    slice_sizes = tuple(s.size if isinstance(s, Slice) else 1 for s in indices)
    out_ones = lax.dynamic_slice(ref, slice_starts, slice_sizes=slice_sizes)
    val_indexer = tuple(
        None if scalar else slice(None) for scalar in scalar_dims
    )
    val = val[val_indexer]
    val = monoid(val, out_ones)
    x_new = lax.dynamic_update_slice(ref, val, start_indices=slice_starts)
    out_indexer = tuple(0 if scalar else slice(None) for scalar in scalar_dims)
    out = out_ones[out_indexer]
  elif all(not isinstance(s, Slice) for s in idx.indices):
    out = ref[idx.indices]
    x_new = ref.at[idx.indices].set(monoid(out, val))
  else:
    raise NotImplementedError
  return (x_new,) + (None,) * (len(in_avals) - 1), out


state_discharge.register_discharge_rule(atomic_rmw_p)(
    _atomic_rmw_discharge_rule
)


@atomic_rmw_p.def_effectful_abstract_eval
def _atomic_abstract_eval(*avals_flat, args_tree, atomic_type: AtomicOpType):
  ref, _, _, _ = args_tree.unflatten(avals_flat)
  if ref.dtype == jnp.dtype("float16") and atomic_type != AtomicOpType.ADD:
    raise ValueError(f"`atomic_{atomic_type.value}` does not support f16.")
  if ref.dtype in {
      jnp.dtype("bool"),
      jnp.dtype("int8"),
      jnp.dtype("int16"),
      jnp.bfloat16,
  }:
    raise ValueError(
        f"`atomic_{atomic_type.value}` does not support {ref.dtype}."
    )
  return pallas_primitives._swap_abstract_eval(*avals_flat, args_tree=args_tree)


def _atomic_rmw(
    x_ref_or_view,
    idx,
    val,
    *,
    mask: Any | None = None,
    atomic_type: AtomicOpType,
):
  x_ref, transforms = state_primitives.get_ref_and_transforms(
      x_ref_or_view, idx, "atomic_rmw"
  )
  args_flat, args_tree = tree_util.tree_flatten((x_ref, transforms, val, mask))
  return atomic_rmw_p.bind(
      *args_flat, args_tree=args_tree, atomic_type=atomic_type
  )


@lowering.register_lowering(atomic_rmw_p)
def _atomic_lowering_rule(
    ctx: lowering.LoweringRuleContext,
    *args_flat,
    args_tree,
    atomic_type: AtomicOpType,
):
  block_info, *_ = ctx.block_infos
  assert block_info is not None
  ptr, indexers, val, mask = args_tree.unflatten(args_flat)
  *_, value_aval, mask_aval = args_tree.unflatten(ctx.avals_in)
  indexers = list(indexers)
  if not indexers or not isinstance(indexers[-1], indexing.NDIndexer):
    ref_aval = state.transform_type(indexers, ctx.avals_in[0])
    assert isinstance(ref_aval, state.AbstractRef)
    indexers.append(indexing.NDIndexer.make_trivial_indexer(ref_aval.shape))
  if len(indexers) != 1:
    raise NotImplementedError("Only single indexer is supported.")
  idx = indexers[0]
  ptr = lowering._compute_pointers_from_indices(ptr, block_info, idx)
  val = lowering._ensure_ir_value(val, value_aval)
  if mask is not None:
    mask = lowering._ensure_ir_value(mask, mask_aval)
  if atomic_type == AtomicOpType.XCHG:
    op = tt_dialect.RMWOp.XCHG
  elif atomic_type == AtomicOpType.ADD:
    if isinstance(val.type, ir.IntegerType):
      op = tt_dialect.RMWOp.ADD
    else:
      op = tt_dialect.RMWOp.FADD
  elif atomic_type == AtomicOpType.MIN:
    op = tt_dialect.RMWOp.MIN
  elif atomic_type == AtomicOpType.MAX:
    op = tt_dialect.RMWOp.MAX
  elif atomic_type == AtomicOpType.AND:
    op = tt_dialect.RMWOp.AND
  elif atomic_type == AtomicOpType.OR:
    op = tt_dialect.RMWOp.OR
  elif atomic_type == AtomicOpType.XOR:
    op = tt_dialect.RMWOp.XOR
  else:
    raise NotImplementedError(f"unsupported atomic operation: {atomic_type}")
  return lowering._atomic_rmw(op, ptr, val, mask=mask)


def atomic_xchg(x_ref_or_view, idx, val, *, mask: Any | None = None):
  """Atomically exchanges the given value with the value at the given index.

  Args:
    x_ref_or_view: The ref to operate on.
    idx: The indexer to use.
    mask: TO BE DOCUMENTED.

  Returns:
    The value at the given index prior to the aupdate.
  """
  return _atomic_rmw(
      x_ref_or_view, idx, val, mask=mask, atomic_type=AtomicOpType.XCHG
  )


def atomic_add(x_ref_or_view, idx, val, *, mask: Any | None = None):
  """Atomically computes ``x_ref_or_view[idx] += val``.

  Args:
    x_ref_or_view: The ref to operate on.
    idx: The indexer to use.
    mask: TO BE DOCUMENTED.

  Returns:
    The value at the given index prior to the atomic operation.
  """
  return _atomic_rmw(
      x_ref_or_view, idx, val, mask=mask, atomic_type=AtomicOpType.ADD
  )


def atomic_max(x_ref_or_view, idx, val, *, mask: Any | None = None):
  """Atomically computes ``x_ref_or_view[idx] = max(x_ref_or_view[idx], val)``.

  Args:
    x_ref_or_view: The ref to operate on.
    idx: The indexer to use.
    mask: TO BE DOCUMENTED.

  Returns:
    The value at the given index prior to the atomic operation.
  """
  return _atomic_rmw(
      x_ref_or_view, idx, val, mask=mask, atomic_type=AtomicOpType.MAX
  )


def atomic_min(x_ref_or_view, idx, val, *, mask: Any | None = None):
  """Atomically computes ``x_ref_or_view[idx] = min(x_ref_or_view[idx], val)``.

  Args:
    x_ref_or_view: The ref to operate on.
    idx: The indexer to use.
    mask: TO BE DOCUMENTED.

  Returns:
    The value at the given index prior to the atomic operation.
  """
  return _atomic_rmw(
      x_ref_or_view, idx, val, mask=mask, atomic_type=AtomicOpType.MIN
  )


def atomic_and(x_ref_or_view, idx, val, *, mask: Any | None = None):
  """Atomically computes ``x_ref_or_view[idx] &= val``.

  Args:
    x_ref_or_view: The ref to operate on.
    idx: The indexer to use.
    mask: TO BE DOCUMENTED.

  Returns:
    The value at the given index prior to the atomic operation.
  """
  return _atomic_rmw(
      x_ref_or_view, idx, val, mask=mask, atomic_type=AtomicOpType.AND
  )


def atomic_or(x_ref_or_view, idx, val, *, mask: Any | None = None):
  """Atomically computes ``x_ref_or_view[idx] |= val``.

  Args:
    x_ref_or_view: The ref to operate on.
    idx: The indexer to use.
    mask: TO BE DOCUMENTED.

  Returns:
    The value at the given index prior to the atomic operation.
  """
  return _atomic_rmw(
      x_ref_or_view, idx, val, mask=mask, atomic_type=AtomicOpType.OR
  )


def atomic_xor(x_ref_or_view, idx, val, *, mask: Any | None = None):
  """Atomically computes ``x_ref_or_view[idx] ^= val``.

  Args:
    x_ref_or_view: The ref to operate on.
    idx: The indexer to use.
    mask: TO BE DOCUMENTED.

  Returns:
    The value at the given index prior to the atomic operation.
  """
  return _atomic_rmw(
      x_ref_or_view, idx, val, mask=mask, atomic_type=AtomicOpType.XOR
  )


atomic_cas_p = jax_core.Primitive("atomic_cas")


@atomic_cas_p.def_effectful_abstract_eval
def _atomic_cas_abstract_eval(ref_aval, cmp_aval, val_aval):
  if cmp_aval.dtype != val_aval.dtype or cmp_aval.shape != val_aval.shape:
    raise ValueError("cmp and val must have identical dtypes and shapes")
  if ref_aval.shape:
    raise ValueError("ref must be scalar.")
  if cmp_aval.shape:
    raise ValueError("cmp must be scalar.")
  if val_aval.shape:
    raise ValueError("val must be scalar.")
  return jax_core.ShapedArray(val_aval.shape, val_aval.dtype), {
      state.WriteEffect(0)
  }


def atomic_cas(ref, cmp, val):
  """Performs an atomic compare-and-swap of the value in the ref with the

  given value.

  Args:
    ref: The ref to operate on.
    cmp: The expected value to compare against.
    val: The value to swap in.

  Returns:
    The value at the given index prior to the atomic operation.
  """
  return atomic_cas_p.bind(ref, cmp, val)


@state_discharge.register_discharge_rule(atomic_cas_p)
def _atomic_cas_discharge_rule(in_avals, out_avals, ref, cmp, val):
  del in_avals, out_avals
  new_val = jnp.where(ref == cmp, val, ref)
  return (new_val, None, None), ref


@lowering.register_lowering(atomic_cas_p)
def _atomic_cas_lowering_rule(ctx: lowering.LoweringRuleContext, ptr, cmp, val):
  _, cmp_aval, val_aval = ctx.avals_in
  if isinstance(ptr.type, ir.RankedTensorType):
    ptr_type = ir.RankedTensorType(ptr.type)
    element_type = tt_dialect.PointerType(ptr_type.element_type)
    result_type = ir.RankedTensorType.get(
        ptr_type.shape, element_type.pointee_type, ptr_type.encoding
    )
  else:
    result_type = tt_dialect.PointerType(ptr.type).pointee_type
  return tt_dialect.atomic_cas(
      result_type,
      ptr,
      lowering._ensure_ir_value(cmp, cmp_aval),
      lowering._ensure_ir_value(val, val_aval),
      sem=tt_dialect.MemSemantic.ACQUIRE_RELEASE,
      scope=tt_dialect.MemSyncScope.GPU,
  )


max_contiguous_p = jax_core.Primitive("max_contiguous")

max_contiguous_p.def_impl(lambda x, **_: x)
mlir.register_lowering(max_contiguous_p, lambda _, x, **__: [x])


def max_contiguous(x, values):
  """A compiler hint that asserts the ``values`` first values of ``x`` are contiguous."""
  if not isinstance(values, (list, tuple)):
    values = (values,)
  return max_contiguous_p.bind(x, values=tuple(values))


@max_contiguous_p.def_abstract_eval
def _max_contiguous_abstract_eval(aval, **_):
  return aval


@lowering.register_lowering(max_contiguous_p)
def _max_contiguous_rule(
    ctx: lowering.LoweringRuleContext, x, values: Sequence[int]
):
  [x_aval] = ctx.avals_in
  assert len(x_aval.shape) == len(values)
  lowering._set_attr(
      x,
      "tt.contiguity",
      ir.DenseIntElementsAttr.get(np.asarray(values, dtype=np.int32)),
  )
  return x
