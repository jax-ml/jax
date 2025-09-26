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
"""Pallas primitives for SparseCore."""

from collections.abc import Callable, Sequence
import enum
import functools
from typing import TypeAlias, TypeVar, overload

import jax
from jax import api_util
from jax import lax
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import effects
from jax._src import linear_util as lu
from jax._src.interpreters import partial_eval as pe
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import lowering as tc_lowering
from jax._src.pallas.mosaic import sc_lowering
from jax._src.state import primitives as state_primitives
from jax._src.state import types as state_types
from jax.experimental.mosaic.dialects import tpu
import jax.numpy as jnp


_ensure_ir_value = tc_lowering._ensure_mlir_value
aval_to_ir_type = functools.partial(
    tc_lowering.aval_to_ir_type, sc_lowering.dynamic_shape_replacement_fn
)

TransformedRef: TypeAlias = state_types.TransformedRef
Ref: TypeAlias = state_types.AbstractRef | TransformedRef

_T = TypeVar("_T")

load_p = jax_core.Primitive("load")
load_p.is_effectful = lambda params: True  # type: ignore


@load_p.def_effectful_abstract_eval
def _load_abstract_eval(ref, *args, has_mask, tree):
  flat_transforms = args[:-1] if has_mask else args
  tref = state_types.TransformedRef(
      ref, jax.tree.unflatten(tree, flat_transforms))
  if has_mask:
    mask = args[-1]
    if mask.dtype != jnp.bool:
      raise TypeError(f"Mask must be a boolean array, got {mask.dtype}")
    if mask.shape != tref.shape:
      raise ValueError(f"Mask must have shape {tref.shape}, got {mask.shape}")
  return (
      jax_core.ShapedArray(tref.shape, ref.dtype), {state_types.ReadEffect(0)})


@sc_lowering.register_lowering_rule(load_p)
def _load_lowering_rule(
    ctx: sc_lowering.LoweringRuleContext, ref, *args, has_mask, tree
):
  if has_mask:
    *flat_transforms, mask = args
  else:
    flat_transforms, mask = list(args), None
  return sc_lowering._load_lowering_rule(
      ctx, ref, mask, *flat_transforms, tree=tree
  )


def load_expanded(ref: Ref, *, mask: jax.Array) -> jax.Array:
  """Performs and expanded masked load from a ref.

  Elements from ``ref`` are placed into positions where ``mask`` is ``True``.
  The elements are taken from ``ref`` sequentially, meaning that the i-th
  ``True`` value in ``mask`` corresponds to accessing ``ref[i]``. The result is
  expanded into the shape of the ``mask``.

  For example, if the mask is ``[True, False, True, True]``, the result is
  ```[ref[0], <?>,  ref[2], ref[3]]``, where ``<?>`` is an undefined value.

  Args:
    ref: The ref to load from.
    mask: A boolean mask specifying which elements to load into.

  Returns:
    The loaded array, with the same shape as the mask. No assumptions can be
    made about the elements at the indices where the mask is ``False``.
  """
  if not isinstance(ref, Ref):
    raise TypeError(f"ref must be an AbstractRef or TransformedRef, got {ref}")
  if not isinstance(ref, TransformedRef):
    ref = ref.at[...]  # type: ignore
  assert isinstance(ref, TransformedRef)
  flat_transforms, tree = jax.tree.flatten(ref.transforms)
  return load_p.bind(ref.ref, *flat_transforms, mask, has_mask=True, tree=tree)


swap_p = jax_core.Primitive("swap")
swap_p.is_effectful = lambda params: True  # type: ignore


@swap_p.def_effectful_abstract_eval
def _swap_abstract_eval(ref, x, *args, has_mask, tree, add):
  flat_transforms = args[:-1] if has_mask else args
  tref = state_types.TransformedRef(
      ref, jax.tree.unflatten(tree, flat_transforms))
  if has_mask:
    mask = args[-1]
    if mask.dtype != jnp.bool:
      raise TypeError(f"Mask must be a boolean array, got {mask.dtype}")
    if mask.shape != tref.shape:
      raise ValueError(f"Mask must have shape {tref.shape}, got {mask.shape}")
  if ref.dtype != x.dtype:
    raise TypeError(
        f"Ref and value must have the same dtype, got {ref.dtype} and {x.dtype}"
    )
  if tref.shape != x.shape:
    raise ValueError(f"Value must have shape {tref.shape}, got {x.shape}")
  effects = {state_types.WriteEffect(0)}
  if add:
    effects.add(state_types.ReadEffect(0))
  return x, effects


@sc_lowering.register_lowering_rule(swap_p)
def _swap_lowering_rule(
    ctx: sc_lowering.LoweringRuleContext, ref, x, *args, has_mask, tree, add
):
  if has_mask:
    *flat_transforms, mask = args
  else:
    flat_transforms, mask = list(args), None
  return sc_lowering._store_lowering_rule(
      ctx, ref, x, mask, *flat_transforms, tree=tree, add=add
  )


def store_compressed(ref: Ref, x: jax.Array, *, mask: jax.Array) -> None:
  """Performs a compressed masked store to a ref.

  Elements from ``x`` where ``mask`` is ``True`` are placed into ``ref``.
  The elements are written to ``ref`` sequentially, meaning the i-th ``True``
  value in ``mask`` corresponds to writing to ``ref[i]``.

  For example, if the mask is ``[True, False, True, True]``, the elements
  ``x[0]``, ``x[2]``, and ``x[3]`` are written to ``ref[0]``, ``ref[1]``, and
  ``ref[2]`` respectively.

  Args:
    ref: The ref to store into.
    x: The array to store. Must have the same shape as ``ref``.
    mask: A boolean mask specifying which elements from ``x`` to store.
  """
  if not isinstance(ref, Ref):
    raise TypeError(f"ref must be an AbstractRef or TransformedRef, got {ref}")
  if not isinstance(ref, TransformedRef):
    ref = ref.at[...]  # type: ignore
  assert isinstance(ref, TransformedRef)
  flat_transforms, tree = jax.tree.flatten(ref.transforms)
  _ = swap_p.bind(
      ref.ref,
      x,
      *flat_transforms,
      mask,
      has_mask=True,
      tree=tree,
      add=False,
  )
  return None


def addupdate(ref: Ref, x: jax.Array) -> None:
  """Performs an atomic add to a ref.

  Args:
    ref: The ref to store into.
    x: The array to store. Must have the same shape as ``ref``.
  """
  if not isinstance(ref, Ref):
    raise TypeError(f"ref must be an AbstractRef or TransformedRef, got {ref}")
  if not isinstance(ref, TransformedRef):
    ref = ref.at[...]  # type: ignore
  assert isinstance(ref, TransformedRef)
  flat_transforms, tree = jax.tree.flatten(ref.transforms)
  _ = swap_p.bind(
      ref.ref, x, *flat_transforms, has_mask=False, tree=tree, add=True
  )
  return None


def addupdate_compressed(ref: Ref, x: jax.Array, *, mask: jax.Array) -> None:
  """Performs a masked atomic add to a ref.

  See ``store_compressed`` for details on how the mask is used.
  """
  if not isinstance(ref, Ref):
    raise TypeError(f"ref must be an AbstractRef or TransformedRef, got {ref}")
  if not isinstance(ref, TransformedRef):
    ref = ref.at[...]  # type: ignore
  assert isinstance(ref, TransformedRef)
  flat_transforms, tree = jax.tree.flatten(ref.transforms)
  _ = swap_p.bind(
      ref.ref, x, *flat_transforms, mask, has_mask=True, tree=tree, add=True
  )
  return None


def _indexed_shape(ref: Ref, indices: Sequence[jax.Array]) -> tuple[int, ...]:
  if len(indices) != ref.ndim:
    raise ValueError(f"The number of indices does not match {ref.ndim=}")
  prev_idx = None
  for idx in indices:
    if idx.ndim != 1:
      raise ValueError(
          f"Indices must be a 1-D array, got an index with shape {idx.shape}"
      )
    if prev_idx is not None and idx.size != prev_idx.size:
      raise ValueError(
          "Indices must have the same size, got {prev_idx.size} and {idx.size}"
      )
    prev_idx = idx
  assert prev_idx is not None
  return (prev_idx.size,)


gather_p = jax_core.Primitive("gather")
gather_p.is_effectful = lambda params: True  # type: ignore


@gather_p.def_effectful_abstract_eval
def _gather_abstract_eval(*flat_args, tree):
  ref, transforms, indices, mask = tree.unflatten(flat_args)
  if transforms:
    ref = state_types.TransformedRef(ref, transforms)
  if ref.dtype not in (jnp.int32, jnp.float32):
    raise TypeError(f"ref.dtype={ref.dtype} must be int32 or float32")
  out_aval = jax_core.ShapedArray(_indexed_shape(ref, indices), ref.dtype)
  sc_lowering._check_aval_is_supported("Gather", out_aval)
  if mask is not None and mask.shape != out_aval.shape:
    raise ValueError(
        f"{mask.shape=} does not match the expected shape {out_aval.shape}"
    )
  return out_aval, {state_types.ReadEffect(0)}


@sc_lowering.register_lowering_rule(gather_p)
def _gather_lowering_rule(
    ctx: sc_lowering.LoweringRuleContext, *flat_args, tree
):
  ref, transforms, indices, mask = tree.unflatten(flat_args)
  ref_aval, *_ = tree.unflatten(ctx.avals_in)
  if ref_aval.memory_space not in (tpu_core.MemorySpace.VMEM, None):
    raise ValueError(
        f"Gather only supports loading from VMEM, got {ref_aval.memory_space}"
    )
  if transforms:
    ref_block_shape, *_ = ctx.block_shapes
    ref, _ = tc_lowering._transform_ref(
        ref, ref_aval.dtype, ref_block_shape, transforms
    )
  [out_aval] = ctx.avals_out
  vec_type = ir.VectorType.get(
      out_aval.shape, sc_lowering._dtype_to_ir_type(ref_aval.dtype)
  )
  return tpu.vector_load_idx(vec_type, ref, indices, mask=mask)


def load_gather(
    ref: Ref, indices: Sequence[jax.Array], *, mask: jax.Array | None = None
) -> jax.Array:
  """Gathers an array from a ref.

  Args:
    ref: The ref in ``VMEM`` to gather from.
    indices: A sequence of 1D arrays, one for each dimension of ``ref``. Each
      array specifies an index for that dimension. All arrays must have the same
      size.
    mask: An optional boolean array, which specifies which elements to load. If
      ``None``, all elements are loaded.

  Returns:
    The gathered array.
  """
  ref, transforms = state_primitives.get_ref_and_transforms(
      ref, None, "load_gather"
  )
  flat_args, tree = jax.tree.flatten((ref, transforms, indices, mask))
  return gather_p.bind(*flat_args, tree=tree)


scatter_p = jax_core.Primitive("scatter")
scatter_p.is_effectful = lambda params: True  # type: ignore
scatter_p.multiple_results = True


@scatter_p.def_effectful_abstract_eval
def _scatter_abstract_eval(*flat_args, tree, add):
  ref, transforms, indices, x, mask = jax.tree.unflatten(tree, flat_args)
  if transforms:
    ref = state_types.TransformedRef(ref, transforms)
  if ref.dtype not in (jnp.int32, jnp.float32):
    raise TypeError(f"ref.dtype={ref.dtype} must be int32 or float32")
  expected_shape = _indexed_shape(ref, indices)
  if x.shape != expected_shape:
    raise ValueError(
        f"{x.shape=} does not match expected shape {expected_shape}"
    )
  if x.dtype != ref.dtype:
    raise TypeError(f"val.dtype={x.dtype} != ref.dtype={ref.dtype}")
  if mask is not None and mask.shape != expected_shape:
    raise ValueError(
        f"{mask.shape=} does not match expected shape {expected_shape}"
    )
  effects = {state_types.WriteEffect(0)}
  if add:
    effects.add(state_types.ReadEffect(0))
  return (), effects


@sc_lowering.register_lowering_rule(scatter_p)
def _scatter_lowering_rule(
    ctx: sc_lowering.LoweringRuleContext, *flat_args, tree, add
):
  ref, transforms, indices, x, mask = jax.tree.unflatten(tree, flat_args)
  ref_aval, *_ = tree.unflatten(ctx.avals_in)
  if ref_aval.memory_space not in (tpu_core.MemorySpace.VMEM, None):
    raise ValueError(
        f"Scatter only supports storing to VMEM, got {ref_aval.memory_space}"
    )
  if transforms:
    ref_block_shape, *_ = ctx.block_shapes
    ref, _ = tc_lowering._transform_ref(
        ref, ref_aval.dtype, ref_block_shape, transforms
    )
  tpu.vector_store_idx(x, ref, indices, mask=mask, add=add)
  return ()


def store_scatter(
    ref: Ref,
    indices: Sequence[jax.Array],
    x: jax.Array,
    *,
    mask: jax.Array | None = None,
) -> None:
  """Scatters an array to a ref.

  Args:
    ref: The ref in ``VMEM`` to scatter to.
    indices: A sequence of 1D arrays, one for each dimension of ``ref``. Each
      array specifies an index for that dimension. All arrays must have the same
      size.
    val: The array to store.
    mask: An optional boolean array, which specifies which elements to store. If
      ``None``, all elements are stored.
  """
  if not indices:
    raise ValueError("Indices must not be empty")
  ref, transforms = state_primitives.get_ref_and_transforms(
      ref, None, "store_scatter"
  )
  flat_args, tree = jax.tree.flatten((ref, transforms, indices, x, mask))
  _ = scatter_p.bind(*flat_args, tree=tree, add=False)
  return None


def addupdate_scatter(
    ref: Ref,
    indices: Sequence[jax.Array],
    x: jax.Array,
    *,
    mask: jax.Array | None = None,
) -> None:
  """Scatters an array to a ref atomically adding to existing values."""
  if not indices:
    raise ValueError("Indices must not be empty")
  ref, transforms = state_primitives.get_ref_and_transforms(
      ref, None, "store_scatter"
  )
  flat_args, tree = jax.tree.flatten((ref, transforms, indices, x, mask))
  _ = scatter_p.bind(*flat_args, tree=tree, add=True)


bitcast_p = jax_core.Primitive("bitcast")


@bitcast_p.def_abstract_eval
def _bitcast_abstract_eval(x, dtype):
  old_bitwidth = dtypes.bit_width(x.dtype)
  new_bitwidth = dtypes.bit_width(dtype)
  if old_bitwidth == new_bitwidth:
    return jax_core.ShapedArray(x.shape, dtype)
  if x.ndim == 0:
    raise ValueError(
        "Cannot bitcast a ()-shaped array to a dtype with a different bitwidth:"
        f" {old_bitwidth=} vs {new_bitwidth=}"
    )
  new_last_dim, rem = divmod(x.shape[-1] * old_bitwidth, new_bitwidth)
  if rem:
    raise ValueError(
        f"Cannot bitcast from {x.dtype} ({old_bitwidth} bits) to"
        f" {dtype} ({new_bitwidth} bits), because {x.shape[-1]=} *"
        f" {old_bitwidth} is not divisible by {new_bitwidth}"
    )
  return jax_core.ShapedArray((*x.shape[:-1], new_last_dim), dtype)


@sc_lowering.register_lowering_rule(bitcast_p)
def _bitcast_lowering_rule(ctx: sc_lowering.LoweringRuleContext, x, *, dtype):
  del dtype  # Unused.
  [out_aval] = ctx.avals_out
  return vector.bitcast(aval_to_ir_type(out_aval), x)


def bitcast(x: jax.Array, dtype: jax.typing.DTypeLike) -> jax.Array:
  """Bitcasts an array to a different dtype.

  Unlike ``lax.bitcast_convert_type``, this function returns an array of the
  same rank as the input. The minormost dimension is expanded/shrunk to
  account for the difference in the element bitwidth.
  """
  if x.dtype == dtype:
    return x
  return bitcast_p.bind(x, dtype=jnp.dtype(dtype))


scan_count_p = jax_core.Primitive("unique")
scan_count_p.multiple_results = True


@scan_count_p.def_abstract_eval
def _scan_count_abstract_eval(x, mask):
  if x.dtype != jnp.int32 and x.dtype != jnp.float32:
    raise NotImplementedError(f"x.dtype={x.dtype} must be int32 or float32")
  if not jnp.issubdtype(mask.dtype, jnp.bool):
    raise TypeError(f"mask.dtype={mask.dtype} is not a boolean dtype")
  if x.shape != mask.shape:
    raise ValueError(f"x.shape={x.shape} != mask.shape={mask.shape}")
  return jax_core.ShapedArray(x.shape, jnp.int32), mask


@sc_lowering.register_lowering_rule(scan_count_p)
def _scan_count_lowering_rule(ctx: sc_lowering.LoweringRuleContext, x, mask):
  del ctx  # Unused.
  # Reverse, because the MLIR op returns the mask first.
  return tpu.scan_count(mask, x)[::-1]


def scan_count(
    x: jax.Array, mask: jax.Array | None = None
) -> tuple[jax.Array, jax.Array]:
  """Computes the running duplicate occurrence count of the array.

  Args:
    x: An array of integers or floats.
    mask: An optional array of booleans, which specifies which elements ``x``
      are eligible for counting. If ``None``, all elements are eligible.

  Returns:
    A tuple of two arrays:

      * the running duplicate occurrence count of ``x``;
      * the mask indicating the last occurrence of each duplicate that was
        counted.
  """
  return scan_count_p.bind(x, lax.full(x.shape, True) if mask is None else mask)


masked_cumsum_p = jax_core.Primitive("masked_cumsum")
masked_cumsum_p.multiple_results = False

@masked_cumsum_p.def_abstract_eval
def _masked_cumsum_abstract_eval(x, mask):
  if x.dtype != jnp.int32 and x.dtype != jnp.float32:
    raise NotImplementedError(f"x.dtype={x.dtype} must be int32 or float32")
  if not jnp.issubdtype(mask.dtype, jnp.bool):
    raise TypeError(f"mask.dtype={mask.dtype} is not a boolean dtype")
  if x.shape != mask.shape:
    raise ValueError(f"x.shape={x.shape} != mask.shape={mask.shape}")
  return jax_core.ShapedArray(x.shape, x.dtype)

@sc_lowering.register_lowering_rule(masked_cumsum_p)
def _masked_cumsum_lowering_rule(ctx: sc_lowering.LoweringRuleContext, x, mask):
  del ctx  # Unused.
  return tpu.scan(
      x.type, x, ir.Attribute.parse("#tpu.reduction_kind<sum>"), mask=mask)

@sc_lowering.register_lowering_rule(lax.cumsum_p)
def _lax_cumsum_lowering_rule(ctx: sc_lowering.LoweringRuleContext, x, axis,
                              reverse):
  if axis != 0:
    raise NotImplementedError(f"SC cumsum: axis={axis} must be 0.")
  if len(ctx.avals_in[0].shape) != 1:
    raise NotImplementedError(f"SC cumsum: x={ctx.avals_in[0]} must be rank 1")
  if reverse:
    raise NotImplementedError("SC cumsum: reverse=True is not yet supported")
  i1t = ir.IntegerType.get_signless(1)
  c1 = arith.constant(i1t, ir.IntegerAttr.get(i1t, 1))
  c1v = vector.splat(ir.VectorType.get(x.type.shape, c1.type), c1)
  return tpu.scan(
      x.type, x, ir.Attribute.parse("#tpu.reduction_kind<sum>"), mask=c1v)

def masked_cumsum(x: jax.Array, mask: jax.Array) -> jax.Array:
  """Returns the cumulative sum of the array along its innermost axis.

  This differs from `jnp.cumsum` in that it takes an additional `mask` argument.

  Args:
    x: An array of integers or floats.
    mask: An optional array of booleans, which specifies which elements ``x``
      are eligible for summing. If ``None``, all elements are eligible.
  """
  if x.ndim != 1:
    raise NotImplementedError(f"masked_cumsum: x={x.aval} must be rank 1")
  return masked_cumsum_p.bind(x, mask)


parallel_loop_p = jax_core.Primitive("parallel_loop")
parallel_loop_p.is_effectful = lambda params: bool(params["jaxpr"].effects)  # type: ignore
parallel_loop_p.multiple_results = True


@parallel_loop_p.def_effectful_abstract_eval
def _parallel_loop_abstract_eval(*args, jaxpr, tree, **params):
  del params  # Unused.
  _, _, _, _, carries = tree.unflatten(args)
  if any(isinstance(c, (Ref, TransformedRef)) for c in carries):
    raise TypeError(f"Carried values may not be refs, but got: {carries}")
  updated_effects = set()
  for eff in jaxpr.effects:
    if isinstance(eff, effects.JaxprInputEffect):
      # Offset for the parallel_loop eqn to account for start, stop, and step
      # args passed to parallel_loop_p.bind.
      eff = eff.replace(input_index=eff.input_index + 3)
    updated_effects.add(eff)
  return carries, updated_effects


@sc_lowering.register_lowering_rule(parallel_loop_p)
def _parallel_loop_lowering_rule(
    ctx: sc_lowering.LoweringRuleContext,
    *flat_args,
    tree,
    unroll,
    jaxpr,
):
  lower, upper, step, consts, carry = tree.unflatten(flat_args)
  for_op = scf.ForOp(
      _ensure_ir_value(lower, pallas_core.index_map_grid_aval),
      _ensure_ir_value(upper, pallas_core.index_map_grid_aval),
      _ensure_ir_value(step, pallas_core.index_map_grid_aval),
      carry,
  )
  for_op.attributes["sc.parallel_access"] = ir.UnitAttr.get()
  for_op.attributes["sc.loop_unroll_factor"] = ir.IntegerAttr.get(
      ir.IntegerType.get_signless(64), unroll
  )
  with ir.InsertionPoint(for_op.body):
    _, _, _, consts_block_shapes, *_ = tree.unflatten(ctx.block_shapes)
    lowering_ctx = ctx.lowering_context.replace(
        block_shapes=[*consts_block_shapes, None] + [None] * len(carry),
    )
    carry_out = tc_lowering.jaxpr_subcomp(
        lowering_ctx,
        pe.convert_constvars_jaxpr(jaxpr),
        *consts,
        for_op.induction_variable,
        *for_op.inner_iter_args,
    )
    scf.yield_(carry_out)
  return for_op.results


@overload
def parallel_loop(
    lower: jax.typing.ArrayLike,
    upper: jax.typing.ArrayLike,
    step: jax.typing.ArrayLike = ...,
    *,
    unroll: int = ...,
    carry: None = None,
) -> Callable[[Callable[[jax.Array], None]], None]:
  ...


@overload
def parallel_loop(
    lower: jax.typing.ArrayLike,
    upper: jax.typing.ArrayLike,
    step: jax.typing.ArrayLike = ...,
    *,
    unroll: int = ...,
    carry: _T,
) -> Callable[[Callable[[jax.Array, _T], _T]], _T]:
  ...


def parallel_loop(lower, upper, step=1, *, unroll=1, carry=None):
  """A parallel loop decorator.

  The decorated function forms the loop body. It is called with the current
  loop index as the argument and optionally, a single additional carry argument.

  The loop iterations must be independent, meaning that operations in one
  iteration cannot depend on the side effects, especially Ref writes, of any
  other iteration. This allows the compiler to execute instructions from
  different iterations concurrently, potentially reordering them for better
  performance.

  Cross-iteration dependencies traceable via carried values are allowed. Refs
  may not be carried.

  Safe usage of carried value::

    @parallel_loop(0, 64, step=8, carry=jnp.int32(1))
    def body(i, j):
      # Writes are independent across iterations.
      x_ref[pl.ds(i, 8)] = j + jnp.arange(8)
      return j + 1

  Any pytree can be carried. The final value is returned by the decorator::

    def body(i, my_tree: MyTree):
      # Writes are independent across iterations.
      x_ref[pl.ds(i, 8)] = my_tree.transform(jnp.arange(8))
      return my_tree.step(i)
    final_value = parallel_loop(0, 64, step=8, carry=MyTree())(body)

  Undefined result::

    @parallel_loop(0, 64, step=4, carry=jnp.int32(1))
    def body(i, j):
      # Because the step size is 4, the array written is of size 8, and loop
      # iterations may be reordered, the values in indices 4-59 of x_ref are
      # unspecified after the loop. (The values in 0-3 and 60-63 are only
      # written by the first and last iterations, so are well-defined.)
      x_ref[pl.ds(i, 8)] = j + jnp.arange(8)
      return j + 1

  Unsafe read of "previous" iteration's write (don't do this)::

    @parallel_loop(0, 64, 8, carry=jnp.int32(1))
    def body(i, j):
      # Unsafe because it depends on the side-effect of "previous" iterations,
      # which may be executed in parallel or reordered.
      mask = x_ref[pl.ds(0, 8)] < j
      x_ref[pl.ds(0, 8)] += jnp.where(mask, j + jnp.arange(8), 0)
      return j + 1

  Args:
    lower: The starting value of the loop index.
    upper: The exclusive upper bound of the loop index.
    step: The increment of the loop index. Default to 1.
    unroll: The unroll factor of the loop.
    carry: Optional carried state of the loop.

  Returns:
    A decorator that executes the given function in a parallel loop.
  """

  def decorator(body):
    flat_carries, carry_tree = jax.tree.flatten(carry)
    def wrapped(idx, *carries):
      if carry is None:
        body(idx)
        return []
      return jax.tree.leaves(body(idx, carry_tree.unflatten(carries)))
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
        lu.wrap_init(
            wrapped,
            debug_info=api_util.debug_info("parallel_loop", body, (), {}),
        ),
        [pallas_core.index_map_grid_aval, *(c.aval for c in flat_carries)],
    )
    disallowed_effects = effects.control_flow_allowed_effects.filter_not_in(
        jaxpr.effects
    )
    if disallowed_effects:
      raise NotImplementedError(
          f"Effects not supported in parallel_loop: {disallowed_effects}"
      )
    flat_args, tree = jax.tree.flatten(
        (lower, upper, step, consts, flat_carries)
    )
    flat_result = parallel_loop_p.bind(
        *flat_args, tree=tree, unroll=unroll, jaxpr=jaxpr
    )
    if carry is None:
      return None
    return carry_tree.unflatten(flat_result)

  return decorator


class PackFormat(enum.Enum):
  #: [a0, a1], [b0, b1] -> [[a0, a1], [b0, b1]]
  COMPRESSED = "compressed"
  #: [a0, a1], [b0, b1] -> [a0, b0, a1, b1]
  INTERLEAVED = "interleaved"


def _format_to_ir_attribute(format: PackFormat) -> ir.Attribute:
  return ir.Attribute.parse(f"#tpu.pack_format<{format.value}>")


pack_p = jax_core.Primitive("pack")


@pack_p.def_abstract_eval
def _pack_abstract_eval(a, b, *, format, preferred_element_type):
  if a.shape != b.shape:
    raise ValueError(
        f"Packed arrays must have the same shape, got {a.shape} and {b.shape}"
    )
  if a.ndim != 1:
    raise ValueError(f"Packed arrays must be 1-D, got {a.ndim}")
  if a.dtype != b.dtype:
    raise TypeError(
        f"Packed arrays must have the same dtype, got {a.dtype} and {b.dtype}"
    )
  if preferred_element_type is None:
    match a.dtype:
      case jnp.float32:
        packed_dtype = jnp.bfloat16
      case jnp.int32:
        packed_dtype = jnp.int16
      case _:
        # TODO(slebedev): Support more types.
        raise NotImplementedError(
            f"Only packing of float32 and int32 is supported, got {a.dtype}"
        )
  else:
    packed_bw = dtypes.bit_width(a.dtype) // 2
    if dtypes.bit_width(preferred_element_type) != packed_bw:
      raise ValueError(
          f"preferred_element_type= must have bitwidth {packed_bw}, got"
          f" {dtypes.bit_width(preferred_element_type)}"
      )
    packed_dtype = preferred_element_type

  match format:
    case PackFormat.INTERLEAVED:
      packed_shape = (2 * a.size,)
    case PackFormat.COMPRESSED:
      packed_shape = (a.size, 2)
  return jax_core.ShapedArray(packed_shape, packed_dtype)


@sc_lowering.register_lowering_rule(pack_p)
def _pack_lowering_rule(
    ctx: sc_lowering.LoweringRuleContext,
    a,
    b,
    *,
    format,
    preferred_element_type,
):
  del preferred_element_type  # Unused.
  [out_aval] = ctx.avals_out
  return tpu.pack_subelements(
      aval_to_ir_type(out_aval),
      [a, b],
      [0, 1],
      _format_to_ir_attribute(format),
  )


def pack(
    a: jax.Array,
    b: jax.Array,
    /,
    *,
    format: PackFormat,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Packs two arrays according to the given format.

  .. warning:: This API is temporary and will be removed once the SparseCore
               compiler is able to do packing/unpacking automatically.

  Args:
    a: The first array to pack.
    b: The second array to pack.
    format: The packing format to use.
    preferred_element_type: Optional. The preferred element type of the packed
      array. If specified, must have half the bitwidth of the input array types.

  Returns:
    The packed array.
  """
  if preferred_element_type is not None:
    preferred_element_type = jnp.dtype(preferred_element_type)
  return pack_p.bind(
      a, b, format=format, preferred_element_type=preferred_element_type
  )


unpack_p = jax_core.Primitive("unpack")
unpack_p.multiple_results = True


@unpack_p.def_abstract_eval
def _unpack_abstract_eval(ab, *, format, preferred_element_type):
  match format:
    case PackFormat.INTERLEAVED:
      if ab.ndim != 1 or ab.size % 2 != 0:
        raise ValueError(
            "Interleaved unpack requires a 1-D array with an even size, got"
            f" {ab.shape}"
        )
    case PackFormat.COMPRESSED:
      if ab.ndim != 2 or ab.shape[1] != 2:
        raise ValueError(
            "Compressed unpack requires an array with shape (N, 2), got"
            f" {ab.shape}"
        )
  if preferred_element_type is None:
    match ab.dtype:
      case jnp.bfloat16:
        unpacked_dtype = jnp.float32
      case jnp.int16:
        unpacked_dtype = jnp.int32
      case _:
        # TODO(slebedev): Support more types.
        raise NotImplementedError(
            f"Only unpacking of bloat16 and int16 is supported, got {ab.dtype}"
        )
  else:
    unpacked_bw = dtypes.bit_width(ab.dtype) * 2
    if dtypes.bit_width(preferred_element_type) != unpacked_bw:
      raise ValueError(
          f"preferred_element_type= must have bitwidth {unpacked_bw}, got"
          f" {dtypes.bit_width(preferred_element_type)}"
      )
    unpacked_dtype = preferred_element_type
  return (jax_core.ShapedArray((ab.size // 2,), unpacked_dtype),) * 2


@sc_lowering.register_lowering_rule(unpack_p)
def _unpack_lowering_rule(
    ctx: sc_lowering.LoweringRuleContext, ab, *, format, preferred_element_type
):
  del preferred_element_type  # Unused.
  out_aval, _ = ctx.avals_out
  out_type = aval_to_ir_type(out_aval)
  return (
      tpu.unpack_subelements(out_type, ab, 0, _format_to_ir_attribute(format)),
      tpu.unpack_subelements(out_type, ab, 1, _format_to_ir_attribute(format)),
  )


def unpack(
    ab: jax.Array,
    /,
    *,
    format: PackFormat,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Unpacks two arrays according to the given format.

  .. warning:: This API is temporary and will be removed once the SparseCore
               compiler is able to do packing/unpacking automatically.

  Args:
    ab: The array to unpack.
    format: The packing format to use.
    preferred_element_type: Optional. The preferred element type of the unpacked
      arrays. If specified, must have double the bitwidth of the input array
      type.

  Returns:
    The unpacked arrays.
  """
  if preferred_element_type is not None:
    preferred_element_type = jnp.dtype(preferred_element_type)
  return unpack_p.bind(
      ab,
      format=format,
      preferred_element_type=preferred_element_type,
  )


def _mask_all_reduce_abstract_eval(x, *, reduce):
  if x.dtype != jnp.bool:
    raise TypeError(f"Mask all-reduce only supports bool arrays, got {x.dtype}")
  match x.shape:
    case (minor_dim,):
      return jax_core.ShapedArray((minor_dim // reduce,), jnp.int32)
    case _:
      raise ValueError("Mask all-reduce only supports 1D arrays")


def _mask_all_reduce_lowering_rule(
    ctx: sc_lowering.LoweringRuleContext, x, *, reduce, kind: str
):
  [out_aval] = ctx.avals_out
  return tpu.all_reduce(
      ir.VectorType.get(
          out_aval.shape,
          ir.IntegerType.get_signless(32),
      ),
      x,
      0,
      ir.Attribute.parse(f"#tpu.reduction_kind<{kind}>"),
  )


all_reduce_population_count_p = jax_core.Primitive(
    "all_reduce_population_count"
)
all_reduce_population_count_p.def_abstract_eval(_mask_all_reduce_abstract_eval)
sc_lowering.register_lowering_rule(all_reduce_population_count_p)(
    functools.partial(_mask_all_reduce_lowering_rule, kind="sum")
)


def all_reduce_population_count(x: jax.Array, *, reduce: int = 1) -> jax.Array:
  """Computes the number of nonzero elements in the array.

  Args:
    x: A 1D array of bools.
    reduce: The factor to reduce the output shape by.

  Returns:
    An array with each element containing the number of true elements in ``x``.
  """
  return all_reduce_population_count_p.bind(x, reduce=reduce)


all_reduce_ffs_p = jax_core.Primitive("all_reduce_ffs")
all_reduce_ffs_p.def_abstract_eval(_mask_all_reduce_abstract_eval)
sc_lowering.register_lowering_rule(all_reduce_ffs_p)(
    functools.partial(_mask_all_reduce_lowering_rule, kind="find_first_set")
)


def all_reduce_ffs(x: jax.Array, *, reduce: int = 1) -> jax.Array:
  """Computes the index of the first true element in the array.

  Args:
    x: A 1D array of bools.
    reduce: The factor to reduce the output shape by.

  Returns:
    An array with each element containing the index of the first true element in
    ``x`` or ``x.size`` if there are no true elements.
  """
  return all_reduce_ffs_p.bind(x, reduce=reduce)
