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
"""Lowering for Pallas TPU SparseCore."""

from typing import Any, NoReturn, cast
from collections.abc import Sequence
import contextlib
import dataclasses
import functools

from jax._src import api_util
from jax._src import core as jax_core
from jax._src import debugging
from jax._src import lax
from jax._src import linear_util as lu
from jax._src import mesh as mesh_lib
from jax._src import numpy as jnp
from jax._src import source_info_util
from jax._src import state
from jax._src import util
from jax._src import tree_util
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import func
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import vector
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives as pallas_primitives
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import lowering as tc_lowering
from jax._src.pallas.mosaic import primitives as tpu_primitives
from jax._src.pallas.mosaic import sc_core
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax.experimental.mosaic.dialects import tpu


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


MemorySpace = tpu_core.MemorySpace


class GlobalAllocations:
  """Hands out global allocations sequentially during lowering."""
  def __init__(self, allocations: dict[pallas_core.MemoryRef, list[ir.Value]]):
    self._allocations = {k: list(v) for k, v in allocations.items()}

  def next_allocation(self, what: state.AbstractRef | pallas_core.TransformedRef) -> Any:
    """Returns the next available allocation for the given shape."""
    what = pallas_core.MemoryRef(what.inner_aval, what.memory_space)
    if what not in self._allocations:
      raise LookupError(f"No allocations are available for {what}.")
    if not self._allocations[what]:
      raise LookupError(f"No more allocations available for {what}.")
    return self._allocations[what].pop()

  @contextlib.contextmanager
  def verify_usage(self):
    """Scope that verifies all allocations are used."""
    try:
      yield
    finally:
      unused = [k for k, v in self._allocations.items() if v]
      if unused:
        raise AssertionError(f"Some allocations unused ({unused}).")


@dataclasses.dataclass
class ScLoweringContext(tc_lowering.LoweringContext):
  """Lowering context for SparseCore."""
  global_allocations: GlobalAllocations

LoweringRuleContext = tc_lowering.LoweringRuleContext

_transform_ref = tc_lowering._transform_ref
_dtype_to_ir_type = tc_lowering._dtype_to_ir_type

# pylint: disable=protected-access


def dynamic_shape_replacement_fn(x):
  return x


def lower_jaxpr_to_module(
    lowering_context: mlir.LoweringRuleContext,
    grid_mapping: pallas_core.GridMapping,
    jaxpr: jax_core.Jaxpr,
    *,
    dimension_semantics: Sequence[tpu_core.DimensionSemantics] | None,
    kernel_type: tpu_core.KernelType,
    mesh: mesh_lib.Mesh | None = None,
    dynamic_shape_replacement_enabled: bool = False,
) -> ir.Module:
  """Lowers a Jaxpr to a Mosaic SparseCore module."""
  if dynamic_shape_replacement_enabled:
    raise NotImplementedError(
        "Dynamic shape replacement is not supported for SparseCore."
    )
  if not grid_mapping.grid:
    index_map_avals, index_map_tree = tree_util.tree_flatten(
        ((jax_core.ShapedArray((), jnp.int32),), {})
    )
    if grid_mapping.num_index_operands:
      raise ValueError(
          "Index operands not supported for SparseCore when grid is empty."
      )
    new_grid = (1,)
    new_block_mappings = []
    for bm in grid_mapping.block_mappings:

      def new_index_map(*args, bm=bm):
        return jax_core.eval_jaxpr(
            # Discard the leading grid index.
            bm.index_map_jaxpr.jaxpr, bm.index_map_jaxpr.consts, *args[1:]
        )

      debug_info = bm.index_map_jaxpr.jaxpr.debug_info
      if debug_info.arg_names is not None:
        debug_info = debug_info._replace(
            arg_names=("idx", *debug_info.arg_names)
        )
      flat_fun, _ = api_util.flatten_fun(
          lu.wrap_init(new_index_map, debug_info=debug_info), index_map_tree
      )
      with pallas_core.tracing_grid_env(new_grid, grid_mapping.vmapped_dims):
        index_map_jaxpr, _, index_map_jaxpr_consts = pe.trace_to_jaxpr_dynamic(
            flat_fun, index_map_avals
        )
      new_block_mappings.append(
          bm.replace(
              index_map_jaxpr=jax_core.ClosedJaxpr(
                  index_map_jaxpr, index_map_jaxpr_consts
              )
          )
      )

    grid_mapping = grid_mapping.replace(
        grid=new_grid,
        index_map_avals=index_map_avals,
        index_map_tree=index_map_tree,
        block_mappings=tuple(new_block_mappings),
    )
    dimension_semantics = ("arbitrary",)

  for bm in grid_mapping.block_mappings:
    for bd in bm.block_shape:
      if not isinstance(bd, pallas_core.Blocked):
        raise NotImplementedError(
            "Unsupported block dimension type: "
            f"{type(bd)} for block shape: {bm.block_shape}"
        )

  backend = lowering_context.module_context.get_backend(optional=True)
  mosaic_grid_mapping = MosaicGridMapping(
      jaxpr, grid_mapping, dimension_semantics, mesh=mesh
  )
  m = ir.Module.create()
  sym_tab = ir.SymbolTable(m.operation)
  func_op = lower_jaxpr_to_func(
      jaxpr,
      name="main",
      kernel_type=kernel_type,
      mosaic_grid_mapping=mosaic_grid_mapping,
      forward_compatible=lowering_context.is_forward_compat(),
      backend=backend,
  )
  m.body.append(func_op)
  sym_tab.insert(func_op)
  func_op.attributes["iteration_bounds"] = ir.DenseI64ArrayAttr.get(
      mosaic_grid_mapping.grid
  )
  func_op.attributes["dimension_semantics"] = (
      mosaic_grid_mapping.get_dimension_semantics()
  )
  window_params = []
  for i, bm in enumerate(grid_mapping.block_mappings):
    func_name = f"transform_{i}"
    mlir_func = tc_lowering.lower_jaxpr_to_transform_func(
        bm.index_map_jaxpr.jaxpr,
        bm.block_aval,
        name=func_name,
        mosaic_grid_mapping=mosaic_grid_mapping,
        kernel_type=kernel_type,
        forward_compatible=lowering_context.is_forward_compat(),
        backend=backend,
    )
    assert mlir_func.verify(), mlir_func
    m.body.append(mlir_func)
    sym_tab.insert(mlir_func)

    block_shape = list(pallas_core._get_block_shape(bm.block_shape))
    block_params = dict(
        window_bounds=ir.DenseI64ArrayAttr.get(block_shape),
        transform_indices=ir.FlatSymbolRefAttr.get(func_name),
    )
    window_params.append(ir.DictAttr.get(block_params))
  func_op.attributes["window_params"] = ir.ArrayAttr.get(window_params)
  return m


@dataclasses.dataclass(init=False)
class MosaicGridMapping(tc_lowering.MosaicGridMapping):
  """Abstracts a grid mapping for Mosaic SparseCore."""

  def __init__(
      self,
      jaxpr: jax_core.Jaxpr,
      grid_mapping: pallas_core.GridMapping,
      dimension_semantics: Sequence[tpu_core.DimensionSemantics] | None,
      mesh: mesh_lib.Mesh | None,
  ):
    for bm in grid_mapping.block_mappings:
      shape = pallas_core._get_block_shape(bm.block_shape)
      if len(shape) > 1 and shape[-1] % 8:
        raise ValueError(
            f"The minormost dimension of a block for {bm.origin} must be a"
            f" multiple of 8, got shape {shape}"
        )
    if any(
        isinstance(var.aval, sc_core.AbstractRef)
        for var in jaxpr.invars[grid_mapping.slice_scratch_ops]
    ):
      # TODO(slebedev): Support tiling annotations for kernel operands.
      raise NotImplementedError(
          "`plsc.MemoryRef`s are not supported as scratch operands to the"
          " kernel. Allocate them in the kernel body via `pl.run_scoped`."
      )
    super().__init__(
        jaxpr,
        grid_mapping,
        dimension_semantics,
        mesh,
        dynamic_shape_replacement_fn=dynamic_shape_replacement_fn,
    )


def lower_jaxpr_to_func(
    jaxpr: jax_core.Jaxpr,
    *,
    name: str,
    kernel_type: tpu_core.KernelType,
    mosaic_grid_mapping: MosaicGridMapping,
    forward_compatible: bool,
    backend: Any | None,
) -> func.FuncOp:
  """Lowers a Jaxpr to a Mosaic SparseCore function."""
  num_grid = len(mosaic_grid_mapping.grid_types)
  num_scalar_prefetch = len(mosaic_grid_mapping.scalar_prefetch_types)
  if num_scalar_prefetch:
    raise NotImplementedError("Scalar prefetch not supported.")
  num_scratch = len(mosaic_grid_mapping.scratch_types)
  arg_types = [
      *mosaic_grid_mapping.grid_types,
      *mosaic_grid_mapping.scalar_prefetch_types,
      *mosaic_grid_mapping.operand_types,
      *mosaic_grid_mapping.scratch_types,
  ]
  arg_block_shapes = [
      *mosaic_grid_mapping.scalar_prefetch_block_shapes,
      *mosaic_grid_mapping.operand_block_shapes,
      *mosaic_grid_mapping.scratch_block_shapes,
  ]

  def body_func(*args: ir.Value):
    grid_indices, scalar_prefetch, operands_and_scratch = util.split_list(
        args, [num_grid, num_scalar_prefetch]
    )
    grid_indices = mosaic_grid_mapping.get_grid_indices(
        grid_indices, maybe_include_mapped_dims=False
    )
    jaxpr_indices = tuple(
        idx
        for i, idx in enumerate(grid_indices)
        if i not in mosaic_grid_mapping.vmapped_dims
    )

    allocations = sc_core.gather_global_allocations(jaxpr)
    flat_allocations, allocations_tree = tree_util.tree_flatten(allocations)
    allocation_operands = operands_and_scratch[
        len(operands_and_scratch) - len(flat_allocations):]
    allocations = allocations_tree.unflatten(allocation_operands)
    lowering_context = ScLoweringContext(
        mosaic_grid_mapping.grid,  # type: ignore
        mosaic_grid_mapping.grid_names,
        mosaic_grid_mapping.vmapped_dims,
        jaxpr_indices,
        arg_block_shapes,
        source_info_util.NameStack(),
        mesh_context=mosaic_grid_mapping.mesh_info,
        traceback_caches=mlir.TracebackCaches(),
        kernel_type=kernel_type,
        forward_compatible=forward_compatible,
        backend=backend,
        dynamic_shape_replacement_fn=dynamic_shape_replacement_fn,
        global_allocations=GlobalAllocations(allocations),
    )
    with lowering_context.global_allocations.verify_usage():
      return tc_lowering.jaxpr_subcomp(
          lowering_context, jaxpr, *scalar_prefetch, *operands_and_scratch
      )

  body = func.FuncOp.from_py_func(*arg_types, name=name)(body_func)
  func_op = cast(func.FuncOp, body.func_op)
  func_op.attributes["tpu.core_type"] = ir.Attribute.parse(
      f"#tpu.core_type<{kernel_type.name.lower()}>"
  )
  func_op.attributes["scratch_operands"] = ir.IntegerAttr.get(
      ir.IntegerType.get_signless(64), num_scratch
  )
  arg_attrs = [ir.DictAttr.get({})] * num_grid
  for arg, bm in zip(
      func_op.arguments[num_grid : len(func_op.arguments) - num_scratch],
      mosaic_grid_mapping.block_mappings,
  ):
    d = {}
    if str(arg.type.memory_space) == "#tpu.memory_space<hbm>":
      d["sc.persistent"] = ir.UnitAttr.get()
    if isinstance(bm, sc_core.BlockMapping) and bm.indexed_by is not None:
      d["sc.indexed_by"] = mlir.i32_attr(bm.indexed_by)
      d["sc.indexed_dim"] = mlir.i32_attr(bm.indexed_dim)
    arg_attrs.append(ir.DictAttr.get(d))
  arg_attrs.extend([ir.DictAttr.get({})] * num_scratch)

  func_op.arg_attrs = ir.ArrayAttr.get(arg_attrs)
  try:
    func_op.verify()
  except Exception as e:
    raise ValueError(
        f"Body failed to verify: {func_op}.\nThis is an internal error."
        " Please report a bug at:"
        " https://github.com/jax-ml/jax/issues/new?assignees=sharadmv."
    ) from e
  return func_op


register_lowering_rule = functools.partial(
    tc_lowering.register_lowering_rule,
    kernel_types=(
        tpu_core.KernelType.SC_SCALAR_SUBCORE,
        tpu_core.KernelType.SC_VECTOR_SUBCORE,
    ),
)

@register_lowering_rule(pallas_primitives.get_global_p)
def _lower_get_global(ctx: LoweringRuleContext, *, what):
  lctx = ctx.lowering_context
  assert isinstance(lctx, ScLoweringContext)
  return lctx.global_allocations.next_allocation(what)


@register_lowering_rule(state_primitives.get_p)
def _get_lowering_rule(ctx: LoweringRuleContext, ref, *flat_transforms, tree):
  return _load_lowering_rule(ctx, ref, None, *flat_transforms, tree=tree)


def _load_lowering_rule(
    ctx: LoweringRuleContext, ref, mask, *flat_transforms, tree
):
  ref_aval, *_flat_index_avals = ctx.avals_in
  assert isinstance(ref_aval, state.AbstractRef)
  [out_aval] = ctx.avals_out
  assert isinstance(out_aval, jax_core.ShapedArray)

  if (
      (ref_memory_space := ref_aval.memory_space) is MemorySpace.HBM or
      ref_memory_space is MemorySpace.VMEM_SHARED
  ):
    raise NotImplementedError(
        f"Get does not support loading from {ref_memory_space.name}."
        " Copy the data to a core-local memory space, e.g. VMEM,"
        " via `pltpu.async_copy`."
    )

  transforms = list(tree_util.tree_unflatten(tree, flat_transforms))
  if not transforms or not isinstance(transforms[-1], indexing.NDIndexer):
    ref_shape = state.get_transforms_shape(transforms, ref_aval.shape)
    transforms.append(indexing.NDIndexer.make_trivial_indexer(ref_shape))
  *prev_transforms, indexer = transforms
  ref_block_shape, *_ = ctx.block_shapes
  ref, ref_block_shape = _transform_ref(
      ref, ref_aval.dtype, ref_block_shape, prev_transforms
  )
  starts, sizes, strides, _, _ = tc_lowering._indexer_to_start_size_stride(
      indexer, ref_block_shape, cast_to_index=True
  )
  del sizes  # Currently unused.
  if not all(s == 1 for s in strides):
    raise NotImplementedError(
        "Get only supports slices with stride 1, got {strides}"
    )

  if not out_aval.ndim:
    if mask is not None:
      raise NotImplementedError("Get does not support masked scalar loads")
    return memref.load(ref, starts)

  if ref_memory_space is MemorySpace.SMEM:
    raise NotImplementedError("Get can only load scalars from SMEM")
  else:
    _check_aval_is_supported("Get", out_aval)

  vec_type = ir.VectorType.get(
      out_aval.shape, _dtype_to_ir_type(ref_aval.dtype)
  )
  return tpu.vector_load(vec_type, ref, indices=starts, strides=[], mask=mask)


@register_lowering_rule(state_primitives.swap_p)
def _swap_lowering_rule(
    ctx: LoweringRuleContext, ref, val, *flat_transforms, tree
):
  return _store_lowering_rule(
      ctx, ref, val, None, *flat_transforms, tree=tree, add=False
  )


def _store_lowering_rule(
    ctx: LoweringRuleContext, ref, val, mask, *flat_transforms, tree, add
):
  ref_aval, _, *_flat_index_avals = ctx.avals_in
  assert isinstance(ref_aval, state.AbstractRef)
  [out_aval] = ctx.avals_out
  assert isinstance(out_aval, jax_core.ShapedArray)

  if (
      (ref_memory_space := ref_aval.memory_space) is MemorySpace.HBM or
      ref_memory_space is MemorySpace.VMEM_SHARED
  ):
    raise NotImplementedError(
        f"Swap does not support storing to {ref_memory_space.name}."
        " Copy the data to a core-local memory space, e.g. VMEM,"
        " via `pltpu.async_copy`."
    )

  transforms = list(tree_util.tree_unflatten(tree, flat_transforms))
  if not transforms or not isinstance(transforms[-1], indexing.NDIndexer):
    ref_shape = state.get_transforms_shape(transforms, ref_aval.shape)
    transforms.append(indexing.NDIndexer.make_trivial_indexer(ref_shape))
  *prev_transforms, indexer = transforms
  ref_block_shape, *_ = ctx.block_shapes
  ref, ref_block_shape = _transform_ref(
      ref, ref_aval.dtype, ref_block_shape, prev_transforms
  )
  starts, sizes, strides, _, _ = tc_lowering._indexer_to_start_size_stride(
      indexer, ref_block_shape, cast_to_index=True
  )
  del sizes  # Currently unused.
  if not all(s == 1 for s in strides):
    raise NotImplementedError(
        "Swap only supports slices with stride 1, got {strides}"
    )

  if not out_aval.ndim:
    if mask is not None:
      raise NotImplementedError("Swap does not support masked scalar stores")
    if add:
      # TODO(slebedev): We can use memref.atomic_rmw here, but the SC compiler
      # doesn't support it yet.
      raise NotImplementedError("Swap does not support atomic scalar adds")
    old_val = memref.load(ref, starts)
    memref.store(val, ref, starts)
    return old_val

  if ref_memory_space is MemorySpace.SMEM:
    raise NotImplementedError("Swap can only store scalars to SMEM")
  else:
    _check_aval_is_supported("Swap", out_aval)

  vec_type = ir.VectorType.get(
      out_aval.shape, _dtype_to_ir_type(ref_aval.dtype)
  )
  old_val = tpu.vector_load(vec_type, ref, starts, strides=[], mask=mask)
  tpu.vector_store(val, ref, starts, strides=[], mask=mask, add=add)
  return old_val


@register_lowering_rule(lax.iota_p,
                        kernel_types=[tpu_core.KernelType.SC_VECTOR_SUBCORE])
def _iota_lowering_rule_sc(ctx: LoweringRuleContext, dtype, shape, dimension,
                           sharding):
  sc_info = sc_core.get_sparse_core_info()
  if shape != (sc_info.num_lanes,):
    raise ValueError(
        f"Unsupported iota shape for SC vector subcore. Got {shape}, supported "
        f"shape is {(sc_info.num_lanes,)}."
    )
  [out_aval] = ctx.avals_out
  out_type = ir.VectorType.get(
      [sc_info.num_lanes], _dtype_to_ir_type(out_aval.dtype)
  )
  return tpu.iota(out_type, dimensions=[dimension])


def _check_aval_is_supported(caller: str, aval: jax_core.ShapedArray) -> None:
  if aval.shape in sc_core.SUPPORTED_VECTOR_SHAPES.get(aval.dtype, []):
    return
  supported_shapes = ", ".join(
      map(repr, sc_core.SUPPORTED_VECTOR_SHAPES[aval.dtype])
  )
  if not supported_shapes:
    raise NotImplementedError(f"{caller} does not support {aval.dtype} arrays")
  else:
    raise NotImplementedError(
        f"{caller} only supports {aval.dtype} arrays of shapes"
        f" [{supported_shapes}], got {aval.shape}"
    )


@register_lowering_rule(debugging.debug_print_p)
def _debug_print_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    fmt: str,
    ordered,
    partitioned,
    in_tree,
    static_args,
    np_printoptions,
    has_placeholders,
    logging_record,
):
  del partitioned, np_printoptions, in_tree, static_args
  def fail(reason: str) -> NoReturn:
    raise NotImplementedError(
        f"pl.debug_print() {reason} when lowering to SparseCore"
    )

  if ordered:
    fail("does not support ordered print")
  if has_placeholders:
    fail("does not support placeholders")

  match args:
    case []:
      tpu.log(inputs=[], tag=fmt)
    case [arg] if ir.MemRefType.isinstance(arg.type):
      tpu.log_buffer(arg, ctx.avals_in[0].shape, fmt)  # pytype: disable=attribute-error
    case [arg]:
      tpu.log(inputs=[arg], tag=fmt)
    case _:
      fail("does not support multiple inputs")
  return []


def _memref_memory_space(ref: ir.Value) -> MemorySpace:
  match str(ir.MemRefType(ref.type).memory_space):
    case "#tpu.memory_space<hbm>":
      return MemorySpace.HBM
    case "#tpu.memory_space<vmem>":
      return MemorySpace.VMEM
    case "#tpu.memory_space<vmem_shared>":
      return MemorySpace.VMEM_SHARED
    case "#tpu.memory_space<smem>":
      return MemorySpace.SMEM
    case _:
      raise LookupError(f"Unknown memory space: {ref.type}")


def _prepare_dma_refs(
    src_ref,
    src_transforms,
    dst_ref,
    dst_transforms,
    src_aval,
    dst_aval,
    is_add: bool = False,
):
  """Prepares the DMA source and destination references."""
  src_memory_space = _memref_memory_space(src_ref)
  dst_memory_space = _memref_memory_space(dst_ref)
  match src_memory_space, dst_memory_space:
    case MemorySpace.HBM | MemorySpace.VMEM_SHARED, MemorySpace.VMEM:
      if _has_indirect_offsets(dst_transforms):
        raise ValueError(
            "Only the source ref can be indexed when doing a gather via"
            " `pltpu.async_copy`"
        )
      dst_ref, _ = _transform_ref(
          dst_ref, dst_aval.dtype, dst_aval.shape, dst_transforms
      )
      dst_ref_shape = ir.MemRefType(dst_ref.type).shape
      indirect_offsets, src_transforms = _extract_indirect_offsets(
          src_transforms, tuple(dst_ref_shape)
      )
      src_ref, _ = _transform_ref(
          src_ref, src_aval.dtype, src_aval.shape, src_transforms
      )
      indirect_offsets_ref_str = "src_ref"
    case MemorySpace.VMEM, MemorySpace.HBM | MemorySpace.VMEM_SHARED:
      if _has_indirect_offsets(src_transforms):
        raise ValueError(
            "Only the destination ref can be indexed when doing a scatter via"
            " `pltpu.async_copy`"
        )
      src_ref, _ = _transform_ref(
          src_ref, src_aval.dtype, src_aval.shape, src_transforms
      )
      src_ref_shape = ir.MemRefType(src_ref.type).shape
      indirect_offsets, dst_transforms = _extract_indirect_offsets(
          dst_transforms, tuple(src_ref_shape)
      )
      dst_ref, _ = _transform_ref(
          dst_ref, dst_aval.dtype, dst_aval.shape, dst_transforms
      )
      indirect_offsets_ref_str = "dst_ref"
    case _:  # Indirect DMA is not supported.
      if (
          # fmt: off
          _has_indirect_offsets(src_transforms) or
          _has_indirect_offsets(dst_transforms)
          # fmt: on
      ):
        raise NotImplementedError(
            "Scatter/gather via `pltpu.async_copy` from"
            f" {src_memory_space.name} to {dst_memory_space.name} is not"
            " supported"
        )
      if is_add:
        raise ValueError(
            "DMAs with `add=True` are only supported between VMEM and "
            f"HBM/VMEM_SHARED. "
            f"Got (src, dst)={(src_aval.memory_space, dst_aval.memory_space)}"
        )
      src_ref, _ = _transform_ref(
          src_ref, src_aval.dtype, src_aval.shape, src_transforms
      )
      dst_ref, _ = _transform_ref(
          dst_ref, dst_aval.dtype, dst_aval.shape, dst_transforms
      )
      indirect_offsets = None
      indirect_offsets_ref_str = ""
  if is_add and indirect_offsets is None:
    raise NotImplementedError(
        "DMAs with `add=True` must (for now) specify offsets of the"
        " majormost dimension. You can do this by writing"
        " `pltpu.async_copy(..., {ref}={ref}.at[jnp.arange(vec_dim)], ...)`"
        " or `pltpu.async_copy(..., {ref}={ref}.at[indices_ref],"
        " ...)`.".format(ref=indirect_offsets_ref_str)
    )
  return src_ref, dst_ref, indirect_offsets


# TODO(slebedev): Use the TC rule once we align the ``LoweringRuleContext``
# with the TC lowering.
@register_lowering_rule(tpu_primitives.dma_start_p)
def _dma_start_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    tree,
    device_id_type: pallas_primitives.DeviceIdType,
    priority: int,
    add: bool,
):
  (
      src_ref,
      src_transforms,
      dst_ref,
      dst_transforms,
      sem,
      sem_transforms,
      src_sem,
      src_sem_transforms,
      device_id,
  ) = tpu_primitives._dma_unflatten(tree, args)
  src_aval, _, dst_aval, _, sem_aval, _, src_sem_aval, _, _ = (
      tpu_primitives._dma_unflatten(tree, ctx.avals_in)
  )

  src_ref, dst_ref, indirect_offsets = _prepare_dma_refs(
      src_ref, src_transforms, dst_ref, dst_transforms, src_aval, dst_aval, add
  )
  if add and indirect_offsets is None:
    # TODO: Support regular DMA with add=True.
    raise NotImplementedError(
        "DMAs with `add=True` must (for now) specify offsets of the majormost "
        "dimension. You can do this by writing "
        "`pltpu.async_copy(..., dst_ref=ref.at[jnp.arange(vec_dim)], ...)` or "
        "`pltpu.async_copy(..., dst_ref=ref.at[iota_ref], ...)`."
    )
  sem, _ = _transform_ref(sem, sem_aval.dtype, sem_aval.shape, sem_transforms)
  if src_sem is not None:
    src_sem, _ = _transform_ref(
        src_sem, src_sem_aval.dtype, src_sem_aval.shape, src_sem_transforms
    )

  # If not ``None``, we lower to an indirect DMA instead.
  if indirect_offsets is None:
    if device_id is not None:
      device_id, _ = tc_lowering._device_id_to_logical(
          ctx, device_id, device_id_type
      )
    tpu.enqueue_dma(
        src_ref,
        dst_ref,
        sem,
        source_semaphore=src_sem,
        device_id=device_id,
        priority=priority,
    )
    return []

  if device_id is not None:
    raise NotImplementedError(
        "Scatter/gather to or from a remote device via `pltpu.async_copy` is"
        " not supported"
    )
  del priority  # Unused by indirect DMAs.
  tpu.enqueue_indirect_dma(src_ref, dst_ref, indirect_offsets, sem, add=add)
  return []


# TODO(slebedev): Use the TC rule once we align the ``LoweringRuleContext``
# with the TC lowering.
@register_lowering_rule(tpu_primitives.dma_wait_p)
def _dma_wait_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    tree,
    device_id_type: pallas_primitives.DeviceIdType,
):
  (
      src_ref,
      src_transforms,
      dst_ref,
      dst_transforms,
      sem,
      sem_transforms,
      _,
      _,
      device_id,
  ) = tpu_primitives._dma_unflatten(tree, args)
  src_aval, _, dst_aval, _, sem_aval, _, _, _, _ = (
      tpu_primitives._dma_unflatten(tree, ctx.avals_in)
  )

  src_ref, dst_ref, indirect_offsets = _prepare_dma_refs(
      src_ref, src_transforms, dst_ref, dst_transforms, src_aval, dst_aval,
  )
  sem, _ = _transform_ref(sem, sem_aval.dtype, sem_aval.shape, sem_transforms)

  # If not ``None``, we lower to an indirect DMA instead of a regular DMA.
  if indirect_offsets is None:
    if device_id is not None:
      device_id, _ = tc_lowering._device_id_to_logical(
          ctx, device_id, device_id_type
      )
    tpu.wait_dma2(sem, src_ref, dst_ref, device_id=device_id)
    return []

  if device_id is not None:
    raise NotImplementedError(
        "Scatter/gather to or from a remote device via `pltpu.async_copy` is"
        " not supported"
    )
  tpu.wait_indirect_dma(sem, src_ref, dst_ref)
  return []


def _extract_indirect_offsets_from_indexer(
    indexer: indexing.NDIndexer, expected_shape: tuple[int, ...] | None = None
) -> ir.Value | None:
  offsets_ref: Any  # Make mypy happy.
  match indexer.indices:
    case [ir.Value() as offsets, *_] if (
        # fmt: off
        ir.MemRefType.isinstance(offsets.type) or
        ir.VectorType.isinstance(offsets.type)
    ):  # fmt: on
      shape = indexer.get_indexer_shape()
      if expected_shape is not None and shape != expected_shape:
        raise NotImplementedError(
            "The indexer shape in scatter/gather via `pltpu.async_copy` does"
            f" not match the expected shape. Want: {expected_shape}, got:"
            f" {shape}."
        )
    case [state.TransformedRef() as offsets_ref, *_]:
      offsets_type = ir.MemRefType(offsets_ref.ref.type)
      if offsets_type.element_type != ir.IntegerType.get_signless(32):
        raise NotImplementedError(
            "Only int32 indices are supported by scatter/gather via"
            " `pltpu.async_copy` with a dynamically-shaped indexer"
        )
      offsets, _ = _transform_ref(
          offsets_ref.ref,
          jnp.int32,
          offsets_type.shape,  # The shape before the indexing.
          offsets_ref.transforms,
      )
    case _:
      return None

  if ir.MemRefType.isinstance(offsets.type):
    offsets_memory_space = _memref_memory_space(offsets)
    if offsets_memory_space is not MemorySpace.VMEM:
      raise NotImplementedError(
          "Indices for scatter/gather via `pltpu.async_copy` must be in VMEM,"
          f" got {offsets_memory_space.name}"
      )
  if not state_discharge._is_trivial_indexer(
      indexing.NDIndexer(indexer.indices[1:], indexer.shape[1:], ())
  ):
    # TODO(slebedev): Consider lifting this restriction.
    raise NotImplementedError(
        "Only indexing along the major dimension is supported in scatter/gather"
        " via `pltpu.async_copy`"
    )
  return offsets


def _extract_indirect_offsets(
    transforms: Sequence[ir.Value], expected_shape: tuple[int, ...]
) -> tuple[ir.Value | None, Sequence[pallas_core.MemoryRefTransform]]:
  for i, indexer in enumerate(transforms):
    if not isinstance(indexer, indexing.NDIndexer):
      continue
    offsets = _extract_indirect_offsets_from_indexer(indexer, expected_shape)
    if offsets is None:
      continue
    if i != len(transforms) - 1:
      raise NotImplementedError(
          "The indexed ref in scatter/gather via `pltpu.async_copy` cannot have"
          " any transforms following the indexer"
      )
    return offsets, transforms[:i]

  return None, transforms


def _has_indirect_offsets(transforms: Sequence[ir.Value]) -> bool:
  return any(
      _extract_indirect_offsets_from_indexer(indexer) is not None
      for indexer in transforms
      if isinstance(indexer, indexing.NDIndexer)
  )


@register_lowering_rule(pallas_primitives.run_scoped_p)
def _run_scoped_lowering_rule(
    ctx: LoweringRuleContext, *consts, jaxpr, collective_axes
):
  return tc_lowering._run_scoped_lowering_rule(
      ctx,
      *consts,
      jaxpr=jaxpr,
      collective_axes=collective_axes,
      alloc_fn=_alloc_value,
  )


@register_lowering_rule(
    lax.sort_p, kernel_types=[tpu_core.KernelType.SC_VECTOR_SUBCORE]
)
def _sort_lowering_rule(
    ctx: LoweringRuleContext, *xs, dimension, is_stable, num_keys
):
  del is_stable  # Unused, always stable.
  if dimension not in (0, -1):
    raise ValueError(f"Unsupported dimension: {dimension}")
  if num_keys != 1:
    raise NotImplementedError("Multiple sort keys not supported")
  sc_info = sc_core.get_sparse_core_info()
  supported_shape = (sc_info.num_lanes,)
  for i, aval in enumerate(ctx.avals_in):
    if aval.shape != supported_shape:
      raise NotImplementedError(
          f"Unsupported shape for operand {i} of SC sort: Got {aval.shape}, "
          f"expected {supported_shape}"
      )
  keys = xs[0]
  values = xs[1:]
  mask_type = ir.VectorType.get(
      [sc_info.num_lanes], ir.IntegerType.get_signless(1))
  mask = arith.constant(mask_type, ir.DenseElementsAttr.get_splat(
      mask_type, ir.BoolAttr.get(True)))
  if not values:
    _, sorted_keys, _ = tpu.sort(
        mask_type, keys.type, keys.type, keys, keys, mask=mask
    )
    return (sorted_keys,)
  results: list[ir.Value] = []
  for value in values:
    _, sorted_keys, sorted_value = tpu.sort(
        mask_type, keys.type, value.type, keys, value, mask=mask
    )
    if not results:
      results.append(sorted_keys)
    results.append(sorted_value)
  return tuple(results)


@register_lowering_rule(
    lax.gather_p, kernel_types=[tpu_core.KernelType.SC_VECTOR_SUBCORE]
)
def _gather_lowering_rule(
    ctx: LoweringRuleContext,
    x,
    indices,
    *,
    dimension_numbers,
    slice_sizes,
    unique_indices,
    indices_are_sorted,
    mode,
    fill_value,
):

  in_aval, indices_aval = ctx.avals_in
  out_aval, = ctx.avals_out

  if len(in_aval.shape) != 1:
    raise NotImplementedError("Only 1D gather is supported")
  if in_aval.shape != indices_aval.shape[:-1] != out_aval.shape:
    raise ValueError(
        "Shape mismatch in input, indices and output:"
        f" {in_aval.shape}, {indices_aval.shape[:-1]}, {out_aval.shape}"
    )

  # During lowering jnp.take_along_axis to lax.gather, we append extra dimension
  # to the end of the indices array. We should reshape it back to the original
  # shape before lowering to Mosaic and rely on MLIR canonicalization to remove
  # the reshapes.
  assert indices_aval.shape == in_aval.shape + (1,)
  recovered_indices = vector.shape_cast(
      ir.VectorType.get(in_aval.shape, indices.type.element_type),
      indices,
  )
  # Note: current support for lax.gather is still very limited.
  del fill_value
  if slice_sizes == (1,) and mode == lax.GatherScatterMode.PROMISE_IN_BOUNDS:
    if dimension_numbers == lax.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
        operand_batching_dims=(),
        start_indices_batching_dims=(),
    ):
      return tpu.dynamic_gather(x, recovered_indices, [0])
  raise NotImplementedError("Unsupported gather")


@register_lowering_rule(
    lax.rev_p, kernel_types=[tpu_core.KernelType.SC_VECTOR_SUBCORE]
)
def _rev_lowering_rule(ctx: LoweringRuleContext, x, dimensions):
  del ctx  # Unused.
  if dimensions != (0,):
    raise NotImplementedError(f"Invalid dimensions for SC lax.rev: {dimensions}")
  i32 = ir.IntegerType.get_signless(32)
  vec_dim = sc_core.get_sparse_core_info().num_lanes
  cdim = arith.constant(i32, ir.IntegerAttr.get(i32, vec_dim - 1))
  cdim_vec = vector.broadcast(ir.VectorType.get((vec_dim,), cdim.type), cdim)
  return tpu.dynamic_gather(
      x,
      arith.subi(cdim_vec, tpu.iota(cdim_vec.type, dimensions=[0])),
      dimensions=[0],
  )


def _default_tile_strides(
    tiling: sc_core.Tiling, shape: Sequence[int]
) -> Sequence[int]:
  """Returns default tile strides for a given shape and tiling."""
  assert tiling

  cdiv = lambda a, b: (a + b - 1) // b

  strides = [0] * len(shape)
  stride = 1
  first_tile, *_ = tiling
  for d in reversed(range(len(shape))):
    assert shape[d] != ir.ShapedType.get_dynamic_size()
    strides[d] = stride
    if d >= len(shape) - len(first_tile):
      tile_d = d - (len(shape) - len(first_tile))
      stride *= cdiv(shape[d], first_tile[tile_d])
    else:
      stride *= shape[d]
  return strides


def _alloc_value(
    aval: jax_core.AbstractValue, *, ctx: LoweringRuleContext
) -> ir.Value:
  if isinstance(aval, sc_core.AbstractRef) and aval.tiling is not None:
    tiling = "".join(f"({','.join(map(str, tile))})" for tile in aval.tiling)
    strides = _default_tile_strides(aval.tiling, aval.shape)
    out_type = ir.MemRefType.get(
        aval.shape,
        _dtype_to_ir_type(aval.dtype, is_kernel_boundary=True),
        layout=ir.Attribute.parse(f"#tpu.tiled<{tiling},{strides}>"),
        memory_space=tc_lowering._memory_space_to_mosaic_attribute(
            aval.memory_space or MemorySpace.VMEM
        ),
    )
    return memref.alloca(out_type, [], [])
  return tc_lowering._alloc_value(aval, ctx=ctx)
