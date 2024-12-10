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

"""Lowering rules and pass for the MLIR Mosaic GPU dialect."""

from collections.abc import Callable
import functools
import operator
from typing import Any, List, Sequence, Tuple, Type, cast

from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib import mosaic_gpu_dialect as mgpu
from jax._src.util import safe_zip
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import nvvm

from .utils import c, ptr_as_memref, single_thread_predicate

# mypy: ignore-errors


def strided_fragmented_layout():
  layout = mgpu.FragmentedLayout.WGStridedFragLayout
  return ir.Attribute.parse(f"#mosaic_gpu.fragmented_layout<{layout}>")


def splat_fragmented_layout():
  layout = mgpu.FragmentedLayout.WGSplatFragLayout
  return ir.Attribute.parse(f"#mosaic_gpu.fragmented_layout<{layout}>")


def set_layout_attributes(
    op: ir.OpView,
    in_layouts: List[ir.Attribute],
    out_layouts: List[ir.Attribute],
):
  op.attributes["in_layouts"] = ir.ArrayAttr.get(in_layouts)
  op.attributes["out_layouts"] = ir.ArrayAttr.get(out_layouts)


def _extract_any_layout_from_op(op: ir.OpView) -> ir.Attribute | None:
  if "in_layouts" in op.attributes and len(op.operands) > 0:
    return cast(ir.ArrayAttr, op.attributes["in_layouts"])[0]
  elif "out_layouts" in op.attributes and len(op.results) > 0:
    return cast(ir.ArrayAttr, op.attributes["out_layouts"])[0]

  return None


def _infer_pointwise_op_layouts(
    op: ir.OpView,
) -> Tuple[List[ir.Attribute], List[ir.Attribute]] | None:
  layout = _extract_any_layout_from_op(op)
  # The op had no layout set. Since we're annotating ops, we may need to
  # derive layout information from user or producer ops.
  if not layout:
    # First, we iterate on users.
    for op_result in op.results:
      for op_user in cast(ir.OpResult, op_result).uses:
        layout = _extract_any_layout_from_op(op_user.owner)
        if layout:
          break
      else:
        continue
      break

  if not layout:
    # Still no layout set. We iterate on producers.
    for operand in op.operands:
      layout = _extract_any_layout_from_op(operand.owner)
      if layout:
        break

  if not layout:
    return None

  return ([layout for _ in op.operands], [layout for _ in op.results])


def _check_pointwise_op_layouts_match_inferred_layouts(
    op: ir.OpView,
    in_layouts: List[ir.Attribute],
    out_layouts: List[ir.Attribute],
):
  """Checks whether the layouts associated with the operation match the inferred layouts.

  Returns:
    'false' if all the layouts are set on the input operation, and 'true' if
    any of the layouts was not set.
  Raises:
    ValueError: if the layouts of the operation do not match the inferred ones.
  """
  # TODO(bchetioui): abstract prototype and move comment there.

  # TODO(bchetioui): allow checking for only partially set in_layouts or
  # out_layouts? I imagine that may come in handy.
  has_in_layouts = "in_layouts" in op.attributes
  has_out_layouts = "out_layouts" in op.attributes

  if has_in_layouts:
    op_in_layouts = op.attributes["in_layouts"]
    for i, (op_in_layout, in_layout) in enumerate(
        safe_zip(op_in_layouts, in_layouts)
    ):
      if op_in_layout != in_layout:
        raise ValueError(
            f"Inferred in_layout {in_layout} for operand {i} of operation {op},"
            f" but operand has layout {op_in_layout}."
        )

  if has_out_layouts:
    op_out_layouts = op.attributes["out_layouts"]
    for i, (op_out_layout, out_layout) in enumerate(
        safe_zip(op_out_layouts, out_layouts)
    ):
      if op_out_layout != out_layout:
        raise ValueError(
            f"Inferred out_layout {out_layout} for result {i} of operation"
            f" {op}, but result has layout {op_out_layout}."
        )

  return not (has_in_layouts and has_out_layouts)


def trivially_replicates_layout(op: ir.OpView) -> bool:
  """Returns 'true' if the parameter operation carries over a layout."""
  query = lambda cls: isinstance(op, cls)
  # TODO(bchetioui): complete this list with every possible pointwise ops.
  if query(arith.AddFOp) or query(arith.MulFOp) or query(llvm.UndefOp):
    return True
  return False


def infer_layout(module: ir.Module):
  progress = True

  def inference_step(op: ir.Operation):
    if trivially_replicates_layout(op):
      inference_rule = _infer_pointwise_op_layouts
      check_rule = _check_pointwise_op_layouts_match_inferred_layouts
    else:
      raise NotImplementedError(f"Can not infer layout for {op}")

    maybe_layouts = inference_rule(op)
    if maybe_layouts is None:
      return False

    inferred_operands_layout, inferred_results_layout = maybe_layouts

    nonlocal progress
    progress = progress or check_rule(
        op, inferred_operands_layout, inferred_results_layout
    )
    set_layout_attributes(op, inferred_operands_layout, inferred_results_layout)

  def run_layout_inference(op: ir.OpView):
    for region in op.operation.regions:
      for block in region:
        for block_op in list(block):
          inference_step(block_op)
    inference_step(op)

  while progress:
    progress = False
    for op in module.body:
      run_layout_inference(op)


MlirLoweringRule = Callable[[ir.Operation | ir.OpView], Sequence[ir.Value]]


_lowerings: dict[str, MlirLoweringRule] = {}


# TODO(bchetioui): Remove this when minimum jaxlib version >= 0.4.36.
# Jaxlib doesn't contain Mosaic GPU dialect bindings.
InitializeBarrierOp = mgpu.InitializeBarrierOp if mgpu is not None else None

def _register_lowering(
    op: str | Type[ir.OpView]
) -> Callable[[MlirLoweringRule], MlirLoweringRule]:
  def wrapper(f):
    op_name = op if isinstance(op, str) else op.OPERATION_NAME  # pytype: disable=attribute-error
    _lowerings[op_name] = f
    return f

  return wrapper


def _lowered_barrier_type() -> ir.Type:
  return ir.IntegerType.get_signless(64)


def gpu_address_space_to_nvptx(address_space: gpu.AddressSpace) -> int:
  match address_space:
    case gpu.AddressSpace.Global:
      return 1
    case gpu.AddressSpace.Workgroup:
      return 3
    case _:
      raise NotImplementedError(f"address_space not supported: {address_space}")


@_register_lowering(InitializeBarrierOp)
def _initialize_barrier_op_lowering_rule(
    initialize_barrier_op: InitializeBarrierOp) -> Sequence[ir.Value]:

  shape = initialize_barrier_op.barriers_ref.type.shape
  num_barriers = functools.reduce(operator.mul, shape, 1)

  i32 = ir.IntegerType.get_signless(32)
  workgroup_nvptx_address_space = gpu_address_space_to_nvptx(
      gpu.AddressSpace.Workgroup)
  ptr_ty = ir.Type.parse(f"!llvm.ptr<{workgroup_nvptx_address_space}>")

  lowered_barrier_type = _lowered_barrier_type()

  predicate = single_thread_predicate(per_block=True)
  for i in range(num_barriers):
    nvvm.mbarrier_init_shared(
        llvm.getelementptr(ptr_ty, initialize_barrier_op.base_pointer, [], [i],
                           lowered_barrier_type),
        c(initialize_barrier_op.arrival_count.value, i32),
        predicate=predicate
    )

  barrier_base_ptr = llvm.getelementptr(
      ir.Type.parse("!llvm.ptr"),
      initialize_barrier_op.base_pointer, [], [0], lowered_barrier_type)

  return ptr_as_memref(
      barrier_base_ptr, initialize_barrier_op.barriers_ref.type),


def lower_mgpu_dialect(module: ir.Module):
  module.context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  module.context.load_all_available_dialects()

  lowered_operations: set[ir.Operation | ir.OpView] = set()

  def _lower_op(op: ir.OpView):
    if op.name not in _lowerings:
      return
    lowering_rule = _lowerings[op.name]
    new_results = lowering_rule(op)
    for old, new in zip(op.results, new_results):
      old.replace_all_uses_with(new)
    lowered_operations.add(op)

  def _traverse_and_lower_op(op: ir.OpView):
    for region in op.operation.regions:
      for block in region:
        for block_op in list(block):
          with ir.InsertionPoint(block_op):
            _traverse_and_lower_op(block_op)
    _lower_op(op)

  with ir.InsertionPoint(module.body):
    for op in module.body:
      _traverse_and_lower_op(op)

  for lowered_op in lowered_operations:
    lowered_op.erase()
