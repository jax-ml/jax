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

"""Layout and transform inference pass for the MLIR Mosaic GPU dialect."""

# mypy has been causing more problems than it solves here. Disable it for these
# files. We have pytype checks anyway.
# mypy: ignore-errors

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import dataclasses
import enum
import itertools
import math
import re
from typing import assert_never, cast

from absl import logging
from jax._src.lib import mosaic_gpu_dialect as mgpu  # noqa: F401
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import math as mlir_math
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
import numpy as np

from . import constraints as cs
from . import fragmented_array as fa
from . import inference_utils
from . import launch_context as lc
from . import layouts as layouts_lib
from . import tcgen05
from . import utils


# This value was arrived at by looking at an existing kernel where layout
# inference would never be able to complete successfully, and kernels where it
# would, as well as existing tests as of 2025-11-03. We observed the following:
#
#   1. all tests would pass with a fuel that is at least ~15_000;
#   2. the kernel for which layout inference fails would fail in less than 12
#      seconds when using a fuel of 100_000.
#
# All in all, this seems like a reasonable compromise: the value is high
# enough that we can comfortably find a solution to even the most complicated
# layout inference problems that we have seen so far, but the runtime is fast
# enough that users will not waste much time waiting for a never-ending pass to
# complete when the system is unable to find a solution.
_DEFAULT_LAYOUT_INFERENCE_FUEL = 100_000


class VariableType(enum.IntEnum):
  """The type of a variable.

  Variables are operands, results, or arguments of MLIR operations.
  """
  OPERAND = 0
  RESULT = 1
  ARGUMENT = 2


class MemorySpace(enum.Enum):
  """The memory space of a variable."""
  REG = enum.auto()
  SMEM = enum.auto()
  TMEM = enum.auto()


_op_name_regex = re.compile(r"^(%\d+ = )?\S+")


@dataclasses.dataclass(frozen=True)
class ValueSite:
  """A unique identifier for a variable.

  This class describes a particular role of a Value, either as a result of an
  operation, an operand of an operation, or a block argument.
  """
  # A MLIR operation. If the type is `ARGUMENT`, this is the owner of the block
  # and region_index is the region that contains the block with the argument.
  # The block is always the first block of the region.
  operation: ir.OpView
  # Whether this represents an operand, a result, or an argument.
  type: VariableType
  # The index of the operand/result/argument within the op's
  # operands/results/arguments.
  index: int
  # The index of the region that contains the block with the argument.
  region_index: int | None = None

  def __post_init__(self):
    assert (self.type != VariableType.ARGUMENT) == (self.region_index is None)

  @property
  def value(self) -> ir.Value:
    """Returns the IR value corresponding to this value site."""
    if self.type == VariableType.OPERAND:
      return self.operation.operands[self.index]
    elif self.type == VariableType.RESULT:
      return self.operation.results[self.index]
    else:
      return self.operation.regions[self.region_index].blocks[0].arguments[self.index]

  @property
  def shape(self) -> tuple[int, ...]:
    """Returns the shape of the underlying value."""
    return tuple(self.value.type.shape)  # pytype: disable=attribute-error

  @property
  def memory_space(self) -> MemorySpace:
    """Returns the memory space associated with this value."""
    ty = self.value.type
    if isinstance(ty, ir.VectorType):
      return MemorySpace.REG
    assert isinstance(ty, ir.MemRefType)
    if utils.is_tmem_ref(ty):
      return MemorySpace.TMEM
    elif utils.is_smem_ref(ty):
      return MemorySpace.SMEM
    raise ValueError(f"Unsupported memory space for: {ty}")

  def __str__(self):
    match = _op_name_regex.match(str(self.operation))
    assert match is not None
    if self.type == VariableType.OPERAND:
      return f"{match.group(0)}:o-{self.index}"
    elif self.type == VariableType.RESULT:
      return f"{match.group(0)}:r-{self.index}"
    else:
      return f"{match.group(0)}:a-{self.index}"


def extract_assignment_candidates_from_reduce_equation(
    small: cs.RegisterLayout,
    large: cs.Variable,
    reduction_dims: tuple[int, ...]
) -> Iterator[cs.RegisterLayout]:
  """Yields layout candidates for the reduce equation `small = reduce(large, reduction_dims)."""
  large_shape = large.key.value.type.shape  # pytype: disable=attribute-error
  candidates = [
      fa.WGMMA_LAYOUT,
      fa.WGMMA_TRANSPOSED_LAYOUT,
      fa.TCGEN05_LAYOUT,
      fa.TCGEN05_TRANSPOSED_LAYOUT,
      tcgen05.TMEM_NATIVE_LAYOUT,
  ]
  if large_shape[-1] % 16 == 0:
    candidates.append(tcgen05.fa_m64_collective_layout(large_shape[-1]))

  for candidate in candidates:
    if len(candidate.base_tile_shape) > len(large_shape):
      continue
    if candidate.reduce(reduction_dims) == small.value:
      yield cs.RegisterLayout(candidate)


def _strided_layout_for_variable(
    variable: cs.Variable,
) -> fa.WGStridedFragLayout | None:
  """Returns a strided layout for the given variable.

  If the given variable cannot have a strided layout, returns `None`.
  """
  # TODO(bchetioui): should we make variables carry a shape as well, to make
  # things easier?
  ty = variable.key.value.type
  assert isinstance(ty, ir.VectorType)
  return fa.WGStridedFragLayout.from_shaped_type(ty)


def _default_tmem_layout_for_variable(
    variable: cs.Variable,
) -> tcgen05.TMEMLayout | None:
  """Returns a default TMEM layout for the given variable, if one is defined."""
  value = variable.key.value
  parent = value.owner
  if isinstance(parent, mgpu.TmemAllocOp):
    return tcgen05._infer_tmem_layout(  # pylint: disable=protected-access
        tuple(value.type.shape), parent.collective, packing=1
    )
  return None


def _extract_tiling_candidate(
    divide_constraint: cs.Divides, num_tiled_dims: int
) -> Iterator[tuple[cs.Variable, cs.Constant]]:
  if not isinstance(divide_constraint.expr, cs.Variable):
    return
  if num_tiled_dims > len(divide_constraint.tiling_multiple):
    # The tiling's rank cannot be larger than the size of `tiling_multiple`.
    return
  tiling = divide_constraint.tiling_multiple[-num_tiled_dims:]
  yield divide_constraint.expr, cs.SMEMTiling(lc.TileTransform(tiling))


def _extract_layout_candidates_from_memory_space_transfer(
    constraint: cs.IsTransferable,
    division_constraint_per_var: dict[cs.Variable, cs.Divides],
) -> Iterator[tuple[cs.Variable, cs.Constant]]:
  """Attempts to extract variable assignments from a `Constraint`."""
  # This code assumes that the `IsTransferable` constraint is bidirectional.
  # This is currently true for TMEM <-> REG transfers and SMEM <-> REG
  # transfers.
  src, tgt = constraint.source, constraint.target
  match src, tgt:
    case cs.Variable(), cs.Constant():
      variable, constant = src, tgt
    case cs.Constant(), cs.Variable():
      variable, constant = tgt, src
    case _:
      return

  assert isinstance(variable, cs.Variable)  # Satisfy type checkers.
  if isinstance(constant, cs.RegisterLayout):
    layout = constant.value
    if variable.key.memory_space == MemorySpace.TMEM:
      dtype = ir.MemRefType(variable.key.value.type).element_type
      for packing in (1, 32 // utils.bitwidth(dtype)):
        for tmem_layout, reg_layout in constraint.supported_tmem_transfers(
            packing
        ):
          if layout == reg_layout:
            yield variable, cs.TMEMLayout(tmem_layout)
    elif variable.key.memory_space == MemorySpace.SMEM:
      if inference_utils.is_mma_layout(layout):
        tiling = _infer_tiling_for_mma_ref(
            variable.key.value.type,
            max_swizzle=mgpu.SwizzlingMode.k128ByteSwizzle
        )
        divide = cs.Divides(variable, tiling)
        if (divide2 := division_constraint_per_var.get(variable)) is not None:
          # This is done on two lines to satisfy type checkers.
          # TODO(b/447079781): clean up the `merge_divides_constraints` to
          # avoid the need for this.
          [merged] = cs.merge_divides_constraints([divide, divide2])
          divide = cast(cs.Divides, merged)
        yield from _extract_tiling_candidate(divide, len(tiling))
      else:
        # An empty tiling is valid here but we don't yield it in order to
        # avoid duplicating the empty tiling yielded by the caller.
        return

  if isinstance(constant, cs.TMEMLayout):
    layout = constant.value
    packing = layout.vector_length
    for tmem_layout, reg_layout in constraint.supported_tmem_transfers(packing):
      if layout == tmem_layout:
        yield variable, cs.RegisterLayout(reg_layout)


def _extract_layout_candidates_from_mma_tiling(
    mma_tiling: cs.IsValidMmaTiling,
) -> Iterator[tuple[cs.Variable, cs.Constant]]:
  v: cs.Variable
  match mma_tiling.expr:
    case cs.Variable() as var:
      is_transposed = False
      v = var
    case cs.Transpose(cs.Variable() as var):
      assert isinstance(var, cs.Variable)
      is_transposed = True
      v = var
    case _:
      return

  tiled_dimensions = v.key.shape[-2:]
  for swizzle in (128, 64, 32):
    swizzle_elems = swizzle * 8 // mma_tiling.bitwidth
    tiling = (swizzle_elems, 8) if is_transposed else (8, swizzle_elems)
    if any(s % t for s, t in zip(tiled_dimensions, tiling)):
      continue
    yield v, cs.SMEMTiling(lc.TileTransform(tiling))


def _divides_per_var(
    constraints: Sequence[cs.Constraint],
) -> dict[cs.Variable, cs.Divides]:
  result: dict[cs.Variable, cs.Divides] = {}
  for constraint in constraints:
    if isinstance(constraint, cs.Divides) and isinstance(
        constraint.expr, cs.Variable
    ):
      assert constraint.expr not in result
      result[constraint.expr] = constraint
  return result


# TODO(bchetioui): flatten this call hierarchy.
def _extract_variable_assignments_from_constraints(
    constraints: Sequence[cs.Constraint],
) -> Iterator[tuple[cs.Variable, cs.Constant]]:
  """Attempts to extract variable assignments from all constraints."""
  dpv = _divides_per_var(constraints)
  for c in constraints:
    match c:
      case cs.IsTransferable():
        yield from _extract_layout_candidates_from_memory_space_transfer(c, dpv)
      case cs.Equals(cs.Reduce(cs.Variable() as large, axes=axes), cs.RegisterLayout() as small):
        for layout in extract_assignment_candidates_from_reduce_equation(small, large, axes):
          yield large, layout
      case cs.Equals(cs.RegisterLayout() as small, cs.Reduce(cs.Variable() as large, axes=axes)):
        for layout in extract_assignment_candidates_from_reduce_equation(small, large, axes):
          yield large, layout
      case cs.Relayout(cs.Variable() as var, cs.RegisterLayout() as layout):
        yield var, layout
      case cs.Relayout(cs.RegisterLayout() as layout, cs.Variable() as var):
        yield var, layout
      case cs.IsValidMmaTiling() as mma_tiling:
        yield from _extract_layout_candidates_from_mma_tiling(mma_tiling)


def conjure_assignment(
    unknowns: Sequence[cs.Variable],
    constraint_system: cs.ConstraintSystem,
) -> Iterator[tuple[cs.Variable, cs.Constant]]:
  """Attempts to conjure an assignment for an unknown variable."""
  # TODO(allanrenucci): We should be able to short-circuit the search here if
  # the constraint is not satisfiable.

  # As we extract assignment candidates from constraints, we prioritize
  # candidates that are more "interesting"; e.g., in the case of registers,
  # introducing splat layout candidate assignments often leads to a dead end in
  # practice---as opposed to tiled layouts, which are more likely to yield
  # solutions to the constraint system.
  low_priority_assignments: list[tuple[cs.Variable, cs.Constant]] = []
  for variable, constant in _extract_variable_assignments_from_constraints(
      constraint_system.constraints
  ):
    match constant:
      case cs.RegisterLayout(value=value) if not isinstance(value, fa.TiledLayout):
        low_priority_assignments.append((variable, constant))
      case _:
        yield variable, constant

  # After all high-priority assignments have been attempted, switch to using
  # low-priority assignments.
  for variable, constant in low_priority_assignments:
    yield variable, constant

  # Here, we have not managed to find an assignment for all the unknown
  # variables. We now try to introduce new arbitrary (valid) assignments into
  # the system, and hope that they turn out to be compatible with the constraint
  # system.
  for variable in unknowns:
    if variable in constraint_system.assignments:
      continue
    # Try to instantiate a single variable to a default layout and see if it
    # reduces the system.
    match variable.key.memory_space:
      case MemorySpace.REG:
        layout = _strided_layout_for_variable(variable)
        if layout is not None:
          yield variable, cs.RegisterLayout(layout)
      case MemorySpace.SMEM:
        yield variable, cs.SMEMTiling(None)
      case MemorySpace.TMEM:
        layout = _default_tmem_layout_for_variable(variable)
        if layout is not None:
          yield variable, cs.TMEMLayout(layout)
      case _:
        raise ValueError(f"Unsupported memory space: {variable.key.memory_space}")


def find_assignments_for(
    unknowns: Sequence[cs.Variable],
    constraint_system: cs.ConstraintSystem,
    *,
    fuel: int,
) -> tuple[dict[cs.Variable, cs.Constant] | cs.Unsatisfiable, int]:
  """Attempts to find assignments that satisfy `constraint_system` for `unknowns`.

  Args:
    unknowns: the set of variables that are unknown. Represented as a sequence
      of `Variable`s for determinism purposes.
    constraint_system: the constraint system to satisfy.
    fuel: the fuel to use for the search. Once the fuel is exhausted, we raise
      an error.

  Returns:
    A tuple where the first element is the solution, and the second element is
    the fuel remaining after the search. The solution is either:
      - Unsatisfiable() if the constraint system has unsatisfiable constraints.
      - A dictionary assigning all the unknown variables to
        `ConstantExpression`s such that the assignment satisfies the constraint
        system otherwise.
  """
  constraint_system = cs.reduce(constraint_system)
  if isinstance(constraint_system, cs.Unsatisfiable):
    return cs.Unsatisfiable(), fuel

  remaining_unknowns = [
      u for u in unknowns if u not in constraint_system.assignments.keys()
  ]

  # In this case, we have determined an assignment for all the unknown
  # variables. Return their respective assignment.
  if not remaining_unknowns:
    assert not constraint_system.constraints, (
        "A satisfiable system should not have remaining unsatisfied"
        " constraints. This is a bug."
    )
    return {
        v: k for v, k in constraint_system.assignments.items() if v in unknowns
    }, fuel

  # If unknowns remain and we have fully reduced the system, we may still
  # be able to make progress by trying out potential assignments. These
  # new assignments could make the system unsatisfiable, so we use a recursive
  # call to be able to backtrack if necessary.
  for assignment in conjure_assignment(
      remaining_unknowns, constraint_system
  ):
    if fuel <= 0:
      raise ValueError(
          "Layout inference failed to find a solution. Consider adding layout "
          "annotations to your program to guide the search."
      )
    # Trying one assignment consumes fuel.
    fuel -= 1
    variable, expr = assignment
    new_constraint_system = (
        cs.ConstraintSystem(assignments={variable: expr}) & constraint_system
    )
    if isinstance(new_constraint_system, cs.Unsatisfiable):
      # This assignment is not compatible with the constraint system.
      continue
    solution, fuel = find_assignments_for(
        unknowns, new_constraint_system, fuel=fuel
    )
    if not isinstance(solution, cs.Unsatisfiable):
      return solution, fuel

  # TODO(bchetioui): should we have a way to give a useful dump to the user
  # here, perhaps indicating what to layout cast.
  return cs.Unsatisfiable(), fuel


@dataclasses.dataclass()
class DerivationContext:
  """Holds context information used for deriving an constraint system."""
  # A map of `ValueSite` to the variable that it is associated with.
  variable_for_value_site: dict[ValueSite, cs.Variable] = dataclasses.field(
      default_factory=dict, init=False
  )
  # A map of `cs.Variable` to all the `ValueSite`s that it is associated with.
  value_sites_for_variable: ValueSitesForVariable = (
      dataclasses.field(default_factory=dict, init=False)
  )

  def update(self, mapping: ValueSitesForVariable) -> None:
    for variable, value_sites in mapping.items():
      if variable in self.value_sites_for_variable:
        self.value_sites_for_variable[variable].extend(value_sites)
      else:
        self.value_sites_for_variable[variable] = value_sites
      for value_site in value_sites:
        assert value_site not in self.variable_for_value_site
        self.variable_for_value_site[value_site] = variable

  def producer_ref(self, operand: ValueSite) -> cs.Variable:
    """Returns the producer reference variable for the given operand."""
    return self.variable_for_value_site[producer_result(operand)]


ValueSitesForVariable = dict[cs.Variable, list[ValueSite]]

# A constraint system derivation rule is a function that takes an MLIR operation
# and returns a constraint system, and a mapping from variables to value site
# identifiers.
#
# The intended meaning of the mapping is that, for each identifier in the list
# keyed by a given variable, the MLIR operand/result/argument corresponding to
# that identifier has the same layout as the variable.
#
# A `ConstraintSystemDerivationRule` must return a mapping such that the
# identifier corresponding to each value site must appear in the mapping,
# and each identifier in the mapping must be keyed by exactly one variable.
# Lastly, the mapping must only refer to variables and
# operands/results/arguments that correspond to the given operation.
ConstraintSystemDerivationRuleResult = cs.Unsatisfiable | tuple[
    cs.ConstraintSystem, ValueSitesForVariable
]
ConstraintSystemDerivationRule = Callable[
    [DerivationContext, ir.OpView],
    ConstraintSystemDerivationRuleResult,
]
_constraint_system_derivation_rules: dict[
    str, ConstraintSystemDerivationRule
] = {}


def _add_constraint_system_derivation_rule(op: type[ir.OpView]):
  def wrapper(rule: ConstraintSystemDerivationRule):
    if op is not None:
      _constraint_system_derivation_rules[op.OPERATION_NAME] = rule  # pytype: disable=attribute-error
    return rule

  return wrapper


def is_vector(v: ir.Value) -> bool:
  return isinstance(v.type, ir.VectorType)


def _is_smem_ref(v: ir.Value) -> bool:
  return isinstance(v.type, ir.MemRefType) and utils.is_smem_ref(v)


def _is_tmem_ref(v: ir.Value) -> bool:
  return isinstance(v.type, ir.MemRefType) and utils.is_tmem_ref(v)


def _pointwise_op_constraint_system(
    ctx: DerivationContext,
    op: ir.OpView,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  all_value_sites = vector_value_sites(op)
  variable = cs.Variable(all_value_sites[-1])
  return cs.ConstraintSystem(), {variable: all_value_sites}


for _op in [
    arith.AddIOp,
    arith.AddFOp,
    arith.AndIOp,
    arith.BitcastOp,
    arith.CmpFOp,
    arith.CmpIOp,
    arith.ExtFOp,
    arith.ExtSIOp,
    arith.ExtUIOp,
    arith.FPToSIOp,
    arith.FPToUIOp,
    arith.MaximumFOp,
    arith.MaxUIOp,
    arith.MaxSIOp,
    arith.MinimumFOp,
    arith.MinUIOp,
    arith.MinSIOp,
    arith.MulIOp,
    arith.MulFOp,
    arith.OrIOp,
    arith.FloorDivSIOp,
    arith.DivUIOp,
    arith.DivFOp,
    arith.RemUIOp,
    arith.RemSIOp,
    arith.RemFOp,
    arith.SIToFPOp,
    arith.UIToFPOp,
    arith.SubIOp,
    arith.SubFOp,
    arith.TruncFOp,
    arith.TruncIOp,
    arith.XOrIOp,
    arith.SelectOp,
    mlir_math.ExpOp,
    mlir_math.Exp2Op,
    mlir_math.SinOp,
    mlir_math.CosOp,
    mlir_math.LogOp,
    mlir_math.RsqrtOp,
    mlir_math.TanhOp,
    mlir_math.AbsFOp,
    mlir_math.AbsIOp,
    mlir_math.RoundOp,
    mlir_math.RoundEvenOp,
    mlir_math.CopySignOp,
]:
  _add_constraint_system_derivation_rule(_op)(_pointwise_op_constraint_system)


@_add_constraint_system_derivation_rule(mgpu.VectorLoadOp)
def _vector_load_constraint_system(
    ctx: DerivationContext,
    op: mgpu.VectorLoadOp,
) -> ConstraintSystemDerivationRuleResult:
  # TODO(b/447079781): Investigate whether we should check for contiguous
  # strides here. An initial implementation of this failed the
  # test_gmem_to_smem_with_multiple_smem_indexers_and_transforms test, but
  # we should confirm that this is properly supported.

  # Registers
  dest = ValueSite(op, VariableType.RESULT, 0)
  dest_var = cs.Variable(dest)
  value_sites_for_variable = {dest_var: [dest]}
  constraints = [cs.NotOfType(dest_var, fa.WGSplatFragLayout)]

  # SMEM
  if utils.is_smem_ref(op.source):
    source = ValueSite(op, VariableType.OPERAND, 0)
    source_var = ctx.producer_ref(source)
    value_sites_for_variable[source_var] = [source]
    shape = tuple(ir.MemRefType(op.source.type).shape)
    constraints.append(cs.IsTransferable(source_var, dest_var, shape))

  system = cs.ConstraintSystem(constraints=constraints)
  return system, value_sites_for_variable


@_add_constraint_system_derivation_rule(mgpu.VectorStoreOp)
def _vector_store_constraint_system(
    ctx: DerivationContext,
    op: mgpu.VectorStoreOp,
) -> ConstraintSystemDerivationRuleResult:
  # TODO(b/447079781): Investigate whether we should check for contiguous
  # strides here. An initial implementaiton of this failed the
  # test_gmem_to_smem_with_multiple_smem_indexers_and_transforms test, but
  # we should confirm that this is properly supported.

  # Registers
  value = ValueSite(op, VariableType.OPERAND, 0)
  value_var = cs.Variable(value)
  value_sites_for_variable = {value_var: [value]}

  # SMEM
  constraints = []
  if utils.is_smem_ref(op.destination):
    dest = ValueSite(op, VariableType.OPERAND, 1)
    dest_var = ctx.producer_ref(dest)
    value_sites_for_variable[dest_var] = [dest]
    shape = tuple(ir.MemRefType(op.destination.type).shape)
    constraints.append(cs.IsTransferable(value_var, dest_var, shape))

  system = cs.ConstraintSystem(constraints=constraints)
  return system, value_sites_for_variable


@_add_constraint_system_derivation_rule(mgpu.DebugPrintOp)
def _debug_print_constraint_system(
    ctx: DerivationContext,
    op: mgpu.DebugPrintOp,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  value = ValueSite(op, VariableType.OPERAND, 0)
  return cs.ConstraintSystem(), {cs.Variable(value): [value]}


@_add_constraint_system_derivation_rule(mgpu.PrintLayoutOp)
def _print_layout_constraint_system(
    ctx: DerivationContext,
    op: mgpu.PrintLayoutOp,
) -> ConstraintSystemDerivationRuleResult:
  value = ValueSite(op, VariableType.OPERAND, 0)
  var = cs.Variable(value) if is_vector(op.value) else ctx.producer_ref(value)
  return cs.ConstraintSystem(), {var: [value]}


@_add_constraint_system_derivation_rule(mgpu.BroadcastedIotaOp)
def _broadcasted_iota_constraint_system(
    ctx: DerivationContext,
    op: mgpu.BroadcastedIotaOp,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  value = ValueSite(op, VariableType.RESULT, 0)
  var = cs.Variable(value)
  constraints = [cs.NotOfType(var, fa.WGSplatFragLayout)]
  return cs.ConstraintSystem(constraints=constraints), {var: [value]}


@_add_constraint_system_derivation_rule(mgpu.OptimizationBarrierOp)
def _optimization_barrier_constraint_system(
    ctx: DerivationContext,
    op: ir.OpView,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  value_sites_for_variable: ValueSitesForVariable = {}

  for i, operand in enumerate(op.operands):
    if not is_vector(operand):
      continue
    variable = cs.Variable(ValueSite(op, VariableType.OPERAND, i))
    value_sites_for_variable[variable] = [
        ValueSite(op, VariableType.OPERAND, i),
        ValueSite(op, VariableType.RESULT, i)
    ]

  return cs.ConstraintSystem(), value_sites_for_variable


@_add_constraint_system_derivation_rule(vector.BroadcastOp)
def _vector_splat_constraint_system(
    ctx: DerivationContext,
    op: ir.OpView,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  result = ValueSite(op, VariableType.RESULT, 0)
  variable = cs.Variable(result)
  layout = fa.WGSplatFragLayout(tuple(cast(ir.ShapedType, op.result.type).shape))
  system = cs.ConstraintSystem(
      assignments={variable: cs.RegisterLayout(layout)}
  )
  return system, {variable: [result]}


@_add_constraint_system_derivation_rule(arith.ConstantOp)
def _constant_constraint_system(
    ctx: DerivationContext,
    constant_op: arith.ConstantOp,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  value = constant_op.value
  result = ValueSite(constant_op, VariableType.RESULT, 0)
  variable = cs.Variable(result)
  shape = tuple(ir.ShapedType(constant_op.result.type).shape)
  if (
      isinstance(value, ir.DenseElementsAttr)
      and ir.DenseElementsAttr(value).is_splat
  ):
    layout = fa.WGSplatFragLayout(shape=shape)
    system = cs.ConstraintSystem(
        assignments={variable: cs.RegisterLayout(layout)}
    )
  else:
    constant_is_not_splat = cs.NotOfType(variable, fa.WGSplatFragLayout)
    system = cs.ConstraintSystem(constraints=[constant_is_not_splat])

  return system, {variable: [result]}


def _terminator(
    block: ir.Block, expected_terminator: type[ir.OpView]
) -> ir.OpView:
  """Returns the terminator of the given block.

  Checks that the terminator is of the expected type.
  """
  terminator = block.operations[len(block.operations) - 1]
  assert isinstance(terminator, expected_terminator)
  return terminator.opview


@_add_constraint_system_derivation_rule(scf.ForOp)
def _for_constraint_system(
    ctx: DerivationContext,
    op: scf.ForOp,
) -> ConstraintSystemDerivationRuleResult:
  [block] = op.region.blocks
  yield_op = _terminator(block, scf.YieldOp)
  value_sites_for_variable: ValueSitesForVariable = {}

  # Account for the lower bound, upper bound, and step of the loop, which appear
  # in the operands but not in the results.
  num_leading_args = 3
  for index, o in enumerate(op.operands):
    if not is_vector(o) and not _is_smem_ref(o):
      continue
    result_index = index - num_leading_args
    arg_index = index - num_leading_args + 1  # Account for the induction var.
    operand = ValueSite(op, VariableType.OPERAND, index)
    arg = ValueSite(op, VariableType.ARGUMENT, arg_index, region_index=0)
    result = ValueSite(op, VariableType.RESULT, result_index)
    yield_operand = ValueSite(
        yield_op, VariableType.OPERAND, result_index
    )
    var = cs.Variable(operand) if is_vector(o) else ctx.producer_ref(operand)
    value_sites_for_variable[var] = [operand, arg, result, yield_operand]

  return cs.ConstraintSystem(), value_sites_for_variable


def prime_decomposition(n: int) -> list[int]:
  """Returns the prime decomposition of the given number `n` as a list of ints.

  A factor appears as many times in the list as the power up to which it divides
  `n`.
  """
  # This implementation should be sufficiently efficient for small `n`, which
  # should always be the case for us.
  prime_factors = []
  divisor = 2
  while divisor * divisor <= n:
    while n % divisor == 0:
      n //= divisor
      prime_factors.append(divisor)
    divisor += 1
  if n != 1:
    prime_factors.append(n)
  return prime_factors


# TODO(bchetioui): let's see if we need to parametrize this by depth.
def dynamic_gcd(a: int, b: ir.Value) -> int:
  if a <= 0:
    raise ValueError("a must be strictly positive")
  if isinstance(b.type, ir.VectorType):
    # We don't actually know the values of the vector elements, so we pick 1
    # as the only safe value.
    return 1
  if not isinstance(b.type, ir.IntegerType) and not isinstance(
      b.type, ir.IndexType
  ):
    raise ValueError(f"Expected an integer dynamic value, got a {b.type}")
  if isinstance(b.owner, arith.ConstantOp):
    return math.gcd(a, b.owner.literal_value)
  running_gcd = 1
  for factor in prime_decomposition(a):
    if utils.is_known_divisible(b, running_gcd * factor):
      running_gcd *= factor
  return running_gcd


@_add_constraint_system_derivation_rule(scf.WhileOp)
def _while_constraint_system(
    ctx: DerivationContext,
    op: scf.WhileOp,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  [before_block] = op.before.blocks
  [after_block] = op.after.blocks
  cond_op = _terminator(before_block, scf.ConditionOp)
  yield_op = _terminator(after_block, scf.YieldOp)

  value_sites_for_variable: ValueSitesForVariable = {}

  for value_site in vector_value_sites(op):
    idx = value_site.index
    match value_site.type:
      case VariableType.OPERAND:
        arg = ValueSite(op, VariableType.ARGUMENT, idx, region_index=0)
        yield_operand = ValueSite(yield_op, VariableType.OPERAND, idx)
        value_sites_for_variable[cs.Variable(value_site)] = [
            value_site,
            arg,
            yield_operand,
        ]
      case VariableType.RESULT:
        # Increment by 1 to account for the conditional.
        cond_operand = ValueSite(cond_op, VariableType.OPERAND, idx + 1)
        arg = ValueSite(op, VariableType.ARGUMENT, idx, region_index=1)
        value_sites_for_variable[cs.Variable(value_site)] = [
            value_site,
            arg,
            cond_operand,
        ]
      case _ as never:
        assert_never(never)  # pytype: disable=wrong-arg-types

  return cs.ConstraintSystem(), value_sites_for_variable


@_add_constraint_system_derivation_rule(scf.IndexSwitchOp)
def _index_switch_constraint_system(
    ctx: DerivationContext,
    op: scf.IndexSwitchOp,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  value_sites_for_variable: ValueSitesForVariable = {
      cs.Variable(o): [o] for o in vector_value_sites(op)
  }
  for region in op.regions:
    [block] = region.blocks
    yield_op = _terminator(block, scf.YieldOp)
    for value_site in value_sites_for_variable.keys():
      assert value_site.key.type == VariableType.RESULT
      yield_operand = ValueSite(
          yield_op, VariableType.OPERAND, value_site.key.index
      )
      value_sites_for_variable[value_site].append(yield_operand)

  return cs.ConstraintSystem(), value_sites_for_variable


@_add_constraint_system_derivation_rule(mgpu.LayoutCastOp)
def _layout_cast_constraint_system(
    ctx: DerivationContext,
    op: mgpu.LayoutCastOp,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  operand = ValueSite(op, VariableType.OPERAND, 0)
  result = ValueSite(op, VariableType.RESULT, 0)
  variable = cs.Variable(operand)
  out_layout = layouts_lib.from_layout_attr(op.new_layout)
  # TODO(bchetioui): think about raising a better error here.
  if not is_valid_register_layout_assignment(operand.shape, out_layout):
    return cs.Unsatisfiable()
  return (
      cs.ConstraintSystem(
          assignments={variable: cs.RegisterLayout(out_layout)}
      ),
      {variable: [operand, result]},
  )


def _infer_tiling_for_mma_ref(
    ref_ty: ir.MemRefType, max_swizzle: mgpu.SwizzlingMode
) -> tuple[int, int]:
  element_bytewidth = utils.bytewidth(ref_ty.element_type)
  strides, _ = ref_ty.get_strides_and_offset()
  min_dim_index = np.argmin(strides)
  minor_dim = ref_ty.shape[min_dim_index]

  # Try tiling with all swizzling modes starting from the largest one.
  for swizzle in [
      mgpu.SwizzlingMode.k128ByteSwizzle,
      mgpu.SwizzlingMode.k64ByteSwizzle,
      mgpu.SwizzlingMode.k32ByteSwizzle,
      mgpu.SwizzlingMode.kNoSwizzle,
  ]:
    if swizzle > max_swizzle:
      continue
    swizzle_elems = swizzle // element_bytewidth
    if minor_dim % swizzle_elems == 0:
      minor_tiling = swizzle_elems
      break
  else:
    # No valid tile transform can be inferred.
    raise ValueError(f"{ref_ty.shape} is not a valid WGMMA shape")

  major_tiling = 8
  transposed = min_dim_index != len(strides) - 1
  if transposed:
    tiling = (minor_tiling, major_tiling)
  else:
    tiling = (major_tiling, minor_tiling)
  return tiling


@_add_constraint_system_derivation_rule(mgpu.WGMMAOp)
def _wgmma_constraint_system(
    ctx: DerivationContext,
    op: mgpu.WGMMAOp,
) -> ConstraintSystemDerivationRuleResult:
  assignments: dict[cs.Variable, cs.Constant] = {}
  value_sites_for_variable: ValueSitesForVariable = {}

  acc_out = ValueSite(op, VariableType.RESULT, 0)
  acc_in = ValueSite(op, VariableType.OPERAND, 0)
  acc_var = cs.Variable(acc_out)
  acc_layout = fa.WGMMA_LAYOUT
  assignments[acc_var] = cs.RegisterLayout(acc_layout)
  if not is_valid_register_layout_assignment(acc_out.shape, acc_layout):
    return cs.Unsatisfiable()
  value_sites_for_variable[acc_var] = [acc_in, acc_out]

  b = ValueSite(op, VariableType.OPERAND, 2)
  b_var = ctx.producer_ref(b)
  input_bitwidth = utils.bitwidth(op.b.type.element_type)
  b_is_transposed = utils.is_memref_transposed(ir.MemRefType(op.b.type))
  if b_is_transposed:
    constraints = [cs.IsValidMmaTiling(cs.Transpose(b_var), input_bitwidth)]
  else:
    constraints = [cs.IsValidMmaTiling(b_var, input_bitwidth)]
  value_sites_for_variable[b_var] = [b]

  a = ValueSite(op, VariableType.OPERAND, 1)
  if _is_smem_ref(op.a):
    a_var = ctx.producer_ref(a)
    # We expect the tiling transform to be physically the same on both sides.
    # However, the constraint system assigns tiling transforms based on the
    # logical shape. In the case the tiled dimensions of exactly one of the
    # operands are transposed, we need to transpose the transform as well.
    a_is_transposed = utils.is_memref_transposed(ir.MemRefType(op.a.type))
    if a_is_transposed != b_is_transposed:
      constraints.append(cs.Equals(lhs=a_var, rhs=cs.Transpose(b_var)))
    else:
      constraints.append(cs.Equals(lhs=a_var, rhs=b_var))
  else:
    a_var = cs.Variable(a)
    if utils.bitwidth(op.a.type.element_type) == 8:
      layout = fa.WGMMA_LAYOUT_8BIT
    else:
      layout = fa.WGMMA_LAYOUT
    assignments[a_var] = cs.RegisterLayout(layout)
    # TODO(bchetioui): raise a better error here.
    if not is_valid_register_layout_assignment(a.shape, layout):
      return cs.Unsatisfiable()

  value_sites_for_variable[a_var] = [a]
  return cs.ConstraintSystem(assignments, constraints), value_sites_for_variable


@_add_constraint_system_derivation_rule(vector.BroadcastOp)
def _vector_broadcast_constraint_system(
    ctx: DerivationContext,
    op: vector.BroadcastOp,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  # This is not expected to be necessary at the moment. We should be using
  # mgpu.BroadcastInDimOp instead when dealing with broadcasting vectors.
  if isinstance(op.source.type, ir.ShapedType):
    raise NotImplementedError("Only vector broadcasts from scalars are supported.")
  out_variable = cs.Variable(ValueSite(op, VariableType.RESULT, 0))
  layout = cs.RegisterLayout(fa.WGSplatFragLayout(tuple(op.result.type.shape)))
  return (
      cs.ConstraintSystem(assignments={out_variable: layout}),
      {out_variable: [out_variable.key]},
  )


@_add_constraint_system_derivation_rule(vector.ReductionOp)
def _vector_reduction_constraint_system(
    ctx: DerivationContext,
    op: vector.ReductionOp,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  in_variable = cs.Variable(ValueSite(op, VariableType.OPERAND, 0))
  return cs.ConstraintSystem(), {in_variable: [in_variable.key]}


def _reduction_constraints(
    larger: cs.Variable,
    smaller: cs.Variable,
    reduction_dims: tuple[int, ...],
) -> list[cs.Constraint]:
  return [
      cs.Equals(lhs=smaller, rhs=cs.Reduce(larger, reduction_dims)),
      # TODO(allanrenucci): Remove once we support reduction of strided layouts.
      cs.NotOfType(larger, fa.WGStridedFragLayout),
  ]


@_add_constraint_system_derivation_rule(vector.MultiDimReductionOp)
def _multi_dim_reduction_constraint_system(
    ctx: DerivationContext,
    op: vector.MultiDimReductionOp,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  source = ValueSite(op, VariableType.OPERAND, 0)
  acc = ValueSite(op, VariableType.OPERAND, 1)
  out = ValueSite(op, VariableType.RESULT, 0)
  source_variable = cs.Variable(source)
  out_variable = cs.Variable(out)

  reduction_constraints = _reduction_constraints(
      source_variable,
      out_variable,
      tuple(op.reduction_dims),
  )
  # TODO(bchetioui): in the future, we may need to add rules that prevent
  # strided layouts from being chosen---since trying to reduce a strided layout
  # may cause us to raise an Exception at the moment.
  return (
      cs.ConstraintSystem(constraints=reduction_constraints),
      {source_variable: [source], out_variable: [acc, out]},
  )


@_add_constraint_system_derivation_rule(mgpu.BroadcastInDimOp)
def _broadcast_in_dim_constraint_system(
    ctx: DerivationContext,
    op: mgpu.BroadcastInDimOp,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  out_variable = cs.Variable(ValueSite(op, VariableType.RESULT, 0))
  source_variable = cs.Variable(ValueSite(op, VariableType.OPERAND, 0))
  out_shape = tuple(cast(ir.ShapedType, op.result.type).shape)
  reduction_dims = tuple(
      i for i in range(len(out_shape)) if i not in op.broadcast_dimensions
  )
  reduction_constraints = _reduction_constraints(
      out_variable, source_variable, reduction_dims
  )

  return (
      cs.ConstraintSystem(constraints=reduction_constraints),
      {
          source_variable: [source_variable.key],
          out_variable: [out_variable.key],
      },
  )


@_add_constraint_system_derivation_rule(vector.ShapeCastOp)
def _shape_cast_constraint_system(
    ctx: DerivationContext, op: vector.ShapeCastOp
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  in_shape = tuple(cast(ir.ShapedType, op.source.type).shape)
  out_shape = tuple(cast(ir.ShapedType, op.result.type).shape)

  in_variable = cs.Variable(ValueSite(op, VariableType.OPERAND, 0))
  out_variable = cs.Variable(ValueSite(op, VariableType.RESULT, 0))

  # Here, we are in a case where we are stating
  #
  #   out_variable = reshape(in_variable, in_shape, out_shape).
  #
  # Thanks to the symmetric property of reshape, we can also issue a constraint
  # in the other direction, i.e.
  #
  #   in_variable = reshape(out_variable, out_shape, in_shape)
  #
  # in order to be able to figure out an assignment for `in_variable`. if we
  # happen to know `out_variable`. If we only issue the first constraint, then
  # we will not be able to figure out an assignment for `in_variable` if we
  # only know `out_variable`, even though their relationship is fully
  # determined.
  in_to_out = cs.Reshape(
      in_variable, source_shape=in_shape, target_shape=out_shape
  )
  out_to_in = cs.Reshape(
      out_variable, source_shape=out_shape, target_shape=in_shape
  )

  return (
      cs.ConstraintSystem(
          constraints=[
              cs.Equals(lhs=out_variable, rhs=in_to_out),
              cs.Equals(lhs=in_variable, rhs=out_to_in),
          ],
      ),
      {in_variable: [in_variable.key], out_variable: [out_variable.key]},
  )


@_add_constraint_system_derivation_rule(vector.ExtractStridedSliceOp)
def _extract_strided_slice_constraint_system(
    ctx: DerivationContext, op: vector.ExtractStridedSliceOp
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  if any(ir.IntegerAttr(s).value != 1 for s in op.strides):
    raise NotImplementedError("`strides` must contain only 1s.")
  operand = ValueSite(op, VariableType.OPERAND, 0)
  result = ValueSite(op, VariableType.RESULT, 0)
  variable = cs.Variable(operand)
  offsets = tuple(ir.IntegerAttr(o).value for o in op.offsets)
  constraints = [
      cs.Divides(variable, offsets),
      # TODO(allanrenucci): Remove once vectors with splat and strided layouts
      # can be sliced.
      cs.NotOfType(variable, fa.WGSplatFragLayout),
      cs.NotOfType(variable, fa.WGStridedFragLayout),
  ]
  return (
      cs.ConstraintSystem(constraints=constraints),
      # We use a single variable because lowering does not support two different
      # layouts for `source` and `result`.
      {variable: [operand, result]},
  )


@_add_constraint_system_derivation_rule(vector.ExtractOp)
def _vector_extract_constraint_system(
    ctx: DerivationContext, op: vector.ExtractOp
) -> tuple[cs.ConstraintSystem, ValueSitesForVariable]:
  del ctx
  if not isinstance(op.result.type, ir.VectorType):  # scalar result
    operand = ValueSite(op, VariableType.OPERAND, 0)
    variable = cs.Variable(operand)
    layout = fa.WGSplatFragLayout(tuple(op.source.type.shape))
    # We only support indexing for splat layout.
    assignments = {variable: cs.RegisterLayout(layout)}
    return cs.ConstraintSystem(assignments), {variable: [operand]}

  if op.dynamic_position:
    raise NotImplementedError("Only slicing with static indices allowed.")
  operand = ValueSite(op, VariableType.OPERAND, 0)
  result = ValueSite(op, VariableType.RESULT, 0)
  variable = cs.Variable(operand)
  constraints = [
      cs.Divides(variable, tuple(op.result.type.shape)),
      # TODO(allanrenucci): Remove once vectors with splat and strided layouts
      # can be sliced.
      cs.NotOfType(variable, fa.WGSplatFragLayout),
      cs.NotOfType(variable, fa.WGStridedFragLayout),
  ]
  return (
      cs.ConstraintSystem(constraints=constraints),
      {variable: [operand, result]},
  )


@_add_constraint_system_derivation_rule(mgpu.CustomPrimitiveOp)
def _custom_primitive_constraint_system(
    ctx: DerivationContext,
    op: mgpu.CustomPrimitiveOp,
) -> ConstraintSystemDerivationRuleResult:
  assignments: dict[cs.Variable, cs.Constant] = {}
  constraints: list[cs.Constraint] = []
  in_layouts = iter(op.in_layouts)
  in_transforms = iter(op.in_transforms)
  variables: list[cs.Variable] = []
  for i, operand in enumerate(op.operands):
    if is_vector(operand):
      v = cs.Variable(ValueSite(op, VariableType.OPERAND, i))
      variables.append(v)
      assignments[v] = cs.RegisterLayout(
          layouts_lib.from_layout_attr(next(in_layouts))
      )
    elif _is_smem_ref(operand):
      # Here we need to create a new variable, even though it is equal to the
      # source operand. This is because we directly assign the new variable and
      # if we did that to the source there could be conflicting assignments.
      # For example, the same ref could be passed into the custom op twice with
      # different transforms, which needs to yield an unsatisfiable system.
      #
      # TODO(b/447079781): Consider creating the final constraint system using
      # __and__ and potentially returning Unsatisfiable() directly if there is
      # a conflict between the assignments.
      value_site = ValueSite(op, VariableType.OPERAND, i)
      source_var = ctx.producer_ref(value_site)
      v = cs.Variable(value_site)
      constraints.append(cs.Equals(lhs=source_var, rhs=v))
      variables.append(v)
      transforms = next(in_transforms)
      ref_ty = value_site.value.type
      tiling = _extract_smem_tiling_from_custom_transform_attrs(ref_ty, transforms)
      assignments[v] = tiling

  out_layouts = iter(op.out_layouts)
  for i, result in enumerate(op.results):
    if isinstance(result.type, ir.VectorType):
      v = cs.Variable(ValueSite(op, VariableType.RESULT, i))
      variables.append(v)
      assignments[v] = cs.RegisterLayout(
          layouts_lib.from_layout_attr(next(out_layouts))
      )
  return (
      cs.ConstraintSystem(assignments, constraints),
      {v: [v.key] for v in variables},
  )


def _tmem_layout_from_layout_attr(
    layout_attr: mgpu.TiledLayout,
) -> tcgen05.TMEMLayout:
  layout = layouts_lib.from_layout_attr(layout_attr)
  assert isinstance(layout, fa.TiledLayout)
  return tcgen05.TMEMLayout(
      layout.tiling, layout.warp_dims, layout.lane_dims, layout.vector_dim
  )


@_add_constraint_system_derivation_rule(mgpu.TmemLayoutCastOp)
def _tmem_layout_cast_constraint_system(
    ctx: DerivationContext,
    op: mgpu.TmemLayoutCastOp,
) -> ConstraintSystemDerivationRuleResult:
  operand = ValueSite(op, VariableType.OPERAND, 0)
  variable = ctx.producer_ref(operand)
  result = ValueSite(op, VariableType.RESULT, 0)
  tmem_layout = _tmem_layout_from_layout_attr(op.new_layout)
  if not is_valid_tmem_layout_assignment(operand.shape, tmem_layout):
    return cs.Unsatisfiable()
  out_layout = cs.TMEMLayout(tmem_layout)
  return (
      cs.ConstraintSystem(assignments={variable: out_layout}),
      {variable: [operand, result]},
  )


@_add_constraint_system_derivation_rule(mgpu.TmemAllocOp)
def _tmem_alloc_constraint_system(
    ctx: DerivationContext,
    op: mgpu.TmemAllocOp,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  result = ValueSite(op, VariableType.RESULT, 0)
  result_var = cs.Variable(result)
  in_smem = ValueSite(op, VariableType.OPERAND, 0)
  in_smem_var = cs.Variable(in_smem)
  assignments: dict[cs.Variable, cs.Constant] = {
      in_smem_var: cs.SMEMTiling(None)
  }
  operands_for_variable = {result_var: [result], in_smem_var: [in_smem]}
  return cs.ConstraintSystem(assignments=assignments), operands_for_variable


@_add_constraint_system_derivation_rule(mgpu.TmemDeallocOp)
def _tmem_dealloc_constraint_system(
    ctx: DerivationContext,
    op: mgpu.TmemDeallocOp,
) -> ConstraintSystemDerivationRuleResult:
  operand = ValueSite(op, VariableType.OPERAND, 0)
  variable = ctx.producer_ref(operand)
  return cs.ConstraintSystem(), {variable: [operand]}


@_add_constraint_system_derivation_rule(mgpu.TcGen05MMAOp)
def _tcgen05_mma_constraint_system(
    ctx: DerivationContext,
    op: mgpu.TcGen05MMAOp,
) -> ConstraintSystemDerivationRuleResult:
  assignments: dict[cs.Variable, cs.Constant] = {}
  operands_for_variable: ValueSitesForVariable = {}

  # TMEM
  acc = ValueSite(op, VariableType.OPERAND, 0)
  acc_variable = ctx.producer_ref(acc)
  acc_type = ir.ShapedType(op.accumulator.type)
  acc_layout = tcgen05._infer_tmem_layout(  # pylint: disable=protected-access
      tuple(acc_type.shape), op.collective, packing=1
  )
  assignments[acc_variable] = cs.TMEMLayout(acc_layout)
  acc_is_valid = is_valid_tmem_layout_assignment(acc.shape, acc_layout)
  # TODO(bchetioui): think about raising a better error here.
  if not acc_is_valid:
    return cs.Unsatisfiable()
  operands_for_variable[acc_variable] = [acc]

  element_type_bitwidth = utils.bitwidth(op.b.type.element_type)
  b = ValueSite(op, VariableType.OPERAND, 2)
  b_var = ctx.producer_ref(b)
  operands_for_variable[b_var] = [b]
  b_is_transposed = utils.is_memref_transposed(ir.MemRefType(op.b.type))
  if b_is_transposed:
    constraints = [cs.IsValidMmaTiling(cs.Transpose(b_var), element_type_bitwidth)]
  else:
    constraints = [cs.IsValidMmaTiling(b_var, element_type_bitwidth)]

  # SMEM
  M = op.accumulator.type.shape[0]
  if M == 64 and not op.collective.value:
    # We can't split N into groups if we would partition it below the tile size.
    N = op.b.type.shape[1]
    n_lane_groups = 2
    max_swizzle_elems = next(
        8 * s // element_type_bitwidth
        for s in reversed(mgpu.SwizzlingMode)
        if 8 * s // element_type_bitwidth <= N // n_lane_groups
    )
    if b_is_transposed:
      constraints.append(cs.Divides(b_var, (max_swizzle_elems, 8)))
    else:
      constraints.append(cs.Divides(b_var, (8, max_swizzle_elems)))

  if _is_tmem_ref(op.a):
    a = ValueSite(op, VariableType.OPERAND, 1)
    a_type = ir.ShapedType(op.a.type)
    a_var = ctx.producer_ref(a)
    packing = 32 // utils.bitwidth(a_type.element_type)
    a_layout = tcgen05._infer_tmem_layout(  # pylint: disable=protected-access
        tuple(a_type.shape), op.collective, packing
    )
    assignments[a_var] = cs.TMEMLayout(a_layout)
    operands_for_variable[a_var] = [a]
    a_is_valid = is_valid_tmem_layout_assignment(a.shape, a_layout)
    # TODO(bchetioui): think about raising a better error here.
    if not a_is_valid:
      return cs.Unsatisfiable()
  else:
    assert _is_smem_ref(op.a)
    a_is_transposed = utils.is_memref_transposed(ir.MemRefType(op.a.type))
    a = ValueSite(op, VariableType.OPERAND, 1)
    a_var = ctx.producer_ref(a)
    operands_for_variable[a_var] = [a]
    if a_is_transposed:
      constraints.append(cs.IsValidMmaTiling(cs.Transpose(a_var), element_type_bitwidth))
    else:
      constraints.append(cs.IsValidMmaTiling(a_var, element_type_bitwidth))

  return cs.ConstraintSystem(assignments=assignments, constraints=constraints), operands_for_variable


@_add_constraint_system_derivation_rule(mgpu.AsyncLoadTmemOp)
def _async_load_tmem_constraint_system(
    ctx: DerivationContext,
    op: mgpu.AsyncLoadTmemOp,
) -> ConstraintSystemDerivationRuleResult:
  source = ValueSite(op, VariableType.OPERAND, 0)
  source_variable = ctx.producer_ref(source)
  destination = ValueSite(op, VariableType.RESULT, 0)
  destination_variable = cs.Variable(destination)
  constraint = cs.IsTransferable(
      source_variable,
      destination_variable,
      tuple(ir.ShapedType(op.source.type).shape),
  )
  return (
      cs.ConstraintSystem(constraints=[constraint]),
      {source_variable: [source], destination_variable: [destination]},
  )


@_add_constraint_system_derivation_rule(mgpu.SliceTmemOp)
def _slice_tmem_constraint_system(
    ctx: DerivationContext,
    op: mgpu.SliceTmemOp,
) -> ConstraintSystemDerivationRuleResult:
  operand = ValueSite(op, VariableType.OPERAND, 0)
  operand_variable = ctx.producer_ref(operand)
  result = ValueSite(op, VariableType.RESULT, 0)
  result_variable = cs.Variable(result)
  return (
      cs.ConstraintSystem(),
      {operand_variable: [operand], result_variable: [result]},
  )


@_add_constraint_system_derivation_rule(mgpu.AsyncStoreTmemOp)
def _async_store_tmem_constraint_system(
    ctx: DerivationContext,
    op: mgpu.AsyncStoreTmemOp,
) -> ConstraintSystemDerivationRuleResult:
  source = ValueSite(op, VariableType.OPERAND, 0)
  source_variable = cs.Variable(source)
  destination = ValueSite(op, VariableType.OPERAND, 1)
  destination_variable = ctx.producer_ref(destination)
  constraint = cs.IsTransferable(
      source_variable,
      destination_variable,
      tuple(ir.ShapedType(op.source.type).shape),
  )
  return (
      cs.ConstraintSystem(constraints=[constraint]),
      {source_variable: [source], destination_variable: [destination]},
  )


@_add_constraint_system_derivation_rule(mgpu.SliceSMEMOp)
def _slice_smem_constraint_system(
    ctx: DerivationContext,
    op: mgpu.SliceSMEMOp,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  res = ValueSite(op, VariableType.RESULT, 0)
  res_var = cs.Variable(res)
  return cs.ConstraintSystem(), {res_var: [res]}


@_add_constraint_system_derivation_rule(memref.SubViewOp)
def _memref_subview_constraint_system(
    ctx: DerivationContext,
    op: memref.SubViewOp,
) -> ConstraintSystemDerivationRuleResult:
  source = ValueSite(op, VariableType.OPERAND, 0)
  dest = ValueSite(op, VariableType.RESULT, 0)
  source_dest_var = ctx.producer_ref(source)

  if any(s != 1 for s in op.static_strides):
    raise NotImplementedError(
        f"Only unit strides are supported but got {op.static_strides}."
    )

  # Collect all the constraints from all dimensions.
  tiling_multiple = []
  dynamic_offset_index = 0
  for i, size in enumerate(op.static_sizes):
    offset = op.static_offsets[i]
    if offset == ir.ShapedType.get_dynamic_size():
      offset = op.offsets[dynamic_offset_index]
      dynamic_offset_index += 1

    # Drop all dimensions up to and including the last dynamic size. Dynamic
    # sizes are not supported yet.
    #
    # Supporting dynamic sizes here can be done analogously to how dynamic
    # offsets are supported. The reason we don't support dynamic sizes now is
    # because the lowering does not yet support them.
    if ir.ShapedType.is_dynamic_size(size):
      tiling_multiple = []
    else:
      src_type = ir.MemRefType(op.source.type)
      divisibility_constraint = math.gcd(size, src_type.shape[i])
      if isinstance(offset, int):
        divisibility_constraint = math.gcd(divisibility_constraint, offset)
      else:
        divisibility_constraint = dynamic_gcd(divisibility_constraint, offset)
      tiling_multiple.append(divisibility_constraint)

  constraints = [cs.Divides(source_dest_var, tuple(tiling_multiple))]
  system = cs.ConstraintSystem(constraints=constraints)
  return system, {source_dest_var: [source, dest]}


@_add_constraint_system_derivation_rule(memref.CastOp)
def _memref_cast_op_constraint_system(
    ctx: DerivationContext,
    op: memref.CastOp,
) -> ConstraintSystemDerivationRuleResult:
  source = ValueSite(op, VariableType.OPERAND, 0)
  var_source_dest = ctx.producer_ref(source)
  dest = ValueSite(op, VariableType.RESULT, 0)
  return cs.ConstraintSystem(), {var_source_dest: [source, dest]}


@_add_constraint_system_derivation_rule(memref.TransposeOp)
def _memref_transpose_op_constraint_system(
    ctx: DerivationContext,
    op: memref.TransposeOp,
) -> ConstraintSystemDerivationRuleResult:
  in_ty = ir.MemRefType(op.in_.type)
  in_strides, _ = in_ty.get_strides_and_offset()
  out_strides, _ = ir.MemRefType(op.result.type).get_strides_and_offset()
  transpose = in_strides != out_strides

  source = ValueSite(op, VariableType.OPERAND, 0)
  dest = ValueSite(op, VariableType.RESULT, 0)
  source_var = ctx.producer_ref(source)

  if not transpose:
    return cs.ConstraintSystem(), {source_var: [source, dest]}

  dest_var = cs.Variable(dest)
  constraints = [
      cs.Equals(cs.Transpose(source_var), dest_var),
      cs.Equals(source_var, cs.Transpose(dest_var)),
  ]
  system = cs.ConstraintSystem(constraints=constraints)
  return system, {source_var: [source], dest_var: [dest]}


@_add_constraint_system_derivation_rule(memref.ExpandShapeOp)
def _memref_expand_shape_op_equation_system(
    ctx: DerivationContext,
    op: memref.ExpandShapeOp,
) -> ConstraintSystemDerivationRuleResult:
  if utils.is_memref_transposed(ir.MemRefType(op.src.type)):
    raise NotImplementedError(
        "Transposed memrefs are not supported in ExpandShapeOp."
    )

  source = ValueSite(op, VariableType.OPERAND, 0)
  dest = ValueSite(op, VariableType.RESULT, 0)
  var = ctx.producer_ref(source)

  reverse_tiling_multiple = []
  for dim, idx in zip(
      reversed(op.static_output_shape), reversed(op.reassociation)
  ):
    if ir.ShapedType.is_dynamic_size(dim) or len(idx) > 1:
      # For simplicity, we only support tiling non-expanded static dimensions.
      # These limitations could be lifted later if needed.
      break
    reverse_tiling_multiple.append(dim)

  constraints = [cs.Divides(var, tuple(reversed(reverse_tiling_multiple)))]
  return cs.ConstraintSystem(constraints=constraints), {var: [source, dest]}


# `memref.load` and `memref.store` are used to load barrier phases which are
# scalars---the rule needn't do anything interesting, but we need to have it.
@_add_constraint_system_derivation_rule(memref.LoadOp)
@_add_constraint_system_derivation_rule(memref.StoreOp)
def _memref_load_store_op_constraint_system(
    ctx: DerivationContext,
    op: memref.LoadOp | memref.StoreOp,
) -> ConstraintSystemDerivationRuleResult:
  del ctx

  ref_shape = ir.MemRefType(op.memref.type).shape
  if ref_shape and ref_shape != [1]:
    raise NotImplementedError(
        f"Only scalar memrefs are supported, got {ref_shape}"
    )

  ref_op_index = 0 if isinstance(op, memref.LoadOp) else 1
  ref = ValueSite(op, VariableType.OPERAND, ref_op_index)
  var = cs.Variable(ref)
  assignments: dict[cs.Variable, cs.Constant] = {var: cs.SMEMTiling(None)}
  return cs.ConstraintSystem(assignments=assignments), {var: [ref]}


def _extract_smem_tiling_from_custom_transform_attrs(
    ref_type: ir.MemRefType,
    transform_attrs: ir.ArrayAttr,
) -> cs.SMEMTiling:
  transforms = [layouts_lib.from_transform_attr(x) for x in transform_attrs]
  match transforms:
    case []:
      tile_transform = None
      swizzle = None
    case [lc.TileTransform() as t]:
      tile_transform = t
      swizzle = None
    case [lc.TileTransform() as t, mgpu.SwizzlingMode() as s]:
      tile_transform = t
      swizzle = s
    case _:
      raise NotImplementedError(f"Unsupported transforms {transforms}")

  if swizzle is not None:
    computed_swizzle = _compute_swizzle(ref_type, tile_transform)
    if computed_swizzle != swizzle:
      raise NotImplementedError(
          f"Cannot honor caller-provided swizzle {swizzle} that is different "
          f"from the computed swizle {computed_swizzle} for type {ref_type}."
      )

  return cs.SMEMTiling(tile_transform)


@_add_constraint_system_derivation_rule(mgpu.WithTransformsOp)
def _with_transforms_constraint_system(
    ctx: DerivationContext,
    op: mgpu.WithTransformsOp,
) -> ConstraintSystemDerivationRuleResult:
  source = ValueSite(op, VariableType.OPERAND, 0)
  dest = ValueSite(op, VariableType.RESULT, 0)
  var = ctx.producer_ref(source)
  tiling = _extract_smem_tiling_from_custom_transform_attrs(op.ref.type, op.transforms)
  if tiling.value is not None:
    # TODO(bchetioui): think about raising a better error here.
    if not is_valid_smem_layout_assignment(source.shape, tiling.value):
      return cs.Unsatisfiable()
  assignments: dict[cs.Variable, cs.Constant] = {var: tiling}
  return cs.ConstraintSystem(assignments=assignments), {var: [source, dest]}


def _vector_value_sites_and_assignments_for_async_ops(
    op: mgpu.AsyncLoadOp | mgpu.AsyncStoreOp | mgpu.AsyncPrefetchOp,
) -> tuple[ValueSitesForVariable, dict[cs.Variable, cs.Constant]]:
  values_sites: ValueSitesForVariable = dict()
  assignments: dict[cs.Variable, cs.Constant] = dict()

  match op:
    case mgpu.AsyncLoadOp():
      base_operand_index = 3
    case mgpu.AsyncStoreOp():
      base_operand_index = 2
    case mgpu.AsyncPrefetchOp():
      base_operand_index = 1
    case _:
      raise ValueError(f"Unsupported op type: {op}")  # make pytype happy

  for i, idx in enumerate(op.indices):
    if isinstance(idx.type, ir.VectorType):
      value_site = ValueSite(op, VariableType.OPERAND, base_operand_index + i)
      value_site_var = cs.Variable(value_site)
      layout = cs.RegisterLayout(value=fa.TMA_GATHER_INDICES_LAYOUT)
      values_sites[value_site_var] = [value_site]
      assignments[value_site_var] = layout
  return values_sites, assignments


@_add_constraint_system_derivation_rule(mgpu.AsyncLoadOp)
@_add_constraint_system_derivation_rule(mgpu.AsyncStoreOp)
def _async_load_store_constraint_system(
    ctx: DerivationContext,
    op: mgpu.AsyncLoadOp | mgpu.AsyncStoreOp,
) -> ConstraintSystemDerivationRuleResult:
  tiling_multiple = []
  for size, index in zip(op.slice_lengths, op.indices, strict=True):
    if size == -1:
      # This dimension does not appear in the final smem memref shape.
      continue
    tiling_multiple.append(dynamic_gcd(size, index))

  operand_index = 1 if isinstance(op, mgpu.AsyncLoadOp) else 0
  operand = ValueSite(op, VariableType.OPERAND, operand_index)
  var = ctx.producer_ref(operand)
  constraints = [cs.Divides(expr=var, tiling_multiple=tuple(tiling_multiple))]
  value_sites_for_variable = {var: [operand]}
  value_sites, assignments = _vector_value_sites_and_assignments_for_async_ops(op)
  value_sites_for_variable.update(value_sites)
  return cs.ConstraintSystem(assignments, constraints), value_sites_for_variable


@_add_constraint_system_derivation_rule(mgpu.AsyncPrefetchOp)
def _async_prefetch_constraint_system(
    ctx: DerivationContext,
    op: mgpu.AsyncPrefetchOp,
) -> ConstraintSystemDerivationRuleResult:
  del ctx
  value_sites, assignments = _vector_value_sites_and_assignments_for_async_ops(op)
  return cs.ConstraintSystem(assignments), value_sites


def _ensure_all_layouts_are_set(op: ir.OpView) -> None:
  if inference_utils.should_have_layout(op):
    _ensure_right_number_of_layouts(is_vector, "layouts", "vector", op)
  if inference_utils.should_have_tmem_layout(op):
    _ensure_right_number_of_layouts(_is_tmem_ref, "tmem_layouts", "TMEM ref", op)
  if inference_utils.should_have_transforms(op):
    _ensure_right_number_of_layouts(
        inference_utils.is_transformable_smem_memref, "transforms", "SMEM ref", op,
    )


def _ensure_right_number_of_layouts(
    filter_fn: Callable[[ir.Value], bool],
    attr_suffix: str,
    value_type: str,
    op: ir.OpView,
) -> None:
  """Ensures that the right number of in/out layouts are provided for an op.

  Layouts here are can be vector layouts, TMEM layouts, or SMEM transforms.
  """
  layouts = lambda attr: op.attributes[attr] if attr in op.attributes else []
  in_layouts = layouts(f"in_{attr_suffix}")
  out_layouts = layouts(f"out_{attr_suffix}")

  num_matching_operands = sum(map(filter_fn, op.operands))
  if len(in_layouts) != num_matching_operands:
    raise ValueError(
        f"Expected the same number of in_{attr_suffix} ({len(in_layouts)}) as "
        f"{value_type} operands ({num_matching_operands}). op=\n  {op}"
    )
  num_matching_results = sum(map(filter_fn, op.results))
  if len(out_layouts) != num_matching_results:
    raise ValueError(
        f"Expected the same number of out_{attr_suffix} ({len(out_layouts)}) "
        f"as {value_type} results ({num_matching_results}). op=\n  {op}"
    )


def _compute_swizzle(
    ty: ir.Type, tile_transform: lc.TileTransform | None
) -> mgpu.SwizzlingMode:
  """Computes the swizzle mode given a tiling transform and a data type."""
  if tile_transform is None:
    # TODO(b/447079781): Revisit if this is the behavior we want.
    return mgpu.SwizzlingMode.kNoSwizzle

  if not isinstance(ty, ir.MemRefType):
    raise ValueError(f"Expected a MemRefType, got {ty}.")
  ref_ty = ir.MemRefType(ty)
  strides, _ = ref_ty.get_strides_and_offset()
  tiling = tile_transform.tiling

  if len(tiling) > len(strides):
    raise ValueError(
        f"The tile rank ({len(tiling)}) cannot be greater than the ref's rank"
        f" ({len(strides)})."
    )

  minor_tiling = tiling[np.argmin(strides[-len(tiling):])]
  swizzle = minor_tiling * utils.bytewidth(ref_ty.element_type)
  assert swizzle in (
      mgpu.SwizzlingMode.k128ByteSwizzle,
      mgpu.SwizzlingMode.k64ByteSwizzle,
      mgpu.SwizzlingMode.k32ByteSwizzle,
      mgpu.SwizzlingMode.kNoSwizzle,
  )
  return mgpu.SwizzlingMode(swizzle)


@dataclasses.dataclass(frozen=True)
class _TypeAndLayout:
  type: ir.Type
  layout: cs.Constant


def assign_layouts(solution: dict[ValueSite, cs.Constant]) -> None:
  """Assigns the layouts in `solution` to the MLIR ops they belong to.

  This function requires that, for each MLIR op that appears in `solution`,
  `solution` contains a layout assignment for all of its `vector`, TMEM, and
  SMEM operands and results. Block arguments are ignored.
  """
  solution_sorted_by_op = sorted(
      solution.items(), key=lambda kv: id(kv[0].operation)
  )
  solution_per_op = itertools.groupby(
      solution_sorted_by_op, key=lambda kv: kv[0].operation
  )

  for op, assignments in solution_per_op:
    assignments_sorted_by_type = sorted(assignments, key=lambda kv: kv[0].type)
    assignments_by_type = {
        ty: list(group)
        for ty, group in itertools.groupby(
            assignments_sorted_by_type, key=lambda kv: kv[0].type
        )
    }

    in_assignments = assignments_by_type.get(VariableType.OPERAND, [])
    out_assignments = assignments_by_type.get(VariableType.RESULT, [])

    index = lambda kv: kv[0].index
    in_tls = [
        _TypeAndLayout(v.value.type, ce)
        for v, ce in sorted(in_assignments, key=index)
    ]
    out_tls = [
        _TypeAndLayout(v.value.type, ce)
        for v, ce in sorted(out_assignments, key=index)
    ]

    in_layouts = [
        tl.layout.value
        for tl in in_tls
        if isinstance(tl.layout, cs.RegisterLayout)
    ]
    out_layouts = [
        tl.layout.value
        for tl in out_tls
        if isinstance(tl.layout, cs.RegisterLayout)
    ]
    in_tmem_layouts = [
        tl.layout.value for tl in in_tls if isinstance(tl.layout, cs.TMEMLayout)
    ]
    out_tmem_layouts = [
        tl.layout.value
        for tl in out_tls
        if isinstance(tl.layout, cs.TMEMLayout)
    ]
    in_transforms = [
        tl for tl in in_tls if isinstance(tl.layout, cs.SMEMTiling)
    ]
    out_transforms = [
        tl for tl in out_tls if isinstance(tl.layout, cs.SMEMTiling)
    ]

    if inference_utils.should_have_in_layout(op):
      attrs = [layouts_lib.to_layout_attr(l) for l in in_layouts]
      op.attributes["in_layouts"] = ir.ArrayAttr.get(attrs)
    if inference_utils.should_have_out_layout(op):
      attrs = [layouts_lib.to_layout_attr(l) for l in out_layouts]
      op.attributes["out_layouts"] = ir.ArrayAttr.get(attrs)
    if inference_utils.should_have_in_tmem_layout(op):
      attrs = [layouts_lib.to_layout_attr(l) for l in in_tmem_layouts]
      op.attributes["in_tmem_layouts"] = ir.ArrayAttr.get(attrs)
    if inference_utils.should_have_out_tmem_layout(op):
      attrs = [layouts_lib.to_layout_attr(l) for l in out_tmem_layouts]
      op.attributes["out_tmem_layouts"] = ir.ArrayAttr.get(attrs)

    def _to_transform_attrs(
        transforms: list[_TypeAndLayout],
    ) -> list[ir.ArrayAttr]:
      all_attrs: list[ir.ArrayAttr] = []
      for tl in transforms:
        assert isinstance(tl.layout, cs.SMEMTiling)  # make pytype happy
        attrs = []
        if tl.layout.value is not None:
          attrs.append(layouts_lib.to_transform_attr(tl.layout.value))
          swizzle = _compute_swizzle(tl.type, tl.layout.value)
          attrs.append(layouts_lib.to_transform_attr(swizzle))
        all_attrs.append(ir.ArrayAttr.get(attrs))
      return all_attrs

    if inference_utils.should_have_in_transforms(op):
      attrs = _to_transform_attrs(in_transforms)
      op.attributes["in_transforms"] = ir.ArrayAttr.get(attrs)
    if inference_utils.should_have_out_transforms(op):
      attrs = _to_transform_attrs(out_transforms)
      op.attributes["out_transforms"] = ir.ArrayAttr.get(attrs)

    _ensure_all_layouts_are_set(op)


def vector_value_sites(op: ir.OpView) -> list[ValueSite]:
  """Returns all the vector operands and results for the given op."""
  value_sites = [
      ValueSite(op, VariableType.OPERAND, i)
      for i, o in enumerate(op.operands)
      if is_vector(o)
  ]
  value_sites.extend([
      ValueSite(op, VariableType.RESULT, i)
      for i, o in enumerate(op.results)
      if is_vector(o)
  ])
  return value_sites


def producer_result(operand: ValueSite) -> ValueSite:
  """Given an operand, returns the corresponding result in its producer.

  When the producer is a block, we return the corresponding operand in the
  operation that owns the block.
  """
  assert operand.type == VariableType.OPERAND
  value = operand.value
  producer = value.owner
  if isinstance(producer, ir.OpView):
    index = list(producer.results).index(value)
    return ValueSite(producer, VariableType.RESULT, index)

  if isinstance(producer, ir.Block):
    index = list(producer.arguments).index(value)  # pytype: disable=attribute-error
    region_index = list(producer.owner.regions).index(producer.region)  # pytype: disable=attribute-error
    return ValueSite(producer.owner, VariableType.ARGUMENT, index, region_index)  # pytype: disable=attribute-error

  raise TypeError(
      f"Producer {producer} is not an operation nor a block: {type(producer)}."
  )


def consumer_operands(result: ValueSite) -> Sequence[ValueSite]:
  """Given a result or an argument, returns the corresponding operands in its consumers."""
  assert result.type in (VariableType.RESULT, VariableType.ARGUMENT)
  results: list[ValueSite] = []
  # The layout can also be chosen from the layout of the consumers of the
  # results.
  for use in result.value.uses:
    consumer = use.owner
    index = use.operand_number
    results.append(ValueSite(consumer, VariableType.OPERAND, index))
  return results


def derive_relayout_constraints(
    value_sites_for_variable: ValueSitesForVariable,
) -> list[cs.Relayout]:
  """Derives relayout constraints from the given variable mapping."""
  constraints: list[cs.Relayout] = []
  variable_for_value_site: dict[ValueSite, cs.Variable] = {}
  for variable, value_sites in value_sites_for_variable.items():
    for value_site in value_sites:
      if value_site in variable_for_value_site:
        raise ValueError(
            f"{value_site} is mapped to both {variable} and "
            f"{variable_for_value_site[value_site]}"
        )
    variable_for_value_site |= {k: variable for k in value_sites}

  visited: set[cs.Variable] = set()
  for variable, value_sites in value_sites_for_variable.items():
    producers: list[cs.Variable] = []
    consumers: list[cs.Variable] = []
    for value_site in value_sites:
      # We can only relayout variables that are in registers.
      if value_site.memory_space != MemorySpace.REG:
        continue

      elt_bitwidth = utils.bitwidth(value_site.value.type.element_type)  # pytype: disable=attribute-error
      if value_site.type == VariableType.OPERAND:
        pr = producer_result(value_site)
        producer_variable = variable_for_value_site[pr]
        producers.append(producer_variable)
        # Only add the constraint if we haven't already created that constraint
        # when processing this variable as one of the producer's consumers.
        if producer_variable not in visited:
          # The producer of a variable must be relayout-able to the variable.
          constraints.append(
              cs.Relayout(producer_variable, variable, elt_bitwidth)
          )
      elif value_site.type in (VariableType.RESULT, VariableType.ARGUMENT):
        for co in consumer_operands(value_site):
          consumer_variable = variable_for_value_site[co]
          consumers.append(consumer_variable)
          # Only add the constraint if we haven't already created that
          # constraint when processing this variable as the consumer's producer.
          if consumer_variable not in visited:
            # A variable must be relayout-able to its consumers.
            constraints.append(
                cs.Relayout(variable, consumer_variable, elt_bitwidth)
            )
    visited.add(variable)
  return constraints


def is_terminator(op: ir.OpView) -> bool:
  return isinstance(op, (scf.YieldOp, scf.ConditionOp))


def traverse_op(
    op: ir.OpView,
    callback: Callable[[ir.OpView], None],
):
  """Traverses the operation and applies the callback in pre-order fashion.

  Skips recursing into `mgpu.CustomPrimitiveOp`s, and assumes that the values
  iterated on are not being modified.
  """
  callback(op)
  # The block of a mosaic_gpu.custom_primitive op is already lowered so it
  # should not be traversed.
  if not isinstance(op, mgpu.CustomPrimitiveOp):
    for region in op.operation.regions:
      for block in region:
        for block_op in block.operations:
          traverse_op(block_op, callback)


def is_valid_register_layout_assignment(
    shape: tuple[int, ...], layout: fa.FragmentedLayout
) -> bool:
  match layout:
    case fa.WGStridedFragLayout() as strided_layout:
      return strided_layout.shape == shape
    case fa.WGSplatFragLayout() as splat_layout:
      return splat_layout.shape == shape
    case fa.TiledLayout(tiling=tiling):
      try:
        # `tiling.tile_shape` will raise if the shape is not tileable.
        _ = tiling.tile_shape(shape)
      except ValueError:
        return False
      return True
    case _:
      assert False, f"Unreachable {shape}, {layout}"


def is_valid_smem_layout_assignment(
    shape: tuple[int, ...], tiling: lc.TileTransform
) -> bool:
  try:
    # `tiling.transform_shape` will raise if the shape is not tileable.
    _ = tiling.transform_shape(shape)
  except ValueError:
    return False
  return True


def is_valid_tmem_layout_assignment(
    shape: tuple[int, ...], layout: tcgen05.TMEMLayout
) -> bool:
  try:
    # `layout.tiling.tile_shape` will raise if the shape is not tileable.
    _ = layout.tiling.tile_shape(shape)
  except ValueError:
    return False
  return True


def check_layout_assignment(v: ValueSite, layout: cs.Constant) -> None:
  """Raises if the given layout can not be assigned to the given `ValueSite`."""
  match v.memory_space, layout:
    case MemorySpace.REG, cs.RegisterLayout(value=reg_layout):
      if not is_valid_register_layout_assignment(v.shape, reg_layout):
        raise ValueError(
            f"Layout {reg_layout} is not compatible with register variable "
            f"{v.value}. This is a bug."
        )
    case MemorySpace.TMEM, cs.TMEMLayout(value=tmem_layout):
      if not is_valid_tmem_layout_assignment(v.shape, tmem_layout):
        raise ValueError(
            f"Layout {tmem_layout} is not compatible with TMEM variable "
            f"{v.value}. This is a bug."
        )
    case MemorySpace.SMEM, cs.SMEMTiling(value=tiling_or_none):
      if tiling_or_none is None:
        return
      if not is_valid_smem_layout_assignment(v.shape, tiling_or_none):
        raise ValueError(
            f"Layout {tiling_or_none} is not compatible with SMEM variable "
            f"{v.value}. This is a bug."
        )
    case _:
      raise ValueError(
          f"Variable {v.value} in memory space {v.memory_space} should not be "
          f"assigned a layout of type {type(layout)}. This is a bug."
      )


def infer_layout(
    module: ir.Module, *, fuel: int = _DEFAULT_LAYOUT_INFERENCE_FUEL
):
  """Infers layouts for the given module.

  * If there are vector (respectively SMEM refs, TMEM refs) operands,
  `in_layouts` (respectively `in_transforms`, `in_tmem_layouts`) will be set and
  contain one element per relevant argument in the memory space.
  * If there are vector (respectively SMEM refs, TMEM refs) outputs,
  `out_layouts` (respectively `out_transforms`, `out_tmem_layouts`) will be set
  and contain one element per relevant argument in the memory space.
  * Any of these attributes is guaranteed to not be set if there is no relevant
  input/output in the corresponding memory space.

  The fuel is provided in order to limit the number of attempts made by the
  solver.
  """
  global_constraint_system: cs.ConstraintSystem | cs.Unsatisfiable
  global_constraint_system = cs.ConstraintSystem()
  ctx = DerivationContext()

  def gather_constraints(op: ir.Operation):
    # Terminator ops are handled directly by the op whose region they belong to.
    # This is because they need to be in sync with their parent op's inputs and
    # outputs---and the parent op's constraints therefore need to take them into
    # account.
    if is_terminator(op):
      return
    should_have_layout = (
        inference_utils.should_have_layout(op)
        or inference_utils.should_have_tmem_layout(op)
        or inference_utils.should_have_transforms(op)
    )
    if not should_have_layout:
      return
    rule = _constraint_system_derivation_rules.get(op.OPERATION_NAME, None)  # pytype: disable=attribute-error
    if rule is None:
      raise NotImplementedError(f"No layout inference rule defined for {op}")
    rule_result = rule(ctx, op)
    nonlocal global_constraint_system
    if isinstance(rule_result, cs.Unsatisfiable):
      global_constraint_system = cs.Unsatisfiable()
      return
    constraint_system, mapping = rule_result
    global_constraint_system &= constraint_system
    ctx.update(mapping)

  for op in module.body:
    traverse_op(op, gather_constraints)
    # Short-circuit if we have an unsatisfiable constraint system, we won't
    # construct anything useful anymore.
    if isinstance(global_constraint_system, cs.Unsatisfiable):
      break

  if isinstance(global_constraint_system, cs.Unsatisfiable):
    raise ValueError(
        "Failed to infer a possible set of layouts. This should only happen if "
        "user-provided layout casts are unsatisfiable."
    )

  constraints = derive_relayout_constraints(ctx.value_sites_for_variable)
  global_constraint_system &= cs.ConstraintSystem(constraints=constraints)
  assert not isinstance(global_constraint_system, cs.Unsatisfiable)

  # Add additional (redundant) constraints which helps the search converge
  # faster.
  global_constraint_system = cs.saturate_distinct_from_splat(
      global_constraint_system
  )
  assert not isinstance(global_constraint_system, cs.Unsatisfiable)
  global_constraint_system = cs.saturate_divides_constraints_for_equal_vars(
      global_constraint_system
  )

  # Attempt to find assignments that satisfy the constraint system.
  solution, remaining_fuel = find_assignments_for(
      list(ctx.value_sites_for_variable.keys()),
      global_constraint_system,
      fuel=fuel,
  )

  if logging.vlog_is_on(1):
    print("Finding a solution (or exhausting the entire search space) "
          f"consumed {fuel - remaining_fuel}/{fuel} fuel.")

  if isinstance(solution, cs.Unsatisfiable):
    raise ValueError(
        "Failed to infer a possible set of layouts. This should only happen if "
        "user-provided layout casts are unsatisfiable."
    )

  layout_for_value_site: dict[ValueSite, cs.Constant] = {}
  for variable, value_sites in ctx.value_sites_for_variable.items():
    for value_site in value_sites:
      layout = solution[variable]
      # Ensure that the layout assignment is valid for the value site. This
      # should only ever fail if our implementation is buggy.
      check_layout_assignment(value_site, layout)
      layout_for_value_site[value_site] = layout

  # Assigns the layouts that we found to the ops.
  assign_layouts(layout_for_value_site)

  # Sanity check: ensure that all ops have the right number of in/out layouts.
  for op in module.body:
    traverse_op(op, _ensure_all_layouts_are_set)
