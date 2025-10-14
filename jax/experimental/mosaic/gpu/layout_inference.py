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

from collections.abc import Callable, Iterator, Sequence, Set
import dataclasses
import enum
from functools import partial
import itertools
import re
from typing import Any, assert_never, cast

from jax._src.lib import mosaic_gpu_dialect as mgpu  # noqa: F401
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import gpu
from jax._src.lib.mlir.dialects import math as mlir_math
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector

from . import equations as eqns
from . import fragmented_array as fa
from . import inference_utils
from . import launch_context as lc
from . import layouts as layouts_lib
from . import tcgen05
from . import utils
import numpy as np


class VariableType(enum.IntEnum):
  """The type of a variable.

  Variables here are either operands or results of MLIR operations.
  """
  OPERAND = 0
  RESULT = 1


class MemorySpace(enum.Enum):
  """The memory space of a variable."""
  REG = enum.auto()
  SMEM = enum.auto()
  TMEM = enum.auto()


_op_name_regex = re.compile(r"^(%\d+ = )?\S+")

@dataclasses.dataclass(frozen=True)
class OperandOrResult:
  """A unique identifier for a variable."""
  # A MLIR operation.
  operation: ir.OpView
  # Whether this represents an operand or a result.
  type: VariableType
  # The index of the operand/result within the op's operands/results.
  index: int

  @property
  def value(self) -> ir.Value:
    """Returns the IR value corresponding to this operand or result."""
    if self.type == VariableType.OPERAND:
      return self.operation.operands[self.index]
    else:
      return self.operation.results[self.index]

  @property
  def memory_space(self) -> MemorySpace:
    """Returns the memory space associated with this operand or result."""
    type = self.value.type
    if ir.VectorType.isinstance(type):
      return MemorySpace.REG
    assert ir.MemRefType.isinstance(type)
    if utils.is_tmem_ref(type):
      return MemorySpace.TMEM
    elif utils.is_smem_ref(type):
      return MemorySpace.SMEM
    raise ValueError(f"Unsupported memory space for: {type}")

  def __str__(self):
    match = _op_name_regex.match(str(self.operation))
    assert match is not None
    if self.type == VariableType.OPERAND:
      return f"{match.group(0)}:o-{self.index}"
    else:
      return f"{match.group(0)}:r-{self.index}"


@dataclasses.dataclass(frozen=True)
class Hint:
  """Hints are used to model propagation of layouts across operations.

  Since using `relayout`s is always an option in principle, propagation across
  ops can not rely only on an equation system. Instead, we introduce hints as
  an equation-like form of "soft constraints", i.e., it suggests that
  `variable` should be equal to `expression`.
  """
  variable: eqns.Variable
  expression: eqns.Expression

  def __str__(self):
    return f"{self.variable} ?= {self.expression}"


def extract_constant_from_least_replicated_expression_for_hint(
    expressions: tuple[eqns.Expression, ...],
) -> eqns.Constant | None:
  choices: list[eqns.Constant] = []
  for e in expressions:
    if (red := extract_constant_for_hint(e)) is not None:
      choices.append(red)

  if not choices:
    return None

  # We reduce the expression here in order to recover an unambiguous least
  # replicated layout if it exists.
  maybe_choice = eqns.reduce_expression(
      eqns.LeastReplicated(tuple(choices)), {}
  )

  if isinstance(maybe_choice, eqns.Unsatisfiable):
    # TODO(bchetioui): consider other choices.
    return choices[0]

  assert isinstance(maybe_choice, eqns.Constant)
  return maybe_choice


def extract_constant_from_most_replicated_expression_for_hint(
    expressions: tuple[eqns.Expression, ...],
) -> eqns.Constant | None:
  assert len(expressions) >= 1
  choices: list[eqns.Constant] = []
  for e in expressions:
    if (red := extract_constant_for_hint(e)) is not None:
      choices.append(red)

  if not choices:
    return None

  maybe_choice = eqns.reduce_expression(
      eqns.MostReplicated(tuple(choices)), {}
  )

  if isinstance(maybe_choice, eqns.Unsatisfiable):
    # TODO(bchetioui): consider other choices.
    return choices[0]

  assert isinstance(maybe_choice, eqns.Constant)
  return maybe_choice


def extract_constant_from_broadcast_in_dim_expression_for_hint(
    e: eqns.BroadcastInDim,
) -> eqns.RegisterLayout | None:
  if not isinstance(e.expression, eqns.RegisterLayout):
    return None

  reduced_layout = e.expression.value

  wgmma_tm, wgmma_tn = fa.WGMMA_LAYOUT.base_tile_shape
  # TODO(bchetioui): enable generators to handle TCGEN05 layout from WGMMA_COL.
  if reduced_layout == fa.WGMMA_COL_LAYOUT and e.axes == (1,) and e.shape[0] % wgmma_tm == 0:
    return eqns.RegisterLayout(fa.WGMMA_LAYOUT)

  if reduced_layout == fa.WGMMA_ROW_LAYOUT and e.axes == (0,) and e.shape[1] % wgmma_tn == 0:
    return eqns.RegisterLayout(fa.WGMMA_LAYOUT)

  tcgen05_tm, _ = fa.TCGEN05_LAYOUT.base_tile_shape
  if reduced_layout == fa.TCGEN05_ROW_LAYOUT and e.axes == (0,) and e.shape[0] % tcgen05_tm == 0:
    return eqns.RegisterLayout(fa.TCGEN05_LAYOUT)

  return None


def extract_constant_for_hint(e: eqns.Expression) -> eqns.Constant | None:
  """Attempts to extract a `ConstantExpression` from a `Hint`'s `Expression`.

  Returns `None` if no `ConstantExpression` could be reasonably extracted.
  """
  match e:
    case eqns.Constant():
      return e
    case eqns.LeastReplicated():
      return extract_constant_from_least_replicated_expression_for_hint(e.expressions)
    case eqns.MostReplicated():
      return extract_constant_from_most_replicated_expression_for_hint(e.expressions)
    case eqns.BroadcastInDim():
      return extract_constant_from_broadcast_in_dim_expression_for_hint(e)
    case eqns.Variable():
      return None
    case _:
      raise NotImplementedError(f"Unsupported expression type: {type(e)}")


def extract_variable_assignment_from_hint(
    hint: Hint,
) -> tuple[eqns.Variable, eqns.Constant] | None:
  """Attempts to extract a single variable assignment from a `Hint`."""
  # TODO(bchetioui): make this a generator. This will allow us to maybe extract
  # different assignments that satisfy a replication constraint in the case
  # where replicated expressions are incompatible and several extractions are
  # possible.
  red = extract_constant_for_hint(hint.expression)
  return (hint.variable, red) if red is not None else None


def reduce_hints(
    hints: Sequence[Hint], assignments: dict[eqns.Variable, eqns.Constant]
) -> list[Hint]:
  """Reduces a sequence of `Hint`s.

  We reduce the `Hint`s' expressions, drop `Unsatisfiable` hints, and drop
  `Hint`s pertaining to pre-existing assignments.
  """
  new_hints: list[Hint] = []
  for h in hints:
    if h.variable not in assignments:
      reduced_expression = eqns.reduce_expression(h.expression, assignments)
      if isinstance(reduced_expression, eqns.Unsatisfiable):
        continue
      new_hints.append(dataclasses.replace(h, expression=reduced_expression))

  return new_hints


def _strided_layout_for_variable(
    variable: eqns.Variable,
) -> fa.WGStridedFragLayout | None:
  """Returns a strided layout for the given variable.

  If the given variable cannot have a strided layout, returns `None`.
  """
  # TODO(bchetioui): should we make variables carry a shape as well, to make
  # things easier?
  type = variable.key.value.type
  assert ir.VectorType.isinstance(type)
  return fa.WGStridedFragLayout.from_shaped_type(type)


def _extract_tiling_candidate(
    divides: eqns.Divides, num_tiled_dims: int
) -> Iterator[tuple[eqns.Variable, eqns.Constant]]:
  if not isinstance(divides.expr, eqns.Variable):
    return
  if num_tiled_dims > len(divides.dimensions_to_tile):
    # TODO(b/447079781): Support this case, by just assuming 0 (no constraints).
    return

  if num_tiled_dims == 0:
    yield divides.expr, eqns.SMEMTiling(None)
    return

  static_tiling = []
  dynamic_tiling = []
  for dim in divides.dimensions_to_tile[-num_tiled_dims:]:
    assert isinstance(dim[0], int)
    static_tiling.append(dim[0])
    # Any remaining values are dynamic. Use 1 for those to be safe.
    dynamic_tiling.append(dim[0] if len(dim) == 1 else 1)

  # The static tiling ignores dynamic values. Try this first as it yields
  # larger tiles that are likely to have better performance.
  yield divides.expr, eqns.SMEMTiling(lc.TileTransform(tuple(static_tiling)))

  if dynamic_tiling != static_tiling:
    # If that does not work, we could use a smaller, safe tiles by yielding
    # dynamic_tiling here. However, performance will be worse, and we choose to
    # return instead.
    return

def _extract_layout_candidates_from_memory_space_transfers(
    constraint: eqns.IsTransferable,
    divides_per_var: dict[eqns.Variable, list[eqns.Divides]],
) -> Iterator[tuple[eqns.Variable, eqns.Constant]]:
  """Attempts to extract variable assignments from a `Constraint`."""
  # This code assumes that the `IsTransferable` constraint is bidirectional.
  # This is currently true for TMEM <-> REG transfers and SMEM <-> REG
  # transfers.
  src, tgt = constraint.source, constraint.target
  match src, tgt:
    case eqns.Variable(), eqns.Constant():
      variable, constant = src, tgt
    case eqns.Constant(), eqns.Variable():
      variable, constant = tgt, src
    case _:
      return

  assert isinstance(variable, eqns.Variable)  # Satisfy type checkers.
  if isinstance(constant, eqns.RegisterLayout):
    layout = constant.value
    if variable.key.memory_space == MemorySpace.TMEM:
      for packing in (1, 2, 4, 8):
        for tmem_layout, reg_layout in constraint.supported_tmem_transfers(
            packing
        ):
          if layout == reg_layout:
            yield variable, eqns.TMEMLayout(tmem_layout)
    elif variable.key.memory_space == MemorySpace.SMEM:
      if inference_utils.is_mma_layout(layout):
        tiling = _infer_tiling_for_mma_ref(
            variable.key.value.type,
            max_swizzle=mgpu.SwizzlingMode.k128ByteSwizzle
        )
        assert len(tiling) == 2
        divides = divides_per_var.get(variable, [])
        divides.append(eqns.Divides(variable, ((tiling[0],), (tiling[1],))))
        [divides] = eqns.merge_divides_constraints(divides)
        assert isinstance(divides, eqns.Divides)
        yield from _extract_tiling_candidate(divides, len(tiling))
      else:
        # An empty tiling is valid here but we don't yield it in order to
        # avoid duplicating the empty tiling yielded by the caller.
        return

  elif isinstance(constant, eqns.TMEMLayout):
    layout = constant.value
    packing = layout.vector_length
    for tmem_layout, reg_layout in constraint.supported_tmem_transfers(packing):
      if layout == tmem_layout:
        yield variable, eqns.RegisterLayout(reg_layout)


def _divides_per_var(
    constraints: Sequence[eqns.Constraint],
) -> dict[eqns.Variable, list[eqns.Divides]]:
  """Returns all Divides constraints per variable."""
  result: dict[eqns.Variable, list[eqns.Divides]] = {}
  for constraint in constraints:
    match constraint:
      case eqns.Divides(expr=expr) if isinstance(expr, eqns.Variable):
        result.setdefault(expr, []).append(constraint)
  return result


def _extract_variable_assignments_from_constraints(
    constraints: Sequence[eqns.Constraint],
) -> Iterator[tuple[eqns.Variable, eqns.Constant]]:
  """Attempts to extract variable assignments from all constraints."""
  dpv = _divides_per_var(constraints)
  for c in constraints:
    match c:
      case eqns.IsTransferable():
        yield from _extract_layout_candidates_from_memory_space_transfers(c, dpv)


def conjure_assignment(
    unknowns: Set[eqns.Variable],
    equation_system: eqns.EquationSystem,
    hints: Sequence[Hint],
) -> Iterator[tuple[eqns.Variable, eqns.Constant]]:
  """Attempts to conjure an assignment for an unknown variable."""
  # TODO(allanrenucci): We should be able to short-circuit the search here if
  # the constraint is not satisfiable.
  yield from _extract_variable_assignments_from_constraints(
      equation_system.constraints
  )

  for hint in hints:
    if (assignment := extract_variable_assignment_from_hint(hint)) is not None:
      yield assignment

  # Here, we have not managed to find an assignment for all the unknown
  # variables, and our hints have not proven sufficient to unblock us. We now
  # try to introduce new arbitrary (valid) assignments into the system, and
  # hope that they turn out to be compatible with the equation system.
  for variable in unknowns:
    if variable in equation_system.assignments:
      continue
    # Try to instantiate a single variable to a strided layout and see if it
    # reduces the system.
    if variable.key.memory_space == MemorySpace.REG:
      layout = _strided_layout_for_variable(variable)
      if layout is not None:
        yield variable, eqns.RegisterLayout(layout)
    elif variable.key.memory_space == MemorySpace.SMEM:
      yield variable, eqns.SMEMTiling(None)


def find_assignments_for(
    unknowns: Set[eqns.Variable],
    equation_system: eqns.EquationSystem,
    hints: Sequence[Hint],
) -> dict[eqns.Variable, eqns.Constant] | eqns.Unsatisfiable:
  """Attempts to find assignments that satisfy `equation_system` for `unknowns`.

  Args:
    unknowns: the set of variables that are unknown.
    equation_system: the equation system to satisfy.
    hints: a list of hints that may be used to introduce new assignments.

  Returns:
    - Unsatisfiable() if the equation system has unsatisfiable constraints.
    - A dictionary assigning all the unknown variables to `ConstantExpression`s
      such that the assignment satisfies the equation system otherwise.
  """
  equation_system = eqns.reduce(equation_system)
  if isinstance(equation_system, eqns.Unsatisfiable):
    return eqns.Unsatisfiable()

  remaining_unknowns = unknowns - equation_system.assignments.keys()
  # In this case, we have determined an assignment for all the unknown
  # variables. Return their respective assignment.
  if not remaining_unknowns:
    return {v: k for v, k in equation_system.assignments.items() if v in unknowns}

  # Reduce the expressions in the remaining hints based on the current
  # assignments, and eliminate hints that pertain to variables that already
  # have an assignment.
  hints = reduce_hints(hints, equation_system.assignments)

  # If unknowns remain and we have fully reduced the system, we may still
  # be able to make progress by extracting an assignment from a `Hint`. This
  # new assignment could make the system unsatisfiable, so we use a recursive
  # call to be able to backtrack if necessary.
  for assignment in conjure_assignment(remaining_unknowns, equation_system, hints):
    variable, expr = assignment
    new_equation_system = (
        eqns.EquationSystem(assignments={variable: expr}) & equation_system)
    if isinstance(new_equation_system, eqns.Unsatisfiable):
      # This assignment is not compatible with the equation system.
      continue
    solution = find_assignments_for(unknowns, new_equation_system, hints)
    if isinstance(solution, eqns.Unsatisfiable):
      # This assignment is not compatible with the equation system.
      continue
    return solution

  # TODO(bchetioui): should we have a way to give a useful dump to the user
  # here, perhaps indicating what to layout cast.
  return eqns.Unsatisfiable()


@dataclasses.dataclass()
class DerivationContext:
  """Holds context information used for deriving an equation system."""
  # TODO(b/447079781): Remove once SMEM transform inference is implemented.
  enable_smem_inference : bool = False
  # A map of `OperandOrResult` to the variable that it is associated with.
  variable_for_operand_or_result: dict[OperandOrResult, eqns.Variable] = (
      dataclasses.field(default_factory=dict, init=False)
  )
  # A map of `eqns.Variable` to all the `OperandOrResult`s that it is associated
  # with.
  operand_and_results_for_variable: OperandOrResultsForVariable = (
      dataclasses.field(default_factory=dict, init=False)
  )

  def update(self, mapping: OperandOrResultsForVariable) -> None:
    for variable, operand_and_results in mapping.items():
      if variable in self.operand_and_results_for_variable:
        self.operand_and_results_for_variable[variable].extend(operand_and_results)
      else:
        self.operand_and_results_for_variable[variable] = operand_and_results
      for operand_or_result in operand_and_results:
        assert operand_or_result not in self.variable_for_operand_or_result
        self.variable_for_operand_or_result[operand_or_result] = variable

  def producer_ref(self, operand: OperandOrResult) -> eqns.Variable:
    """Returns the producer reference variable for the given operand."""
    return self.variable_for_operand_or_result[producer_result(operand)]


OperandOrResultsForVariable = dict[eqns.Variable, list[OperandOrResult]]

# An equation system derivation rule is a function that takes an MLIR operation
# and returns an equation system, a mapping from variables to operand/result
# identifiers, and a list of hints.
#
# The intended meaning of the mapping is that, for each identifier in the list
# keyed by a given variable, the MLIR operand/result corresponding to that
# identifier has the same layout as the variable.
#
# An `EquationSystemDerivationRule` must return a mapping such that the
# identifier corresponding to each operand/result must appear in the mapping,
# and each identifier in the mapping must be keyed by exactly one variable.
# Lastly, the mapping must only refer to variables and operands/results that
# correspond to the given operation.
EquationSystemDerivationRule = Callable[
    [DerivationContext, ir.OpView],
    tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]],
]
_equation_system_derivation_rules: dict[str, EquationSystemDerivationRule] = {}


def _add_equation_system_derivation_rule(op: type[ir.OpView]):
  def wrapper(rule: EquationSystemDerivationRule):
    if op is not None:
      _equation_system_derivation_rules[op.OPERATION_NAME] = rule  # pytype: disable=attribute-error
    return rule
  return wrapper


def is_vector(v: ir.Value) -> bool:
  return ir.VectorType.isinstance(v.type)


def _is_smem_ref(v: ir.Value) -> bool:
  return ir.MemRefType.isinstance(v.type) and utils.is_smem_ref(v)


def _is_tmem_ref(v: ir.Value) -> bool:
  return ir.MemRefType.isinstance(v.type) and utils.is_tmem_ref(v)


def _pointwise_op_equation_system(
    ctx: DerivationContext,
    op: ir.OpView,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx
  all_operands_and_results = vector_operands_and_results(op)
  variable = eqns.Variable(all_operands_and_results[0])
  return eqns.EquationSystem(), {variable: all_operands_and_results}, []


for op in [
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
    mlir_math.ExpOp,
    mlir_math.Exp2Op,
    mlir_math.LogOp,
    mlir_math.RsqrtOp,
    mlir_math.TanhOp,
]:
  _add_equation_system_derivation_rule(op)(_pointwise_op_equation_system)


def _divides_constraint_from_indices(
    var: eqns.Variable,
    indices: Sequence[ir.Value],
) -> eqns.Divides:
  """Returns a Divides constraint from the given load/store indices."""
  dimensions_to_tile : list[tuple[int | ir.Value, ...]] = []
  for i in indices:
    index_defining_op = i.owner.opview
    if isinstance(index_defining_op, arith.ConstantOp):
      dimensions_to_tile.append((index_defining_op.literal_value,))
    else:
      dimensions_to_tile.append((i,))
  return eqns.Divides(var, tuple(dimensions_to_tile))


@_add_equation_system_derivation_rule(vector.LoadOp)
def _vector_load_equation_system(
    ctx: DerivationContext,
    op: vector.LoadOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  # TODO(b/447079781): Investigate whether we should check for contiguous
  # strides here. An initial implementaiton of this failed the
  # test_gmem_to_smem_with_multiple_smem_indexers_and_transforms test, but
  # we should confirm that this is properly supported.

  # Registers
  dest = OperandOrResult(op, VariableType.RESULT, 0)
  dest_var = eqns.Variable(dest)
  operand_or_results_for_variable = {dest_var: [dest]}
  constraints = [
      eqns.Distinct(
          dest_var,
          eqns.RegisterLayout(
              fa.WGSplatFragLayout(shape=tuple(ir.ShapedType(op.result.type).shape))
          ),
      ),
  ]

  # SMEM
  if ctx.enable_smem_inference and utils.is_smem_ref(op.base):
    source = OperandOrResult(op, VariableType.OPERAND, 0)
    source_var = ctx.producer_ref(source)
    operand_or_results_for_variable[source_var] = [source]
    constraints.extend([
        eqns.IsTransferable(
            source=source_var,
            target=dest_var,
            shape=tuple(ir.ShapedType(op.result.type).shape),
        ),
        _divides_constraint_from_indices(source_var, op.indices),
    ])

  system = eqns.EquationSystem(constraints=constraints)
  return system, operand_or_results_for_variable, []


@_add_equation_system_derivation_rule(vector.StoreOp)
def _vector_store_equation_system(
    ctx: DerivationContext,
    op: vector.StoreOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  # TODO(b/447079781): Investigate whether we should check for contiguous
  # strides here. An initial implementaiton of this failed the
  # test_gmem_to_smem_with_multiple_smem_indexers_and_transforms test, but
  # we should confirm that this is properly supported.

  # Registers
  value = OperandOrResult(op, VariableType.OPERAND, 0)
  value_var = eqns.Variable(value)
  operand_or_results_for_variable = {value_var: [value]}

  # SMEM
  constraints = []
  if ctx.enable_smem_inference and utils.is_smem_ref(op.base):
    dest = OperandOrResult(op, VariableType.OPERAND, 1)
    dest_var = ctx.producer_ref(dest)
    operand_or_results_for_variable[dest_var] = [dest]
    constraints = [
        eqns.IsTransferable(
            source=value_var,
            target=dest_var,
            shape=tuple(ir.ShapedType(op.base.type).shape),
        ),
        _divides_constraint_from_indices(dest_var, op.indices),
    ]

  system = eqns.EquationSystem(constraints=constraints)
  return system, operand_or_results_for_variable, []


@_add_equation_system_derivation_rule(mgpu.OptimizationBarrierOp)
def _optimization_barrier_equation_system(
    ctx: DerivationContext,
    op: ir.OpView,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx
  operand_or_results_for_variable: OperandOrResultsForVariable = {}

  for i, operand in enumerate(op.operands):
    if not is_vector(operand):
      continue
    variable = eqns.Variable(OperandOrResult(op, VariableType.OPERAND, i))
    operand_or_results_for_variable[variable] = [
        OperandOrResult(op, VariableType.OPERAND, i),
        OperandOrResult(op, VariableType.RESULT, i)
    ]

  return eqns.EquationSystem(), operand_or_results_for_variable, []


@_add_equation_system_derivation_rule(vector.BroadcastOp)
def _vector_splat_equation_system(
    ctx: DerivationContext,
    op: ir.OpView,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx
  result = OperandOrResult(op, VariableType.RESULT, 0)
  variable = eqns.Variable(result)
  layout = fa.WGSplatFragLayout(tuple(cast(ir.ShapedType, op.result.type).shape))
  system = eqns.EquationSystem(
      assignments={variable: eqns.RegisterLayout(layout)}
  )
  return system, {variable: [result]}, []


@_add_equation_system_derivation_rule(arith.ConstantOp)
def _constant_equation_system(
    ctx: DerivationContext,
    constant_op: arith.ConstantOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx
  value = constant_op.value
  result = OperandOrResult(constant_op, VariableType.RESULT, 0)
  variable = eqns.Variable(result)
  shape = tuple(ir.ShapedType(constant_op.result.type).shape)
  if (
      ir.DenseElementsAttr.isinstance(value)
      and ir.DenseElementsAttr(value).is_splat
  ):
    layout = fa.WGSplatFragLayout(shape=shape)
    system = eqns.EquationSystem(
        assignments={variable: eqns.RegisterLayout(layout)}
    )
  else:
    constant_is_not_splat = eqns.Distinct(
        variable,
        eqns.RegisterLayout(fa.WGSplatFragLayout(shape=shape)),
    )
    system = eqns.EquationSystem(constraints=[constant_is_not_splat])

  return system, {variable: [result]}, []


def _terminator(
    block: ir.Block, expected_terminator: type[ir.OpView]
) -> ir.OpView:
  """Returns the terminator of the given block.

  Checks that the terminator is of the expected type.
  """
  terminator = block.operations[len(block.operations) - 1]
  assert isinstance(terminator, expected_terminator)
  return terminator.opview


@_add_equation_system_derivation_rule(scf.ForOp)
def _for_equation_system(
    ctx: DerivationContext,
    op: scf.ForOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  [block] = op.region.blocks
  yield_op = _terminator(block, scf.YieldOp)
  operand_or_results_for_variable: OperandOrResultsForVariable = {}

  # Account for the lower bound, upper bound, and step of the loop, which appear
  # in the operands but not in the results.
  num_leading_args = 3
  for index, o in enumerate(op.operands):
    if not is_vector(o) and not (ctx.enable_smem_inference and _is_smem_ref(o)):
      continue
    result_index = index - num_leading_args
    operand = OperandOrResult(op, VariableType.OPERAND, index)
    result = OperandOrResult(op, VariableType.RESULT, result_index)
    yield_operand = OperandOrResult(
        yield_op, VariableType.OPERAND, result_index
    )
    var = eqns.Variable(operand) if is_vector(o) else ctx.producer_ref(operand)
    operand_or_results_for_variable[var] = [operand, result, yield_operand]

  return eqns.EquationSystem(), operand_or_results_for_variable, []


@_add_equation_system_derivation_rule(scf.WhileOp)
def _while_equation_system(
    ctx: DerivationContext,
    op: scf.WhileOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx
  [before_block] = op.before.blocks
  [after_block] = op.after.blocks
  cond_op = _terminator(before_block, scf.ConditionOp)
  yield_op = _terminator(after_block, scf.YieldOp)

  operand_or_results_for_variable: OperandOrResultsForVariable = {}

  for operand_or_result in vector_operands_and_results(op):
    match operand_or_result.type:
      case VariableType.OPERAND:
        yield_operand = OperandOrResult(
            yield_op, VariableType.OPERAND, operand_or_result.index
        )
        operand_or_results_for_variable[eqns.Variable(operand_or_result)] = [
            operand_or_result,
            yield_operand,
        ]
      case VariableType.RESULT:
        # Increment by 1 to account for the conditional.
        cond_operand = OperandOrResult(
            cond_op, VariableType.OPERAND, operand_or_result.index + 1
        )
        operand_or_results_for_variable[eqns.Variable(operand_or_result)] = [
            operand_or_result,
            cond_operand,
        ]
      case _ as never:
        assert_never(never)

  return eqns.EquationSystem(), operand_or_results_for_variable, []


@_add_equation_system_derivation_rule(scf.IndexSwitchOp)
def _index_switch_equation_system(
    ctx: DerivationContext,
    op: scf.IndexSwitchOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx
  operand_or_results_for_variable: OperandOrResultsForVariable = {
      eqns.Variable(o): [o] for o in vector_operands_and_results(op)
  }
  for region in op.regions:
    [block] = region.blocks
    yield_op = _terminator(block, scf.YieldOp)
    for operand_or_result in operand_or_results_for_variable.keys():
      assert operand_or_result.key.type == VariableType.RESULT
      yield_operand = OperandOrResult(
          yield_op, VariableType.OPERAND, operand_or_result.key.index
      )
      operand_or_results_for_variable[operand_or_result].append(yield_operand)

  return eqns.EquationSystem(), operand_or_results_for_variable, []


@_add_equation_system_derivation_rule(mgpu.LayoutCastOp)
def _layout_cast_equation_system(
    ctx: DerivationContext,
    op: mgpu.LayoutCastOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx
  operand = OperandOrResult(op, VariableType.OPERAND, 0)
  result = OperandOrResult(op, VariableType.RESULT, 0)
  variable = eqns.Variable(operand)
  out_layout = eqns.RegisterLayout(layouts_lib.from_layout_attr(op.new_layout))
  return (
      eqns.EquationSystem(assignments={variable: out_layout}),
      {variable: [operand, result]},
      [],
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


def _infer_wgmma_tiling(
    a_type: ir.Type, b_type: ir.MemRefType
) -> tuple[tuple[int, int] | None, tuple[int, int]]:
  """Infers the tiling for a (if in SMEM) and b of a WGMMAOp.

  If both a and b are in SMEM, this function infers tilings that have matching
  swizzle values.
  """
  b_tiling = _infer_tiling_for_mma_ref(
      b_type, max_swizzle=mgpu.SwizzlingMode.k128ByteSwizzle
  )
  b_swizzle = _compute_swizzle(b_type, lc.TileTransform(b_tiling))
  if not ir.MemRefType.isinstance(a_type):
    return None, b_tiling

  a_tiling = _infer_tiling_for_mma_ref(
      cast(ir.MemRefType, a_type), max_swizzle=b_swizzle
  )
  a_swizzle = _compute_swizzle(a_type, lc.TileTransform(a_tiling))
  if a_swizzle != b_swizzle:
    # The swizzle for a and b has to match. This is not a fundamental
    # limitation, rather the lowering doesn't currently support it.
    b_tiling = _infer_tiling_for_mma_ref(b_type, max_swizzle=a_swizzle)
    b_swizzle = _compute_swizzle(b_type, lc.TileTransform(b_tiling))
    assert a_swizzle == b_swizzle
  return a_tiling, b_tiling


@_add_equation_system_derivation_rule(mgpu.WGMMAOp)
def _wgmma_equation_system(
    ctx: DerivationContext,
    op: mgpu.WGMMAOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  assignments: dict[eqns.Variable, eqns.Constant] = {}
  # Registers
  vector_operands_or_results = vector_operands_and_results(op)
  vec_variable = eqns.Variable(vector_operands_or_results[0])
  assignments[vec_variable] = eqns.RegisterLayout(fa.WGMMA_LAYOUT)
  operands_or_results_for_variable = {vec_variable: vector_operands_or_results}

  # SMEM
  if ctx.enable_smem_inference:
    a_tiling, b_tiling = _infer_wgmma_tiling(op.a.type, op.b.type)
    b = OperandOrResult(op, VariableType.OPERAND, 2)
    b_var = ctx.producer_ref(b)

    assignments[b_var] = eqns.SMEMTiling(lc.TileTransform(b_tiling))
    operands_or_results_for_variable[b_var] = [b]

    if a_tiling is not None:
      # a is in SMEM
      a = OperandOrResult(op, VariableType.OPERAND, 1)
      a_var = ctx.producer_ref(a)
      assignments[a_var] = eqns.SMEMTiling(lc.TileTransform(a_tiling))
      operands_or_results_for_variable[a_var] = [a]

  system = eqns.EquationSystem(
      assignments=assignments,
  )
  return system, operands_or_results_for_variable, []


@_add_equation_system_derivation_rule(vector.BroadcastOp)
def _vector_broadcast_equation_system(
    ctx: DerivationContext,
    op: vector.BroadcastOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx
  # This is not expected to be necessary at the moment. We should be using
  # mgpu.BroadcastInDimOp instead when dealing with broadcasting vectors.
  if ir.ShapedType.isinstance(op.source.type):
    raise NotImplementedError("Only vector broadcasts from scalars are supported.")
  out_variable = eqns.Variable(OperandOrResult(op, VariableType.RESULT, 0))
  layout = eqns.RegisterLayout(fa.WGSplatFragLayout(tuple(op.result.type.shape)))
  return (
      eqns.EquationSystem(assignments={out_variable: layout}),
      {out_variable: [out_variable.key]},
      []
  )


@_add_equation_system_derivation_rule(vector.ReductionOp)
def _vector_reduction_equation_system(
    ctx: DerivationContext,
    op: vector.ReductionOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx
  in_variable = eqns.Variable(OperandOrResult(op, VariableType.OPERAND, 0))
  return eqns.EquationSystem(), {in_variable: [in_variable.key]}, []


def _reduction_equation_and_hint(
    larger: eqns.Variable,
    smaller: eqns.Variable,
    larger_shape: tuple[int, ...],
    reduction_dims: tuple[int, ...]
) -> tuple[eqns.Equation, Hint]:
  reduce_expr = eqns.Reduce(larger, reduction_dims)
  # There are always many options for broadcasting a layout, so we can only
  # derive a broadcast hint in the out_variable -> source_variable direction.
  broadcast_dims = tuple(
      i for i in range(len(larger_shape)) if i not in reduction_dims
  )
  broadcast_expr = eqns.BroadcastInDim(smaller, broadcast_dims, larger_shape)
  broadcast_hint = Hint(variable=larger, expression=broadcast_expr)
  return eqns.Equation(lhs=smaller, rhs=reduce_expr), broadcast_hint


@_add_equation_system_derivation_rule(vector.MultiDimReductionOp)
def _multi_dim_reduction_equation_system(
    ctx: DerivationContext,
    op: vector.MultiDimReductionOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx
  source = OperandOrResult(op, VariableType.OPERAND, 0)
  acc = OperandOrResult(op, VariableType.OPERAND, 1)
  out = OperandOrResult(op, VariableType.RESULT, 0)
  source_variable = eqns.Variable(source)
  out_variable = eqns.Variable(out)

  reduction_equation, broadcast_hint = _reduction_equation_and_hint(
      source_variable, out_variable,
      tuple(ir.ShapedType(op.source.type).shape), tuple(op.reduction_dims)
  )
  # TODO(bchetioui): in the future, we may need to add rules that prevent
  # strided layouts from being chosen---since trying to reduce a strided layout
  # may cause us to raise an Exception at the moment.
  return (
      eqns.EquationSystem(equations=[reduction_equation]),
      {source_variable: [source], out_variable: [acc, out]},
      [broadcast_hint],
  )


@_add_equation_system_derivation_rule(mgpu.BroadcastInDimOp)
def _broadcast_in_dim_equation_system(
    ctx: DerivationContext,
    op: mgpu.BroadcastInDimOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx
  out_variable = eqns.Variable(OperandOrResult(op, VariableType.RESULT, 0))
  source_variable = eqns.Variable(OperandOrResult(op, VariableType.OPERAND, 0))
  out_shape = tuple(cast(ir.ShapedType, op.result.type).shape)
  reduction_dims = tuple(
      i for i in range(len(out_shape)) if i not in op.broadcast_dimensions
  )

  reduction_equation, broadcast_hint = _reduction_equation_and_hint(
      out_variable, source_variable, out_shape, reduction_dims
  )

  return (
      eqns.EquationSystem(equations=[reduction_equation]),
      {source_variable: [source_variable.key], out_variable: [out_variable.key]},
      [broadcast_hint],
  )


@_add_equation_system_derivation_rule(vector.ShapeCastOp)
def _shape_cast_equation_system(
    ctx: DerivationContext, op: vector.ShapeCastOp
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx
  in_shape = tuple(cast(ir.ShapedType, op.source.type).shape)
  out_shape = tuple(cast(ir.ShapedType, op.result.type).shape)

  in_variable = eqns.Variable(OperandOrResult(op, VariableType.OPERAND, 0))
  out_variable = eqns.Variable(OperandOrResult(op, VariableType.RESULT, 0))

  # Here, we are in a case where we are stating
  #
  #   out_variable = reshape(in_variable, in_shape, out_shape).
  #
  # Thanks to the symmetric property of reshape, we can also issue an equation
  # in the other direction, i.e.
  #
  #   in_variable = reshape(out_variable, out_shape, in_shape)
  #
  # in order to be able to figure out an assignment for `in_variable`. if we
  # happen to know `out_variable`. If we only issue the first equation, then
  # we will not be able to figure out an assignment for `in_variable` if we
  # only know `out_variable`, even though their relationship is fully
  # determined.
  in_to_out = eqns.Reshape(
      in_variable, source_shape=in_shape, target_shape=out_shape
  )
  out_to_in = eqns.Reshape(
      out_variable, source_shape=out_shape, target_shape=in_shape
  )

  return (
      eqns.EquationSystem(
          equations=[
              eqns.Equation(lhs=out_variable, rhs=in_to_out),
              eqns.Equation(lhs=in_variable, rhs=out_to_in),
          ],
      ),
      {in_variable: [in_variable.key], out_variable: [out_variable.key]},
      [],
  )


@_add_equation_system_derivation_rule(mgpu.CustomPrimitiveOp)
def _custom_primitive_equation_system(
    ctx: DerivationContext,
    op: mgpu.CustomPrimitiveOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  assignments: dict[eqns.Variable, eqns.Constant] = {}
  equations: list[eqns.Equation] = []
  in_layouts = iter(op.in_layouts)
  in_transforms = iter(op.in_transforms)
  variables: list[eqns.Variable] = []
  for i, operand in enumerate(op.operands):
    if is_vector(operand):
      v = eqns.Variable(OperandOrResult(op, VariableType.OPERAND, i))
      variables.append(v)
      assignments[v] = eqns.RegisterLayout(
          layouts_lib.from_layout_attr(next(in_layouts))
      )
    elif ctx.enable_smem_inference and _is_smem_ref(operand):
      # Here we need to create a new variable, even though it is equal to the
      # source operand. This is because we directly assign the new variable and
      # if we did that to the source there could be conflicting assignments.
      # For example, the same ref could be passed into the custom op twice with
      # different transforms, which needs to yield an unsatisfiable system.
      #
      # TODO(b/447079781): Consider creating the final Equation system using
      # __and__ and potentially returning Unsatisfiable() directly if there is
      # a conflict between the assignments.
      operand_or_result = OperandOrResult(op, VariableType.OPERAND, i)
      source_var = ctx.producer_ref(operand_or_result)
      v = eqns.Variable(operand_or_result)
      equations.append(eqns.Equation(lhs=source_var, rhs=v))
      variables.append(v)
      transforms = next(in_transforms)
      ref_ty = operand_or_result.value.type
      tiling = _extract_smem_tiling_from_custom_transform_attrs(ref_ty, transforms)
      assignments[v] = tiling

  out_layouts = iter(op.out_layouts)
  for i, result in enumerate(op.results):
    if ir.VectorType.isinstance(result.type):
      v = eqns.Variable(OperandOrResult(op, VariableType.RESULT, i))
      variables.append(v)
      assignments[v] = eqns.RegisterLayout(
          layouts_lib.from_layout_attr(next(out_layouts))
      )
  return (
      eqns.EquationSystem(equations=equations, assignments=assignments),
      {v: [v.key] for v in variables},
      [],
  )


def _tmem_layout_from_layout_attr(
    layout_attr: mgpu.TiledLayout,
) -> tcgen05.TMEMLayout:
  layout = layouts_lib.from_layout_attr(layout_attr)
  assert isinstance(layout, fa.TiledLayout)
  return tcgen05.TMEMLayout(
      layout.tiling, layout.warp_dims, layout.lane_dims, layout.vector_dim
  )


@_add_equation_system_derivation_rule(mgpu.TmemLayoutCastOp)
def _tmem_layout_cast_equation_system(
    ctx: DerivationContext,
    op: mgpu.TmemLayoutCastOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  operand = OperandOrResult(op, VariableType.OPERAND, 0)
  variable = ctx.producer_ref(operand)
  result = OperandOrResult(op, VariableType.RESULT, 0)
  out_layout = eqns.TMEMLayout(_tmem_layout_from_layout_attr(op.new_layout))
  return (
      eqns.EquationSystem(assignments={variable: out_layout}),
      {variable: [operand, result]},
      [],
  )


@_add_equation_system_derivation_rule(mgpu.TmemAllocOp)
def _tmem_alloc_equation_system(
    ctx: DerivationContext,
    op: mgpu.TmemAllocOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx
  result = OperandOrResult(op, VariableType.RESULT, 0)
  result_var = eqns.Variable(result)
  layout = tcgen05._infer_tmem_layout(
      tuple(op.result.type.shape), op.collective, packing=1
  )

  in_smem = OperandOrResult(op, VariableType.OPERAND, 0)
  in_smem_var = eqns.Variable(in_smem)
  assignments: dict[eqns.Variable, eqns.Constant] = {
      in_smem_var: eqns.SMEMTiling(None)
  }
  operands_for_variable = {result_var: [result], in_smem_var: [in_smem]}

  # This is a hint, not a hard constraint. This will be the default layout if
  # none can be inferred.
  hint = Hint(result_var, eqns.TMEMLayout(layout))
  system = eqns.EquationSystem(assignments=assignments)
  return system, operands_for_variable, [hint]


@_add_equation_system_derivation_rule(mgpu.TmemDeallocOp)
def _tmem_dealloc_equation_system(
    ctx: DerivationContext,
    op: mgpu.TmemDeallocOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  operand = OperandOrResult(op, VariableType.OPERAND, 0)
  variable = ctx.producer_ref(operand)
  return eqns.EquationSystem(), {variable: [operand]}, []


@_add_equation_system_derivation_rule(mgpu.TcGen05MMAOp)
def _tcgen05_mma_equation_system(
    ctx: DerivationContext,
    op: mgpu.TcGen05MMAOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  assignments: dict[eqns.Variable, eqns.Constant] = {}
  operands_for_variable: OperandOrResultsForVariable = {}

  # TMEM
  acc = OperandOrResult(op, VariableType.OPERAND, 0)
  acc_variable = ctx.producer_ref(acc)
  acc_type = ir.ShapedType(op.accumulator.type)
  acc_layout = tcgen05._infer_tmem_layout(
      tuple(acc_type.shape), op.collective, packing=1
  )
  assignments[acc_variable] = eqns.TMEMLayout(acc_layout)
  operands_for_variable[acc_variable] = [acc]

  if _is_tmem_ref(op.a):
    a = OperandOrResult(op, VariableType.OPERAND, 1)
    a_type = ir.ShapedType(op.a.type)
    a_var = ctx.producer_ref(a)
    packing = 32 // utils.bitwidth(a_type.element_type)
    a_layout = tcgen05._infer_tmem_layout(
        tuple(a_type.shape), op.collective, packing
    )
    assignments[a_var] = eqns.TMEMLayout(a_layout)
    operands_for_variable[a_var] = [a]

  # SMEM
  if ctx.enable_smem_inference:
    b_tiling = _infer_tiling_for_mma_ref(
        ir.MemRefType(op.b.type),
        max_swizzle=mgpu.SwizzlingMode.k128ByteSwizzle,
    )
    b = OperandOrResult(op, VariableType.OPERAND, 2)
    b_var = ctx.producer_ref(b)
    assignments[b_var] = eqns.SMEMTiling(lc.TileTransform(b_tiling))
    operands_for_variable[b_var] = [b]

    if _is_smem_ref(op.a):
      a_tiling = _infer_tiling_for_mma_ref(
          ir.MemRefType(op.a.type),
          max_swizzle=mgpu.SwizzlingMode.k128ByteSwizzle,
      )
      a = OperandOrResult(op, VariableType.OPERAND, 1)
      a_var = ctx.producer_ref(a)
      assignments[a_var] = eqns.SMEMTiling(lc.TileTransform(a_tiling))
      operands_for_variable[a_var] = [a]

  return eqns.EquationSystem(assignments=assignments), operands_for_variable, []


@_add_equation_system_derivation_rule(mgpu.AsyncLoadTmemOp)
def _async_load_tmem_equation_system(
    ctx: DerivationContext,
    op: mgpu.AsyncLoadTmemOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  source = OperandOrResult(op, VariableType.OPERAND, 0)
  source_variable = ctx.producer_ref(source)
  destination = OperandOrResult(op, VariableType.RESULT, 0)
  destination_variable = eqns.Variable(destination)
  constraint = eqns.IsTransferable(
      source_variable,
      destination_variable,
      tuple(ir.ShapedType(op.source.type).shape),
  )
  return (
      eqns.EquationSystem(constraints=[constraint]),
      {source_variable: [source], destination_variable: [destination]},
      [],
  )


@_add_equation_system_derivation_rule(mgpu.AsyncStoreTmemOp)
def _async_store_tmem_equation_system(
    ctx: DerivationContext,
    op: mgpu.AsyncStoreTmemOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  source = OperandOrResult(op, VariableType.OPERAND, 0)
  source_variable = eqns.Variable(source)
  destination = OperandOrResult(op, VariableType.OPERAND, 1)
  destination_variable = ctx.producer_ref(destination)
  constraint = eqns.IsTransferable(
      source_variable,
      destination_variable,
      tuple(ir.ShapedType(op.source.type).shape),
  )
  return (
      eqns.EquationSystem(constraints=[constraint]),
      {source_variable: [source], destination_variable: [destination]},
      [],
  )


@_add_equation_system_derivation_rule(mgpu.SliceSMEMOp)
def _slice_smem_equation_system(
    ctx: DerivationContext,
    op: mgpu.SliceSMEMOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx
  res = OperandOrResult(op, VariableType.RESULT, 0)
  res_var = eqns.Variable(res)
  return (eqns.EquationSystem(), {res_var: [res]}, [])


# TODO(b/447079781): Check whether we still need this rule. Normally,
# DynamicSharedMemory is only generated in the lowering pass. If there is
# another case where a ViewOp is generated beforehand, and this rule cannot be
# removed, document here what that case is.
@_add_equation_system_derivation_rule(memref.ViewOp)
def _memref_view_op_equation_system(
    ctx: DerivationContext,
    op: memref.ViewOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx

  # The source is expected to come from a DynamicSharedMemoryOp which does not
  # participate in layout inference and no variable exists for it.
  if not isinstance(op.source.owner.opview, gpu.DynamicSharedMemoryOp):
    raise NotImplementedError(
        "Memref view transforms are only inferred when the op is a direct user "
        f"of a DynamicSharedMemoryOp but got {op}."
    )

  res = OperandOrResult(op, VariableType.RESULT, 0)
  res_var = eqns.Variable(res)
  return eqns.EquationSystem(), {res_var: [res]}, []


# TODO(b/447079781): Check whether we should create new variables or use
# variables from the producer_ref in all memref rules. Ideally all of those
# rules should handles references the same way. If there is a reason for
# handling them differently, document in each rule.
#
# E.g. here, using the producer_ref variable would work for all existing tests,
# but we create a new variable + an equation in order to be consistent with the
# other memref rules.
@_add_equation_system_derivation_rule(memref.CastOp)
def _memref_cast_op_equation_system(
    ctx: DerivationContext,
    op: memref.CastOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  source = OperandOrResult(op, VariableType.OPERAND, 0)
  var_source = ctx.producer_ref(source)
  dest = OperandOrResult(op, VariableType.RESULT, 0)
  var_dest = eqns.Variable(dest)
  return (
      eqns.EquationSystem(equations=[eqns.Equation(var_source, var_dest)]),
      {var_source: [source], var_dest: [dest]},
      [],
  )


@_add_equation_system_derivation_rule(memref.TransposeOp)
def _memref_transpose_op_equation_system(
    ctx: DerivationContext,
    op: memref.TransposeOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  in_ty = ir.MemRefType(op.in_.type)
  if len(in_ty.shape) != 2:
    raise NotImplementedError(f"Only 2D memrefs are supported, got {in_ty}")
  in_strides, _ = in_ty.get_strides_and_offset()
  out_strides, _ = ir.MemRefType(op.result.type).get_strides_and_offset()
  transpose = in_strides != out_strides

  source = OperandOrResult(op, VariableType.OPERAND, 0)
  dest = OperandOrResult(op, VariableType.RESULT, 0)
  source_var = ctx.producer_ref(source)

  if not transpose:
    return (eqns.EquationSystem(), {source_var: [source, dest]}, [])

  dest_var = eqns.Variable(dest)
  equations = [
      eqns.Equation(source_var, eqns.Transpose(dest_var)),
      eqns.Equation(eqns.Transpose(source_var), dest_var),
  ]
  system = eqns.EquationSystem(equations=equations)
  return system, {source_var: [source], dest_var: [dest]}, []


# `memref.load` and `memref.store` are used to load barrier phases which are
# scalars---the rule needn't do anything interesting, but we need to have it.
@_add_equation_system_derivation_rule(memref.LoadOp)
@_add_equation_system_derivation_rule(memref.StoreOp)
def _memref_load_store_op_equation_system(
    ctx: DerivationContext,
    op: memref.LoadOp | memref.StoreOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  del ctx

  ref_shape = ir.MemRefType(op.memref.type).shape
  if ref_shape and ref_shape != [1]:
    raise NotImplementedError(
        f"Only scalar memrefs are supported, got {ref_shape}"
    )

  ref_op_index = 0 if isinstance(op, memref.LoadOp) else 1
  ref = OperandOrResult(op, VariableType.OPERAND, ref_op_index)
  var = eqns.Variable(ref)
  assignments: dict[eqns.Variable, eqns.Constant] = {var: eqns.SMEMTiling(None)}
  return eqns.EquationSystem(assignments=assignments), {var: [ref]}, []


def _extract_smem_tiling_from_custom_transform_attrs(
    ref_type: ir.MemRefType,
    transform_attrs: ir.ArrayAttr,
) -> eqns.SMEMTiling:
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

  return eqns.SMEMTiling(tile_transform)


@_add_equation_system_derivation_rule(mgpu.WithTransformsOp)
def _with_transforms_equation_system(
    ctx: DerivationContext,
    op: mgpu.WithTransformsOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  source = OperandOrResult(op, VariableType.OPERAND, 0)
  dest = OperandOrResult(op, VariableType.RESULT, 0)
  var = ctx.producer_ref(source)
  tiling = _extract_smem_tiling_from_custom_transform_attrs(op.ref.type, op.transforms)
  assignments: dict[eqns.Variable, eqns.Constant] = {var: tiling}
  return eqns.EquationSystem(assignments=assignments), {var: [source, dest]}, []


@_add_equation_system_derivation_rule(mgpu.AsyncLoadOp)
@_add_equation_system_derivation_rule(mgpu.AsyncStoreOp)
def _async_load_store_equation_system(
    ctx: DerivationContext,
    op: mgpu.AsyncLoadOp | mgpu.AsyncStoreOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  operand_index = 1 if isinstance(op, mgpu.AsyncLoadOp) else 0
  operand = OperandOrResult(op, VariableType.OPERAND, operand_index)
  var = ctx.producer_ref(operand)
  return eqns.EquationSystem(), {var: [operand]}, []


def _ensure_all_layouts_are_set(
    op: ir.OpView, enable_smem_inference: bool
) -> None:
  if inference_utils.should_have_layout(op):
    _ensure_right_number_of_layouts(
        op,
        inference_utils.in_layouts(op)
        if inference_utils.has_in_layouts_set(op)
        else [],
        inference_utils.out_layouts(op)
        if inference_utils.has_out_layouts_set(op)
        else [],
    )
  if inference_utils.should_have_tmem_layout(op):
    _ensure_right_number_of_tmem_layouts(
        op,
        inference_utils.in_tmem_layouts(op)
        if inference_utils.has_in_tmem_layouts_set(op)
        else [],
        inference_utils.out_tmem_layouts(op)
        if inference_utils.has_out_tmem_layouts_set(op)
        else [],
    )
  if enable_smem_inference and inference_utils.should_have_transforms(op):
    _ensure_right_number_of_transforms(
        op,
        inference_utils.in_transforms(op)
        if inference_utils.has_in_transforms_set(op)
        else [],
        inference_utils.out_transforms(op)
        if inference_utils.has_out_transforms_set(op)
        else [],
    )


def _ensure_right_number_of_layouts(
    op: ir.OpView,
    in_layouts: Sequence[fa.FragmentedLayout | ir.Attribute],
    out_layouts: Sequence[fa.FragmentedLayout | ir.Attribute],
) -> None:
  """Ensures that the right number of in/out layouts are provided for an op."""
  if len(in_layouts) != sum(map(is_vector, op.operands)):
    raise ValueError(
        "Expected the same number of in_layouts as vector operands."
    )
  if len(out_layouts) != sum(map(is_vector, op.results)):
    raise ValueError(
        "Expected the same number of out_layouts as vector results."
    )


def _ensure_right_number_of_tmem_layouts(
    op: ir.OpView,
    in_layouts: Sequence[ir.Attribute],
    out_layouts: Sequence[ir.Attribute],
) -> None:
  """Ensures that the right number of in/out TMEM layouts are provided for an op."""
  if len(in_layouts) != sum(map(_is_tmem_ref, op.operands)):
    raise ValueError(
        f"Expected the same number of in_tmem_layouts({in_layouts}) as TMEM ref operands. op=\n  {op}"
    )
  if len(out_layouts) != sum(map(_is_tmem_ref, op.results)):
    raise ValueError(
        f"Expected the same number of out_tmem_layouts({out_layouts}) as TMEM ref results. op=\n  {op}"
    )


def _ensure_right_number_of_transforms(
    op: ir.OpView,
    in_transforms: Sequence[Any],
    out_transforms: Sequence[Any],
) -> None:
  """Ensures that the right number of in/out SMEM transforms are provided for an op."""
  if len(in_transforms) != sum(
      map(inference_utils.is_transformable_smem_memref, op.operands)
  ):
    raise ValueError(
        f"Expected the same number of in_transforms({in_transforms}) as SMEM ref operands. op=\n  {op}"
    )
  if len(out_transforms) != sum(
      map(inference_utils.is_transformable_smem_memref, op.results)
  ):
    raise ValueError(
        f"Expected the same number of out_transforms({out_transforms}) as SMEM ref results. op=\n  {op}"
    )


def _compute_swizzle(
    type: ir.Type, tile_transform: lc.TileTransform | None
) -> mgpu.SwizzlingMode:
  """Computes the swizzle mode given a tiling transform and a data type."""
  if tile_transform is None:
    # TODO(b/447079781): Revisit if this is the behavior we want.
    return mgpu.SwizzlingMode.kNoSwizzle

  if not ir.MemRefType.isinstance(type):
    raise ValueError(f"Expected a MemRefType, got {type}.")
  ref_ty = ir.MemRefType(type)
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
  layout: eqns.Constant


def assign_layouts(
    solution: dict[OperandOrResult, eqns.Constant],
    enable_smem_inference: bool,
) -> None:
  """Assigns the layouts in `solution` to the MLIR ops they belong to.

  This function requires that, for each MLIR op that appears in `solution`,
  `solution` contains a layout assignment for all of its `vector`, TMEM, and
  SMEM operands and results.
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
        tl.layout.value for tl in in_tls if isinstance(tl.layout, eqns.RegisterLayout)
    ]
    out_layouts = [
        tl.layout.value for tl in out_tls if isinstance(tl.layout, eqns.RegisterLayout)
    ]
    in_tmem_layouts = [
        tl.layout.value for tl in in_tls if isinstance(tl.layout, eqns.TMEMLayout)
    ]
    out_tmem_layouts = [
        tl.layout.value for tl in out_tls if isinstance(tl.layout, eqns.TMEMLayout)
    ]
    in_transforms = [tl for tl in in_tls if isinstance(tl.layout, eqns.SMEMTiling)]
    out_transforms = [tl for tl in out_tls if isinstance(tl.layout, eqns.SMEMTiling)]

    _ensure_right_number_of_layouts(op, in_layouts, out_layouts)
    _ensure_right_number_of_tmem_layouts(op, in_tmem_layouts, out_tmem_layouts)
    if enable_smem_inference:
      _ensure_right_number_of_transforms(op, in_transforms, out_transforms)

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

    if enable_smem_inference:

      def _to_transform_attrs(
          transforms: list[_TypeAndLayout],
      ) -> list[ir.ArrayAttr]:
        all_attrs: list[ir.ArrayAttr] = []
        for tl in transforms:
          assert isinstance(tl.layout, eqns.SMEMTiling)  # make pytype happy
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


def vector_operands_and_results(op: ir.OpView) -> list[OperandOrResult]:
  """Returns all the vector operands and results for the given op."""
  operands_or_results = [
      OperandOrResult(op, VariableType.OPERAND, i)
      for i, o in enumerate(op.operands)
      if is_vector(o)
  ]
  operands_or_results.extend([
      OperandOrResult(op, VariableType.RESULT, i)
      for i, o in enumerate(op.results)
      if is_vector(o)
  ])
  return operands_or_results


def producer_result(operand: OperandOrResult) -> OperandOrResult:
  """Given an operand, returns the corresponding result in its producer.

  When the producer is a block, we return the corresponding operand in the
  operation that owns the block.
  """
  assert operand.type == VariableType.OPERAND
  value = operand.operation.operands[operand.index]
  producer = value.owner
  if isinstance(producer, ir.Operation):
    index = list(producer.results).index(value)
    return OperandOrResult(producer.opview, VariableType.RESULT, index)

  # Block case, useful for deriving layouts for ops
  # depending on function parameters, or loop block arguments.
  if isinstance(producer, ir.Block):
    index = list(cast(ir.Block, producer).arguments).index(value)
    if isinstance(producer.owner, scf.ForOp):
      # In this case, the block arguments are offset compared to the loop
      # operands. The loop operands have the lower bound, upper bound, and step
      # as their leading arguments. The block arguments omit these parameters,
      # but start with the iteration variable.
      num_leading_args = 3
      index += num_leading_args - 1
      return OperandOrResult(producer.owner.opview, VariableType.OPERAND, index)
    if isinstance(producer.owner, scf.WhileOp):
      [before_block] = producer.owner.before.blocks
      [after_block] = producer.owner.after.blocks
      if producer == before_block:
        # In this case, the block arguments correspond to the while operands.
        return OperandOrResult(producer.owner.opview, VariableType.OPERAND, index)
      else:
        assert producer == after_block
        # In this case, the block arguments correspond to the while results.
        return OperandOrResult(producer.owner.opview, VariableType.RESULT, index)
    raise NotImplementedError(
        f"Producer {producer} is not a ForOp, a WhileOp: {type(producer)}."
    )

  raise TypeError(
      f"Producer {producer} is not an operation nor a block: {type(producer)}."
  )


def consumer_operands(result: OperandOrResult) -> Sequence[OperandOrResult]:
  """Given a result, returns the corresponding operands in its consumers."""
  assert result.type == VariableType.RESULT
  consumer_operands: list[OperandOrResult] = []
  # The layout can also be chosen from the layout of the consumers of the
  # results.
  for use in cast(ir.OpResult, result.operation.results[result.index]).uses:
    consumer = use.owner.opview  # pytype: disable=attribute-error
    index = use.operand_number
    consumer_operands.append(OperandOrResult(consumer, VariableType.OPERAND, index))
  return consumer_operands


def derive_hints_and_constraints(
    operands_and_results_for_variable: OperandOrResultsForVariable
) -> tuple[list[Hint], list[eqns.Relayout]]:
  """Derives propagation hints from the given variable mapping."""
  hints: list[Hint] = []
  constraints: list[eqns.Relayout] = []
  variable_for_operand_or_result: dict[OperandOrResult, eqns.Variable] = {}
  for variable, operand_and_results in operands_and_results_for_variable.items():
    for operand_or_result in operand_and_results:
      if operand_or_result in variable_for_operand_or_result:
        raise ValueError(
            f"{operand_or_result} is mapped to both {variable} and "
            f"{variable_for_operand_or_result[operand_or_result]}"
        )
    variable_for_operand_or_result |= {k: variable for k in operand_and_results}

  visited: set[eqns.Variable] = set()
  for variable, operand_and_results in operands_and_results_for_variable.items():
    producers: list[eqns.Variable] = []
    consumers: list[eqns.Variable] = []
    for operand_or_result in operand_and_results:
      # We can only relayout variables that are in registers.
      if operand_or_result.memory_space != MemorySpace.REG:
        continue

      if operand_or_result.type == VariableType.OPERAND:
        pr = producer_result(operand_or_result)
        producer_variable = variable_for_operand_or_result[pr]
        producers.append(producer_variable)
        # Only add the constraint if we haven't already created that constraint
        # when processing this variable as one of the producer's consumers.
        if producer_variable not in visited:
          # The producer of a variable must be relayout-able to the variable.
          constraints.append(eqns.Relayout(producer_variable, variable))
      elif operand_or_result.type == VariableType.RESULT:
        for co in consumer_operands(operand_or_result):
          consumer_variable = variable_for_operand_or_result[co]
          consumers.append(consumer_variable)
          # Only add the constraint if we haven't already created that
          # constraint when processing this variable as the consumer's producer.
          if consumer_variable not in visited:
            # A variable must be relayout-able to its consumers.
            constraints.append(eqns.Relayout(variable, consumer_variable))
    visited.add(variable)

    if producers:
      least_replicated_producer = eqns.LeastReplicated(tuple(producers))
      hint_expr = eqns.MostReplicated((least_replicated_producer, *consumers))
      hints.append(Hint(variable, hint_expr))
    elif consumers:
      hint_expr = eqns.MostReplicated(tuple(consumers))
      hints.append(Hint(variable, hint_expr))

  return hints, constraints


def is_terminator(op: ir.OpView) -> bool:
  return isinstance(op, (scf.YieldOp, scf.ConditionOp))


def _drop_smem(
    system: eqns.EquationSystem, ctx: DerivationContext
) -> tuple[eqns.EquationSystem, DerivationContext]:
  """Drops SMEM related variables constraints and hints.

  This is only needed to enable the gradual implementation and testing of
  SMEM inference.
  TODO(b/447079781): Remove this function once SMEM inference is fully
  implemented.
  """
  def is_smem(
      x: (
          OperandOrResult
          | eqns.Constraint
          | eqns.Expression
          | eqns.Equation
      ),
  ) -> bool:
    match x:
      case OperandOrResult(memory_space=MemorySpace.SMEM):
        return True
      case eqns.Transpose():
        return True
      case eqns.IsTransferable(source=source, target=target):
        return is_smem(source) or is_smem(target)
      case eqns.Divides():
        return True
      case eqns.Variable(key=key):
        return is_smem(key)
      case eqns.SMEMTiling():
        return True
      case eqns.Equation(lhs=lhs, rhs=rhs):
        return is_smem(lhs) or is_smem(rhs)
      case _:
        return False

  assign = {k: v for k, v in system.assignments.items() if not is_smem(v)}
  equations = [e for e in system.equations if not is_smem(e)]
  const = [c for c in system.constraints if not is_smem(c)]

  new_ctx = DerivationContext()

  new_ctx.operand_and_results_for_variable = {
      k: v for k, v in ctx.operand_and_results_for_variable.items() if not is_smem(k)
  }

  new_ctx.variable_for_operand_or_result = {
      k: v for k, v in ctx.variable_for_operand_or_result.items() if not is_smem(k)
  }

  return eqns.EquationSystem(assign, equations, const), new_ctx


def infer_layout(module: ir.Module, enable_smem_inference: bool = False):
  """Infers layouts for the given module.

  * If there are vector (respectively SMEM refs, TMEM refs) operands,
  `in_layouts` (respectively `in_transforms`, `in_tmem_layouts`) will be set and
  contain one element per relevant argument in the memory space.
  * If there are vector (respectively SMEM refs, TMEM refs) outputs,
  `out_layouts` (respectively `out_transforms`, `out_tmem_layouts`) will be set
  and contain one element per relevant argument in the memory space.
  * Any of these attributes is guaranteed to not be set if there is no relevant
  input/output in the corresponding memory space.

  If `enable_smem_inference` is False, SMEM transforms are not inferred. This
  is only a temporary flag to allow for an incremental rollout of SMEM
  inference.
  TODO(b/447079781): Remove this flag once SMEM inference is fully implemented.
  """
  global_equation_system: eqns.EquationSystem | eqns.Unsatisfiable
  global_equation_system = eqns.EquationSystem()
  hints: list[Hint] = []
  ctx = DerivationContext(enable_smem_inference=enable_smem_inference)

  def gather_equations(op: ir.Operation):
    # Terminator ops are handled directly by the op whose region they belong to.
    # This is because they need to be in sync with their parent op's inputs and
    # outputs---and the parent op's equations therefore need to take them them
    # into account.
    if is_terminator(op):
      return
    should_have_layout = (
        inference_utils.should_have_layout(op)
        or inference_utils.should_have_tmem_layout(op)
        or (inference_utils.should_have_transforms(op) and enable_smem_inference)
    )
    if not should_have_layout:
      return
    rule = _equation_system_derivation_rules.get(op.OPERATION_NAME, None)  # pytype: disable=attribute-error
    if rule is None:
      raise NotImplementedError(f"No layout inference rule defined for {op}")
    equation_system, mapping, op_hints = rule(ctx, op)
    ctx.update(mapping)
    nonlocal global_equation_system
    global_equation_system &= equation_system
    hints.extend(op_hints)

  for op in module.body:
    inference_utils.traverse_op(op, gather_equations, pre_order=True)

  if isinstance(global_equation_system, eqns.Unsatisfiable):
    raise ValueError(
        "Failed to infer a possible set of layouts. This should only happen if "
        "user-provided layout casts are unsatisfiable."
    )

  if not enable_smem_inference:
    global_equation_system, ctx = _drop_smem(global_equation_system, ctx)

  propagation_hints, constraints = derive_hints_and_constraints(ctx.operand_and_results_for_variable)
  hints = reduce_hints(hints + propagation_hints, global_equation_system.assignments)  # pytype: disable=attribute-error
  global_equation_system &= eqns.EquationSystem(constraints=constraints)
  assert not isinstance(global_equation_system, eqns.Unsatisfiable)

  # Add additional (redundant) constraints which helps the search converge
  # faster.
  global_equation_system = eqns.saturate_distinct_from_splat(global_equation_system)
  assert not isinstance(global_equation_system, eqns.Unsatisfiable)
  global_equation_system = eqns.saturate_divides_constraints_for_equal_vars(global_equation_system)

  # Attempt to find assignments that satisfy the equation system.
  solution = find_assignments_for(
      ctx.operand_and_results_for_variable.keys(), global_equation_system, hints
  )

  if isinstance(solution, eqns.Unsatisfiable):
    raise ValueError(
        "Failed to infer a possible set of layouts. This should only happen if "
        "user-provided layout casts are unsatisfiable."
    )

  layout_for_operand_or_result = {
      k: solution[v]
      for v, ks in ctx.operand_and_results_for_variable.items()
      for k in ks
  }

  # Assigns the layouts that we found to the ops.
  assign_layouts(layout_for_operand_or_result, enable_smem_inference)

  # Sanity check: ensure that all ops have the right number of in/out layouts.
  for op in module.body:
    inference_utils.traverse_op(
        op,
        partial(
            _ensure_all_layouts_are_set,
            enable_smem_inference=enable_smem_inference,
        ),
    )
