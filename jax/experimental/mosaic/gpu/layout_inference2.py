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

from __future__ import annotations

from collections.abc import Callable, Sequence, Set
import dataclasses
import enum
import itertools
from typing import assert_never, cast

from jax._src.lib import mosaic_gpu_dialect as mgpu  # noqa: F401
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import math as mlir_math
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
import numpy as np

from . import equations as eqns
from . import fragmented_array as fa
from . import inference_utils
from . import layouts as layouts_lib
from . import utils


class VariableType(enum.IntEnum):
  """The type of a variable.

  Variables here are either operands or results of MLIR operations.
  """
  OPERAND = 0
  RESULT = 1

@dataclasses.dataclass(frozen=True)
class OperandOrResult:
  """A unique identifier for a variable."""
  # A MLIR operation.
  operation: ir.OpView
  # Whether this represents an operand or a result.
  type: VariableType
  # The index of the operand/result within the op's operands/results.
  index: int


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
    e: eqns.BroadcastInDim
) -> eqns.Constant | None:
  if not isinstance(e.expression, eqns.Constant):
    return None

  reduced_layout = e.expression.value

  wgmma_tm, wgmma_tn = fa.WGMMA_LAYOUT.base_tile_shape
  # TODO(bchetioui): enable generators to handle TCGEN05 layout from WGMMA_COL.
  if reduced_layout == fa.WGMMA_COL_LAYOUT and e.axes == (1,) and e.shape[0] % wgmma_tm == 0:
    return eqns.Constant(fa.WGMMA_LAYOUT)

  if reduced_layout == fa.WGMMA_ROW_LAYOUT and e.axes == (0,) and e.shape[1] % wgmma_tn == 0:
    return eqns.Constant(fa.WGMMA_LAYOUT)

  tcgen05_tm, _ = fa.TCGEN05_LAYOUT.base_tile_shape
  if reduced_layout == fa.TCGEN05_ROW_LAYOUT and e.axes == (0,) and e.shape[0] % tcgen05_tm == 0:
    return eqns.Constant(fa.TCGEN05_LAYOUT)

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
  #
  # Make a copy of the hints to allow deleting unnecessary hints from the list,
  # without modifying the list being iterated over.
  remaining_hints: list[Hint] = [*hints]
  for i, hint in reversed(list(enumerate(hints))):
    if (assignment := extract_variable_assignment_from_hint(hint)) is not None:
      variable, expr = assignment
      new_equation_system = (
          eqns.EquationSystem(assignments={variable: expr}) & equation_system)
      if isinstance(new_equation_system, eqns.Unsatisfiable):
        # This assignment is not compatible with the equation system.
        continue
      other_hints = remaining_hints[:i] + remaining_hints[i + 1:]
      solution = find_assignments_for(unknowns, new_equation_system, other_hints)
      if isinstance(solution, eqns.Unsatisfiable):
        # This assignment is not compatible with the equation system. We remove
        # it from the list of hints.
        remaining_hints.pop(i)
        continue
      return solution
  hints = remaining_hints

  # Here, we have not managed to find an assignment for all the unknown
  # variables, and our hints have not proven sufficient to unblock us. We now
  # try to introduce new arbitrary (valid) assignments into the system, and
  # hope that they turn out to be compatible with the equation system.
  for variable in unknowns:
    if variable in equation_system.assignments:
      continue
    # Try to instantiate a single variable to a strided layout and see if it
    # reduces the system.
    op = variable.key.operation
    # TODO(bchetioui): should we make variables carry a shape as well, to make
    # things easier?
    if variable.key.type == VariableType.OPERAND:
      ty = cast(ir.ShapedType, op.operands[variable.key.index].type)
    else:
      ty = cast(ir.ShapedType, op.results[variable.key.index].type)
    max_vec_size = np.prod(ty.shape) // fa.WARPGROUP_SIZE
    # TODO(bchetioui): can't handle too small shapes.
    if max_vec_size == 0:
      continue
    desired_vec_size = 8 // utils.bytewidth(ty.element_type)
    vec_size = min(max_vec_size, desired_vec_size)
    layout = fa.WGStridedFragLayout(shape=tuple(ty.shape), vec_size=vec_size)
    new_assignment = {variable: eqns.Constant(layout)}
    new_system = equation_system & eqns.EquationSystem(assignments=new_assignment)
    if isinstance(new_system, eqns.Unsatisfiable):
      # This assignment is not compatible with the equation system.
      continue
    solution = find_assignments_for(unknowns, new_system, hints)
    if isinstance(solution, eqns.Unsatisfiable):
      # This assignment is not compatible with the equation system.
      continue
    return solution

  # TODO(bchetioui): should we have a way to give a useful dump to the user
  # here, perhaps indicating what to layout cast.
  return eqns.Unsatisfiable()


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
    [ir.OpView],
    tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]
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


def _pointwise_op_equation_system(
    op: ir.OpView,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  all_operands_and_results = operands_and_results(op)
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
    vector.StoreOp,
]:
  _add_equation_system_derivation_rule(op)(_pointwise_op_equation_system)


@_add_equation_system_derivation_rule(vector.LoadOp)
def _vector_load_equation_system(
    op: vector.LoadOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  equation_system: eqns.EquationSystem | eqns.Unsatisfiable
  equation_system, operand_or_results_for_variable, hints = (
      _pointwise_op_equation_system(op)
  )
  [result_variable] = operand_or_results_for_variable.keys()
  result_is_not_splat = eqns.Distinct(
      result_variable,
      eqns.Constant(fa.WGSplatFragLayout(shape=tuple(op.result.type.shape))),
  )
  equation_system &= eqns.EquationSystem(constraints=[result_is_not_splat])
  assert not isinstance(equation_system, eqns.Unsatisfiable)
  return equation_system, operand_or_results_for_variable, hints


@_add_equation_system_derivation_rule(mgpu.OptimizationBarrierOp)
def _optimization_barrier_equation_system(
    op: ir.OpView,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
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


@_add_equation_system_derivation_rule(vector.SplatOp)
def _vector_splat_equation_system(
    op: ir.OpView,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  result = OperandOrResult(op, VariableType.RESULT, 0)
  variable = eqns.Variable(result)
  layout = fa.WGSplatFragLayout(tuple(cast(ir.ShapedType, op.result.type).shape))
  system = eqns.EquationSystem(assignments={variable: eqns.Constant(layout)})
  return system, {variable: [result]}, []


@_add_equation_system_derivation_rule(arith.ConstantOp)
def _constant_equation_system(
    constant_op: arith.ConstantOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  value = constant_op.value
  result = OperandOrResult(constant_op, VariableType.RESULT, 0)
  variable = eqns.Variable(result)
  shape = tuple(constant_op.result.type.shape)
  if (
      ir.DenseElementsAttr.isinstance(value)
      and ir.DenseElementsAttr(value).is_splat
  ):
    layout = fa.WGSplatFragLayout(shape=shape)
    system = eqns.EquationSystem(assignments={variable: eqns.Constant(layout)})
  else:
    constant_is_not_splat = eqns.Distinct(
        variable, eqns.Constant(fa.WGSplatFragLayout(shape=shape)),
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
    op: scf.ForOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  [block] = op.region.blocks
  yield_op = _terminator(block, scf.YieldOp)
  operand_or_results_for_variable: OperandOrResultsForVariable = {}

  # Account for the lower bound, upper bound, and step of the loop, which appear
  # in the operands but not in the results.
  num_leading_args = 3
  for index, o in enumerate(op.operands):
    if not is_vector(o):
      continue
    result_index = index - num_leading_args
    operand = OperandOrResult(op, VariableType.OPERAND, index)
    result = OperandOrResult(op, VariableType.RESULT, result_index)
    yield_operand = OperandOrResult(
        yield_op, VariableType.OPERAND, result_index
    )
    operand_or_results_for_variable[eqns.Variable(operand)] = [
        operand, result, yield_operand,
    ]

  return eqns.EquationSystem(), operand_or_results_for_variable, []


@_add_equation_system_derivation_rule(scf.WhileOp)
def _while_equation_system(
    op: scf.WhileOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  [before_block] = op.before.blocks
  [after_block] = op.after.blocks
  cond_op = _terminator(before_block, scf.ConditionOp)
  yield_op = _terminator(after_block, scf.YieldOp)

  operand_or_results_for_variable: OperandOrResultsForVariable = {}

  for operand_or_result in operands_and_results(op):
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
    op: scf.IndexSwitchOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  operand_or_results_for_variable: OperandOrResultsForVariable = {
      eqns.Variable(o): [o] for o in operands_and_results(op)
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
    op: mgpu.LayoutCastOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  operand = OperandOrResult(op, VariableType.OPERAND, 0)
  result = OperandOrResult(op, VariableType.RESULT, 0)
  variable = eqns.Variable(operand)
  out_layout = eqns.Constant(layouts_lib.from_layout_attr(op.new_layout))
  return (
      eqns.EquationSystem(
          assignments={eqns.Variable(operand): out_layout},
      ),
      {variable: [operand, result]},
      [],
  )


@_add_equation_system_derivation_rule(mgpu.WGMMAOp)
def _wgmma_equation_system(
    op: mgpu.WGMMAOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  operands_or_results = operands_and_results(op)
  variable = eqns.Variable(operands_or_results[0])
  system = eqns.EquationSystem(
      assignments={variable: eqns.Constant(fa.WGMMA_LAYOUT)}
  )
  return system, {variable: operands_or_results}, []


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
    op: vector.MultiDimReductionOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
  source = OperandOrResult(op, VariableType.OPERAND, 0)
  acc = OperandOrResult(op, VariableType.OPERAND, 1)
  out = OperandOrResult(op, VariableType.RESULT, 0)
  source_variable = eqns.Variable(source)
  out_variable = eqns.Variable(out)

  reduction_equation, broadcast_hint = _reduction_equation_and_hint(
      source_variable, out_variable, tuple(op.source.type.shape), tuple(op.reduction_dims)
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
    op: mgpu.BroadcastInDimOp,
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
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
    op: vector.ShapeCastOp
) -> tuple[eqns.EquationSystem, OperandOrResultsForVariable, list[Hint]]:
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


def _ensure_all_layouts_are_set(op: ir.OpView):
  if not inference_utils.should_have_layout(op):
    return
  _ensure_right_number_of_layouts(
      op, inference_utils.in_layouts(op), inference_utils.out_layouts(op)
  )


def _ensure_right_number_of_layouts(
    op: ir.OpView,
    in_layouts: Sequence[fa.FragmentedLayout | ir.Attribute],
    out_layouts: Sequence[fa.FragmentedLayout | ir.Attribute],
):
  """Ensures that the right number of in/out layouts are provided for an op."""
  if len(in_layouts) != sum(map(is_vector, op.operands)):
    raise ValueError(
        "Expected the same number of in_layouts as vector operands."
    )
  if len(out_layouts) != sum(map(is_vector, op.results)):
    raise ValueError(
        "Expected the same number of out_layouts as vector results."
    )


def assign_layouts(
    solution: dict[OperandOrResult, eqns.Constant],
) -> None:
  """Assigns the layouts in `solution` to the MLIR ops they belong to.

  This function requires that, for each MLIR op that appears in `solution`,
  `solution` contains a layout assignment for all of its `vector` operands and
  results.
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

    in_layouts = [
        ce.value for _, ce in sorted(in_assignments, key=lambda kv: kv[0].index)
    ]
    out_layouts = [
        ce.value
        for _, ce in sorted(out_assignments, key=lambda kv: kv[0].index)
    ]

    _ensure_right_number_of_layouts(op, in_layouts, out_layouts)
    in_layouts_attrs = [layouts_lib.to_layout_attr(l) for l in in_layouts]
    out_layouts_attrs = [layouts_lib.to_layout_attr(l) for l in out_layouts]
    op.attributes["in_layouts"] = ir.ArrayAttr.get(in_layouts_attrs)
    op.attributes["out_layouts"] = ir.ArrayAttr.get(out_layouts_attrs)


def operands_and_results(op: ir.OpView) -> list[OperandOrResult]:
  """Returns all the operands and results for the given op."""
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
    raise NotImplementedError(
        f"Producer {producer} is not a ForOp or a FuncOp: {type(producer)}."
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


def infer_layout(module: ir.Module):
  global_equation_system: eqns.EquationSystem | eqns.Unsatisfiable
  global_equation_system = eqns.EquationSystem()
  operand_and_results_for_variable: OperandOrResultsForVariable = {}
  hints: list[Hint] = []

  def gather_equations(op: ir.Operation):
    # Terminator ops are handled directly by the op whose region they belong to.
    # This is because they need to be in sync with their parent op's inputs and
    # outputs---and the parent op's equations therefore need to take them them
    # into account.
    if not inference_utils.should_have_layout(op) or is_terminator(op):
      return
    elif rule := _equation_system_derivation_rules.get(op.OPERATION_NAME, None):  # pytype: disable=attribute-error
      pass
    else:
      raise NotImplementedError(f"No layout inference rule defined for {op}")
    equation_system, mapping, op_hints = rule(op)
    operand_and_results_for_variable.update(mapping)
    nonlocal global_equation_system
    global_equation_system &= equation_system
    hints.extend(op_hints)

  for op in module.body:
    inference_utils.traverse_op(op, gather_equations)

  assert not isinstance(global_equation_system, eqns.Unsatisfiable)
  propagation_hints, constraints = derive_hints_and_constraints(operand_and_results_for_variable)
  hints = reduce_hints(hints + propagation_hints, global_equation_system.assignments)  # pytype: disable=attribute-error
  global_equation_system &= eqns.EquationSystem(constraints=constraints)
  assert not isinstance(global_equation_system, eqns.Unsatisfiable)

  # Attempt to find assignments that satisfy the equation system.
  solution = find_assignments_for(
      operand_and_results_for_variable.keys(), global_equation_system, hints
  )

  if isinstance(solution, eqns.Unsatisfiable):
    raise ValueError(
        "Failed to infer a possible set of layouts. This should only happen if "
        "user-provided layout casts are unsatisfiable."
    )

  layout_for_operand_or_result = {
      k: solution[v]
      for v, ks in operand_and_results_for_variable.items()
      for k in ks
  }

  # Assigns the layouts that we found to the ops.
  assign_layouts(layout_for_operand_or_result)

  # Sanity check: ensure that all ops have the right number of in/out layouts.
  for op in module.body:
    inference_utils.traverse_op(op, _ensure_all_layouts_are_set)
