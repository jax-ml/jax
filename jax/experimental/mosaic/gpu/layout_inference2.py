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

from collections.abc import Callable, Sequence
import dataclasses
import enum
import itertools
from typing import cast

from jax._src.lib import mosaic_gpu_dialect as mgpu  # noqa: F401
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
import numpy as np

from . import equations as eqns
from . import fragmented_array as fa
from . import inference_utils
from . import layouts as layouts_lib
from . import utils


# mypy: ignore-errors


class VariableType(enum.IntEnum):
  """The type of a variable.

  Variables here are either operands or results of MLIR operations.
  """
  OPERAND = 0
  RESULT = 1

@dataclasses.dataclass(frozen=True)
class VariableKey:
  """A unique identifier for a variable."""
  # A MLIR operation.
  operation: ir.OpView
  # Whether this represents an operand or a result.
  type: VariableType
  # The index of the operand/result within the op's operands/results.
  index: int


@dataclasses.dataclass(frozen=True)
class Variable(eqns.Variable):
  """This variable represents an operand/result of a MLIR operation."""
  def __init__(self, operation: ir.OpView, type: VariableType, index: int):
    super().__init__(VariableKey(operation, type, index))

  @property
  def is_operand(self) -> bool:
    return self.key.type == VariableType.OPERAND

  @property
  def is_result(self) -> bool:
    return self.key.type == VariableType.RESULT


@dataclasses.dataclass(frozen=True)
class Hint:
  """Hints are used to model propagation of layouts across operations.

  Since using `relayout`s is always an option in principle, propagation across
  ops can not rely only on an equation system. Instead, we introduce hints as
  an equation-like form of "soft constraints", i.e., it suggests that
  `variable` should be equal to `expression`.
  """
  variable: Variable
  expression: eqns.Expression


def choose_variable_assignment_from_hints(
    hints: Sequence[Hint],
) -> tuple[Variable, eqns.ConstantExpression] | None:
  """Attempts to choose a single variable assignment from a list of `Hint`s."""
  for hint in hints:
    if isinstance(hint.expression, eqns.ConstantExpression):
      return (hint.variable, hint.expression)
  return None


def simplify_hint(
    h: Hint, assignments: dict[Variable, eqns.ConstantExpression]
) -> Hint:
  """Like `eqns.simplify_equation` but for `Hint`s."""
  return dataclasses.replace(
      h, expression=eqns.simplify_expression(h.expression, assignments))

def find_assignments_for(
    unknowns: set[Variable],
    equation_system: eqns.EquationSystem,
    hints: Sequence[Hint],
) -> dict[Variable, eqns.ConstantExpression] | eqns.Unsatisfiable:
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
  while True:
    equation_system = eqns.simplify(equation_system)
    if isinstance(equation_system, eqns.Unsatisfiable):
      return eqns.Unsatisfiable()

    remaining_unknowns = unknowns - equation_system.assignments.keys()

    # In this case, we have determined an assignment for all the unknown
    # variables. Return their respective assignment.
    if not remaining_unknowns:
      return {v: k for v, k in equation_system.assignments.items() if v in unknowns}

    # Simplify the expressions in the remaining hints based on the current
    # assignments, and eliminate hints that pertain to variables that already
    # have an assignment.
    hints = [simplify_hint(h, equation_system.assignments) for h in hints
             if h.variable not in equation_system.assignments]

    # If unknowns remain and we have fully simplified the system, we may still
    # be able to make progress by extracting an assignment from a `Hint`. In a
    # system that has otherwise been fully simplified, it is guaranteed that
    # introducing a new assignment will yield a system that remains satisfiable
    # if the original system was satisfiable---because this is a sign of an
    # underdetermined system.
    if (assignment := choose_variable_assignment_from_hints(hints)) is not None:
      variable, expr = assignment
      equation_system &= eqns.EquationSystem(assignments={variable: expr})
    else:
      break

  # Here, we have not managed to find an assignment for all the unknown
  # variables, and our hints have not proven sufficient to unblock us. We now
  # try to introduce new arbitrary (valid) assignments into the system, and
  # hope that they turn out to be compatible with the equation system.
  for variable in unknowns:
    if variable in equation_system.assignments:
      continue
    # Try to instantiate a single variable to a strided layout and see if it
    # simplifies the system.
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
    new_assignment = {variable: eqns.ConstantExpression(layout)}
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


EquationSystemDerivationRule = Callable[[ir.OpView], eqns.EquationSystem]
_equation_system_derivation_rules: dict[str, EquationSystemDerivationRule] = {}


def _add_equation_system_derivation_rule(op: type[ir.OpView]):
  def wrapper(rule: EquationSystemDerivationRule):
    if op is not None:
      _equation_system_derivation_rules[op.OPERATION_NAME] = rule  # pytype: disable=attribute-error
    return rule
  return wrapper


def is_vector(v: ir.Value) -> bool:
  return ir.VectorType.isinstance(v.type)


@_add_equation_system_derivation_rule(arith.ConstantOp)
def _constant_equation_system(
    constant_op: arith.ConstantOp
) -> eqns.EquationSystem:
  value = constant_op.value
  variable = Variable(constant_op, VariableType.RESULT, 0)
  if (
      ir.DenseElementsAttr.isinstance(value)
      and ir.DenseElementsAttr(value).is_splat
  ):
    layout = fa.WGSplatFragLayout(shape=tuple(constant_op.result.type.shape))
    return eqns.EquationSystem(assignments={variable: eqns.ConstantExpression(layout)})
  return eqns.EquationSystem()


@_add_equation_system_derivation_rule(mgpu.LayoutCastOp)
def _layout_cast_equation_system(
    op: mgpu.LayoutCastOp
) -> eqns.EquationSystem:
  in_variable = Variable(op, VariableType.OPERAND, 0)
  out_variable = Variable(op, VariableType.RESULT, 0)
  out_layout = eqns.ConstantExpression(layouts_lib.from_layout_attr(op.new_layout))
  return eqns.EquationSystem(
      assignments={out_variable: out_layout, in_variable: out_layout},
  )


def _ensure_all_layouts_are_set(op: ir.OpView):
  if not inference_utils.should_have_layout(op):
    return
  _ensure_right_number_of_layouts(
      op, inference_utils.in_layouts(op), inference_utils.out_layouts(op)
  )

def _ensure_right_number_of_layouts(
    op: ir.OpView,
    in_layouts: list[fa.FragmentedLayout | ir.Attribute],
    out_layouts: list[fa.FragmentedLayout | ir.Attribute],
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


def assign_layouts(solution: dict[Variable, eqns.ConstantExpression]):
  """Assigns the layouts in `solution` to the MLIR ops they belong to.

  This function requires that, for each MLIR op that appears in `solution`,
  `solution` contains a layout assignment for all of its `vector` operands and
  results.
  """
  solution_sorted_by_op = sorted(
      solution.items(), key=lambda kv: id(kv[0].key.operation)
  )
  solution_per_op = itertools.groupby(
      solution_sorted_by_op, key=lambda kv: kv[0].key.operation
  )

  for op, assignments in solution_per_op:
    assignments_sorted_by_type = sorted(assignments, key=lambda kv: kv[0].key.type)
    assignments_by_type = {
        ty: list(group)
        for ty, group in itertools.groupby(
            assignments_sorted_by_type, key=lambda kv: kv[0].key.type
        )
    }

    in_assignments = assignments_by_type.get(VariableType.OPERAND, [])
    out_assignments = assignments_by_type.get(VariableType.RESULT, [])

    in_layouts = [
        ce.value for _, ce in sorted(in_assignments, key=lambda kv: kv[0].key.index)
    ]
    out_layouts = [
        ce.value
        for _, ce in sorted(out_assignments, key=lambda kv: kv[0].key.index)
    ]

    _ensure_right_number_of_layouts(op, in_layouts, out_layouts)
    in_layouts_attrs = [layouts_lib.to_layout_attr(l) for l in in_layouts]
    out_layouts_attrs = [layouts_lib.to_layout_attr(l) for l in out_layouts]
    op.attributes["in_layouts"] = ir.ArrayAttr.get(in_layouts_attrs)
    op.attributes["out_layouts"] = ir.ArrayAttr.get(out_layouts_attrs)


def op_variables(op: ir.OpView) -> list[Variable]:
  """Returns all the operand and result variables for the given op."""
  variables = [
      Variable(op, VariableType.OPERAND, i)
      for i, o in enumerate(op.operands)
      if is_vector(o)
  ]
  variables.extend([
      Variable(op, VariableType.RESULT, i)
      for i, o in enumerate(op.results)
      if is_vector(o)
  ])
  return variables


def producer_variable(variable: Variable) -> Variable:
  """Given a variable, returns the corresponding result variable in its producer.

  The variable has to represent an operand of its operation.
  """
  assert variable.is_operand
  value = variable.key.operation.operands[variable.key.index]
  producer = value.owner
  if isinstance(producer, ir.Operation):
    index = list(producer.results).index(value)
    return Variable(producer.opview, VariableType.RESULT, index)

  # Block case, useful for deriving layouts for ops
  # depending on function parameters, or loop block arguments.
  if isinstance(producer, ir.Block):
    index = list(cast(ir.Block, producer).arguments).index(value)
    return Variable(producer, VariableType.OPERAND, index)

  raise TypeError(
      f"Producer {producer} is not an operation nor a block: {type(producer)}."
  )


def consumer_variables(variable: Variable) -> Sequence[Variable]:
  """Given a variable, returns the corresponding operand variables in its consumers.

  The variable has to represent a result of its operation.
  """
  assert variable.is_result
  consumer_variables: list[Variable] = []
  # The layout can also be chosen from the layout of the consumers of the
  # results.
  for use in cast(ir.OpResult, variable.key.operation.results[variable.key.index]).uses:
    consumer = use.owner.opview  # pytype: disable=attribute-error
    index = use.operand_number
    consumer_variables.append(Variable(consumer, VariableType.OPERAND, index))
  return consumer_variables


def equation_system_and_hints_for_op(
    op: ir.OpView, rule: EquationSystemDerivationRule
) -> tuple[eqns.EquationSystem, list[Hint]]:
  """Produces an equation system and a list of hints for the given op.

  The equation system is derived directly from the given rule, and is not
  further constrained. Hints are subsequently derived from this equation system
  that relate the variables of the op to the producers of the op's operands and
  the consumers of the op's results.
  """
  equation_system = rule(op)
  all_variables: list[Variable] = op_variables(op)
  visited: set[Variable] = set()
  hints: list[Hint] = list()

  for variable in all_variables:
    if variable in visited:
      continue
    # Construct a list containing all the variables that are necessary equal to
    # the current variable. Consider the following pseudo-program:
    #
    #   a = producer0()  # variable v0 is producer0's out_layouts[0]
    #   b = producer1()  # variable v1 is producer1's out_layouts[0]
    #   c = add(a, b)    # variable v2, v3, v4 are respectively add's in_layouts[0], in_layouts[1], and out_layouts[0]
    #   consumer0(c)     # variable v5 is consumer0's in_layouts[0]
    #   consumer1(c)     # variable v6 is consumer1's in_layouts[0]
    #
    # We know that v2 = v3 = v4, and we may want to propagate a layout from v0,
    # v1, v5, or v6. For that reason, we capture all the connected variables,
    # and then extract their producer/consumers to construct a `Hint`.
    #
    # We use a list here because we care about having a deterministic iteration
    # order.
    union: list[Variable] = [variable]
    for equation in equation_system.equations:
      lhs, rhs = equation.lhs, equation.rhs
      if lhs == variable and isinstance(rhs, Variable) and rhs not in union:
        union.append(rhs)
      if rhs == variable and isinstance(lhs, Variable) and lhs not in union:
        union.append(lhs)

    producers = tuple(producer_variable(v) for v in union if v.is_operand)
    consumers: list[Variable] = []
    for v in union:
      if v.is_result:
        consumers.extend(consumer_variables(v))

    if producers:
      least_replicated_producer = eqns.LeastReplicatedExpression(producers)
      hint_expr = eqns.MostReplicatedExpression(
          (least_replicated_producer, *consumers)
      )
      hints.append(Hint(variable, hint_expr))
    elif consumers:
      hint_expr = eqns.MostReplicatedExpression(tuple(consumers))
      hints.append(Hint(variable, hint_expr))
    visited.update(union)

  return equation_system, [simplify_hint(h, equation_system.assignments) for h in hints]


def infer_layout(module: ir.Module):
  global_equation_system = eqns.EquationSystem()
  all_hints: list[Hint] = []
  variables: set[Variable] = set()

  def gather_equations(op: ir.Operation):
    if not inference_utils.should_have_layout(op):
      return
    elif rule := _equation_system_derivation_rules.get(op.OPERATION_NAME, None):  # pytype: disable=attribute-error
      pass
    else:
      raise NotImplementedError(f"No layout inference rule defined for {op}")

    variables.update(op_variables(op))
    nonlocal global_equation_system
    equation_system, hints = equation_system_and_hints_for_op(op, rule)
    global_equation_system &= equation_system
    all_hints.extend(hints)

  for op in module.body:
    inference_utils.traverse_op(op, gather_equations)

  # Attempt to find assignments that satisfy the equation system.
  solution = find_assignments_for(variables, global_equation_system, all_hints)

  if isinstance(solution, eqns.Unsatisfiable):
    raise ValueError(
        "Failed to infer a possible set of layouts. This should never happen."
    )

  # Assigns the layouts that we found to the ops.
  assign_layouts(solution)

  # Sanity check: ensure that all ops have the right number of in/out layouts.
  for op in module.body:
    inference_utils.traverse_op(op, _ensure_all_layouts_are_set)
