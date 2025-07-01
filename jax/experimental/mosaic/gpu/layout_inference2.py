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

from collections.abc import Callable
import dataclasses
import enum
import itertools

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith

from . import equations
from . import fragmented_array as fa
from . import inference_utils
from . import layouts as layouts_lib


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
class Variable(equations.Variable):
  """This variable represents an operand/result of a MLIR operation."""
  def __init__(self, operation: ir.OpView, type: VariableType, index: int):
    super().__init__(VariableKey(operation, type, index))


def find_assignments_for(
    unknowns: set[equations.Variable],
    equation_system: equations.EquationSystem
) -> dict[Variable, equations.ConstantExpression] | equations.Unsatisfiable:
  """Attempts to find assignments that satisfy `equation_system` for `unknowns`.

  Returns:
    - Unsatisfiable() if the equation system has unsatisfiable constraints.
    - A dictionary assigning all the unknown variables to `ConstantExpression`s
      such that the assignment satisfies the equation system otherwise.
  """
  equation_system = equations.simplify(equation_system)
  if isinstance(equation_system, equations.Unsatisfiable):
    return equations.Unsatisfiable()

  remaining_unknowns = equation_system.assignments.keys() - unknowns
  # In this case, we have determined an assignment for all the unknown
  # variables. Return their respective assignment.
  if not remaining_unknowns:
    return {v: k for v, k in equation_system.assignments.items() if v in unknowns}

  raise NotImplementedError("Default assignment logic")


EquationSystemDerivationRule = Callable[[ir.OpView], equations.EquationSystem]
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
) -> equations.EquationSystem:
  value = constant_op.value
  variable = Variable(constant_op, VariableType.RESULT, 0)
  if (
      ir.DenseElementsAttr.isinstance(value)
      and ir.DenseElementsAttr(value).is_splat
  ):
    layout = fa.WGSplatFragLayout(shape=tuple(constant_op.result.type.shape))
    return equations.EquationSystem(assignments={variable: equations.ConstantExpression(layout)})
  return equations.EquationSystem()


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


def assign_layouts(solution: dict[Variable, equations.ConstantExpression]):
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


def infer_layout(module: ir.Module):
  global_equation_system = equations.EquationSystem()
  variables: set[Variable] = set()

  def gather_equations(op: ir.Operation):
    if not inference_utils.should_have_layout(op):
      return
    elif equation_system_derivation_rule := _equation_system_derivation_rules.get(op.OPERATION_NAME, None):  # pytype: disable=attribute-error
      pass
    else:
      raise NotImplementedError(f"No layout inference rule defined for {op}")

    variables.update(op_variables(op))
    nonlocal global_equation_system
    global_equation_system &= equation_system_derivation_rule(op)
    # TODO(bchetioui): add inter-op rules. This will be done very soon, it is
    # just dropped from the initial version to make it easier to review.

  for op in module.body:
    inference_utils.traverse_op(op, gather_equations)

  # Attempt to find assignments that satisfy the equation system.
  solution = find_assignments_for(variables, global_equation_system)

  if isinstance(solution, equations.Unsatisfiable):
    raise ValueError(
        "Failed to infer a possible set of layouts. This should never happen."
    )

  # Assigns the layouts that we found to the ops.
  assign_layouts(solution)

  # Sanity check: ensure that all ops have the right number of in/out layouts.
  for op in module.body:
    inference_utils.traverse_op(op, _ensure_all_layouts_are_set)
