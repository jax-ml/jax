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

"""Defines expressions and equations over layouts."""

from __future__ import annotations

import dataclasses
from typing import assert_never, Any, Callable

from . import fragmented_array as fa
from . import layouts as layouts_lib


VariableKey = Any


@dataclasses.dataclass(frozen=True)
class Variable:
  """A variable is an abstract identifier.

  `key` is supposed to be hashable.
  """
  key: VariableKey


@dataclasses.dataclass(frozen=True)
class Constant:
  """Wraps a known layout."""
  value: fa.FragmentedLayout


@dataclasses.dataclass(frozen=True)
class LeastReplicated:
  expressions: tuple[Expression, ...]

  def __post_init__(self):
    assert len(self.expressions) >= 1


@dataclasses.dataclass(frozen=True)
class MostReplicated:
  expressions: tuple[Expression, ...]

  def __post_init__(self):
    assert len(self.expressions) >= 1


@dataclasses.dataclass(frozen=True)
class Reduce:
  expression: Expression
  axes: tuple[int, ...]


Expression = Variable | Constant | LeastReplicated | MostReplicated | Reduce


def reduce_replicated_expression(
    input_expr: LeastReplicated | MostReplicated,
    assignments: dict[Variable, Constant],
    reducer: Callable[[fa.FragmentedLayout, fa.FragmentedLayout], fa.FragmentedLayout | None]
) -> Expression | Unsatisfiable:
  assert input_expr.expressions

  new_expressions: list[Expression] = []
  # Use a set to eliminate duplicates, but preserve the order.
  seen: set[Expression] = set()
  for expr in input_expr.expressions:
    reduced_expr = reduce_expression(expr, assignments)
    if isinstance(reduced_expr, Unsatisfiable):
      return Unsatisfiable()
    if reduced_expr in seen:
      continue
    new_expressions.append(reduced_expr)
    seen.add(reduced_expr)

  if len(new_expressions) == 1:
    return new_expressions[0]

  consts = [e for e in new_expressions if isinstance(e, Constant)]
  unknowns = [e for e in new_expressions if not isinstance(e, Constant)]

  if consts:
    const_red, *consts = consts
    red = const_red
    for cst in consts:
      red_value = reducer(red.value, cst.value)
      if red_value is None:
        # The layouts are not compatible up to replication, this expression
        # cannot be simplified.
        return Unsatisfiable()
      red = Constant(red_value)
  else:
    red = None

  constructor = type(input_expr)
  if red is not None:
    if unknowns:
      return constructor((red, *unknowns))
    return red

  return constructor(tuple(unknowns))


def reduce_expression(
    expr: Expression, assignments: dict[Variable, Constant]
) -> Expression | Unsatisfiable:
  """Reduces an expression as much as is possible given a set of known variable assignments."""
  match expr:
    case Constant():
      return expr
    case Variable():
      return assignments.get(expr, expr)
    case MostReplicated():
      return reduce_replicated_expression(
          expr, assignments, layouts_lib.join_layouts
      )
    case LeastReplicated():
      return reduce_replicated_expression(
          expr, assignments, layouts_lib.meet_layouts
      )
    case Reduce(expression=expr, axes=axes):
      reduced_expr = reduce_expression(expr, assignments)
      match reduced_expr:
        case Unsatisfiable():
          return Unsatisfiable()
        case Constant(value=layout) if isinstance(layout, fa.TiledLayout):
          return Constant(layout.reduce(axes))
        case Constant():
          # Explicitly raise an error here as opposed to simply failing to
          # simplify, so that we get a clear signal if we ever need to implement
          # this.
          raise NotImplementedError(
              "Reduction of non-tiled layouts is not implemented yet."
          )
        case _:
          return Reduce(expression=reduced_expr, axes=axes)
    case _:
      assert_never(expr)


_SUPPORTED_TILED_RELAYOUTS = frozenset([
    # Transposed layouts.
    (fa.WGMMA_LAYOUT, fa.WGMMA_TRANSPOSED_LAYOUT),
    (fa.TCGEN05_LAYOUT, fa.TCGEN05_TRANSPOSED_LAYOUT),
    # "Conversion-optimized" layouts.
    (fa.WGMMA_LAYOUT_UPCAST_2X, fa.WGMMA_LAYOUT),
    (fa.WGMMA_LAYOUT_UPCAST_4X, fa.WGMMA_LAYOUT_UPCAST_2X),
    (fa.WGMMA_LAYOUT_UPCAST_4X, fa.WGMMA_LAYOUT),
])


@dataclasses.dataclass(frozen=True)
class Relayout:
  """States that `source` must be relayout-able to `target`.

  Relayout-ability here is not defined as a fundamental property of layouts, but
  rather a reflection of our implementation. For instance, when evaluating this
  constraint, we will return `False` systematically if a relayout exists but we
  do not ever plan to support it.

  Modeling this constraint this way is helpful, in order to allow pruning
  inefficient solutions when attempting to solve an equation system.
  """

  source: Expression
  target: Expression

  def holds(self) -> bool | None:
    """Returns whether the relayout constraint holds.

    Returns `None` if the constraint can't be checked.
    """
    source = self.source
    target = self.target

    # Fast path for syntactically identical expressions.
    if source == target:
      return True

    if not isinstance(source, Constant) or not isinstance(target, Constant):
      return None

    source_layout, target_layout = source.value, target.value
    match source_layout, target_layout:
      case fa.WGSplatFragLayout(), fa.WGStridedFragLayout():
        return source_layout.shape == target_layout.shape
      case fa.WGSplatFragLayout(), fa.TiledLayout():
        return layouts_lib.splat_is_compatible_with_tiled(
            source_layout, target_layout
        )
      case fa.TiledLayout(), fa.TiledLayout():
        return (source_layout, target_layout) in _SUPPORTED_TILED_RELAYOUTS
      case _:
        return False


@dataclasses.dataclass(frozen=True)
class Distinct:
  """States that `lhs != rhs`."""
  lhs: Expression
  rhs: Expression

  def holds(self) -> bool | None:
    """Whether the distinctiveness constraint holds.

    Returns `None` if the constraint can't be checked.
    """
    if self.lhs == self.rhs:
      return False
    if isinstance(self.lhs, Constant) and isinstance(self.rhs, Constant):
      return True
    return None


Constraint = Relayout | Distinct


def reduce_constraint(
    constraint: Constraint, assignments: dict[Variable, Constant]
) -> Constraint | Tautological | Unsatisfiable:
  """Reduces a constraint."""
  match constraint:
    case Relayout(source=lhs, target=rhs):
      ...
    case Distinct(lhs=lhs, rhs=rhs):
      ...
    case _ as never:
      assert_never(never)

  lhs_red = reduce_expression(lhs, assignments)
  rhs_red = reduce_expression(rhs, assignments)

  if isinstance(lhs_red, Unsatisfiable) or isinstance(rhs_red, Unsatisfiable):
    return Unsatisfiable()

  new_constraint = type(constraint)(lhs_red, rhs_red)
  constraint_holds = new_constraint.holds()
  if constraint_holds is None:
    return new_constraint
  return Tautological() if constraint_holds else Unsatisfiable()


@dataclasses.dataclass(frozen=True)
class Equation:
  lhs: Expression
  rhs: Expression

  def __str__(self):
    return f"{self.lhs} == {self.rhs}"


def reduce_equation(
    eq: Equation, assignments: dict[Variable, Constant]
) -> Solution:
  """Reduces an equation.

  Args:
    eq: the equation to reduce.
    assignments: a set of known variable assignments.

  Returns:
    A Solution object representing the result of the evaluation. That is:
      - Unsatisfiable(): if the equation is unsatisfiable.
      - Tautological(): if the equation is tautological.
      - Satisfiable(): if the equation is satisfiable by assigning a value to
          a variable.
      - Unknown(): if the equation contains remaining unknown variables.
  """
  lhs = reduce_expression(eq.lhs, assignments)
  rhs = reduce_expression(eq.rhs, assignments)
  match (lhs, rhs):
    case (Variable(), Constant()):
      return SatisfiedBy((lhs, rhs))
    case (Constant(), Variable()):
      return SatisfiedBy((rhs, lhs))
    case (Constant(), Constant()) if lhs != rhs:
      return Unsatisfiable()
    case _ if isinstance(lhs, Unsatisfiable) or isinstance(rhs, Unsatisfiable):
      return Unsatisfiable()
    case _ if lhs == rhs:
      return Tautological()
    case _:
      # This is covered above. Add a check here to appease the type checker.
      assert not isinstance(lhs, Unsatisfiable) and not isinstance(rhs, Unsatisfiable)
      return Unknown(Equation(lhs, rhs))


@dataclasses.dataclass
class EquationSystem:
  """An equation system contains a set of equations and assignments.

  Assignments assign constant values to variables in the system (bound
  variables). Equations describe relationships between variables, and can be
  used to determine assignments for unknown (free) variables.

  Constraints are used to check predicates that must hold for the assignments to
  be valid.
  """
  assignments: dict[Variable, Constant] = dataclasses.field(
      default_factory=dict
  )
  equations: list[Equation] = dataclasses.field(default_factory=list)
  constraints: list[Constraint] = dataclasses.field(default_factory=list)

  def unknowns(self) -> list[Variable]:
    """Returns the list of free variables in the system."""
    seen_variables: set[Variable] = set()
    free_variables: list[Variable] = []
    def extract_variables(expr: Expression) -> None:
      match expr:
        case Variable():
          if expr not in seen_variables and expr not in self.assignments:
            seen_variables.add(expr)
            free_variables.append(expr)
        case Constant():
          ...
        case MostReplicated(expressions=expressions):
          for e in expressions:
            extract_variables(e)
        case LeastReplicated(expressions=expressions):
          for e in expressions:
            extract_variables(e)
        case Reduce(expression=e):
          extract_variables(e)
        case _:
          assert_never(expr)
    for equation in self.equations:
      extract_variables(equation.lhs)
      extract_variables(equation.rhs)
    for constraint in self.constraints:
      match constraint:
        case Relayout(source=source, target=target):
          extract_variables(source)
          extract_variables(target)
        case Distinct(lhs=lhs, rhs=rhs):
          extract_variables(lhs)
          extract_variables(rhs)
        case _ as never:
          assert_never(never)
    return free_variables

  def __and__(self, other: EquationSystem) -> EquationSystem | Unsatisfiable:
    for variable, assignment in self.assignments.items():
      if variable in other.assignments and assignment != other.assignments[variable]:
        return Unsatisfiable()
    return EquationSystem(
        assignments=self.assignments | other.assignments,
        equations=self.equations + other.equations,
        constraints=self.constraints + other.constraints,
    )


class Unsatisfiable:
  ...


@dataclasses.dataclass(frozen=True)
class SatisfiedBy:
  assignment: tuple[Variable, Constant]


@dataclasses.dataclass(frozen=True)
class Unknown:
  equation: Equation


class Tautological:
  ...


# The result of reducing an equation---and by extension, a system of
# equations. An equation can either be unsatisfiable (i.e. there exists no
# assignment for which it holds), satisfied by an assignment, unknown (i.e.
# still undetermined), or tautological (i.e. the equation is guaranteed to
# hold for any assignment).
Solution = Unsatisfiable | SatisfiedBy | Unknown | Tautological


def _reduce_system_once(
    equation_system: EquationSystem,
) -> EquationSystem | Unsatisfiable | None:
  """Performs one reduction step over each equation in an equation system.

  Returns:
    - Unsatisfiable(): if the equation system is unsatisfiable.
    - A new equation system if any equation was reduced.
    - None: if the equation system is not known unsatisfiable, but hasn't been
      reduced.
  """
  changed = False
  assignments: dict[Variable, Constant] = dict()
  equations: list[Equation] = []
  for equation in equation_system.equations:
    match reduce_equation(equation, equation_system.assignments):
      case Unsatisfiable():
        return Unsatisfiable()
      case Tautological():
        changed = True
      case SatisfiedBy() as result:
        variable, expression = result.assignment
        if variable in assignments and assignments[variable] != expression:
          return Unsatisfiable()
        assignments[variable] = expression
        changed = True
      case Unknown(equation=reduced_equation):
        equations.append(reduced_equation)
        changed |= reduced_equation != equation
      case _ as never:
        assert_never(never)

  assignments |= equation_system.assignments
  constraints: list[Constraint] = []
  for constraint in equation_system.constraints:
    match reduce_constraint(constraint, assignments):
      case Unsatisfiable():
        return Unsatisfiable()
      case Tautological():
        changed = True
      case _ as new_constraint:
        changed |= new_constraint != constraint
        constraints.append(new_constraint)

  if changed:
    return EquationSystem(
        assignments=assignments | equation_system.assignments,
        equations=equations,
        constraints=constraints,
    )
  return None


def reduce(equation_system: EquationSystem) -> EquationSystem | Unsatisfiable:
  """Reduces an equation system until it can no longer be reduced.

  Returns:
    - Unsatisfiable(): if the equation system is unsatisfiable.
    - The maximally reduced equation system otherwise.
  """
  while True:
    match _reduce_system_once(equation_system):
      case None:
        break
      case Unsatisfiable():
        return Unsatisfiable()
      case EquationSystem() as new_system:
        equation_system = new_system
      case _ as never:
        assert_never(never)

  return equation_system
