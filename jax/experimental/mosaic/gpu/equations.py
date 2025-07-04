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


VariableKey = Any


@dataclasses.dataclass(frozen=True)
class Variable:
  """A variable is an abstract identifier.

  `key` is supposed to be hashable.
  """
  key: VariableKey


@dataclasses.dataclass(frozen=True)
class ConstantExpression:
  """Wraps a known layout."""
  value: fa.FragmentedLayout


@dataclasses.dataclass(frozen=True)
class LeastReplicatedExpression:
  expressions: tuple[Expression, ...]

  def __post_init__(self):
    assert len(self.expressions) >= 1


@dataclasses.dataclass(frozen=True)
class MostReplicatedExpression:
  expressions: tuple[Expression, ...]

  def __post_init__(self):
    assert len(self.expressions) >= 1


Expression = Variable | ConstantExpression | LeastReplicatedExpression | MostReplicatedExpression


def meet_layouts(
    layout1: fa.FragmentedLayout, layout2: fa.FragmentedLayout
) -> fa.FragmentedLayout:
  """Returns the "meet" of two layouts that are compatible up to replication.

  The "meet" of the two layouts is the most replicated layout that is compatible
  with both layouts and at most as replicated as the least replicated of the
  two layouts.

  Two layouts are compatible up to replication iff:
    - the two layouts are equal, or
    - one of the layout is a `WGSplatFragLayout` that can be broadcasted to the
      other layout's shape, or
    - the two layouts are `TiledLayout`s and are equal, if we ignore their
      replication dimensions.

  This is the dual of `join_layouts`.

  Returns:
    The "meet" of the two layouts if both layouts are compatible up to
    replication.

  Raises:
    ValueError: if the two layouts are not compatible up to replication.
  """
  if layout1 == layout2:
    return layout1

  match (layout1, layout2):
    case (fa.WGSplatFragLayout(), _):
      if isinstance(layout2, fa.TiledLayout):
        shape = layout2.base_tile_shape
      else:
        shape = layout2.shape
      if layout1.can_broadcast_to(shape):
        return layout2
    case (_, fa.WGSplatFragLayout()):
      if isinstance(layout1, fa.TiledLayout):
        shape = layout1.base_tile_shape
      else:
        shape = layout1.shape
      if layout2.can_broadcast_to(shape):
        return layout1
    case (fa.TiledLayout(), fa.TiledLayout()):
      # TODO(bchetioui): handle `TiledLayout` replication.
      raise NotImplementedError("TiledLayout replication not supported yet")

  raise ValueError(
      f"Layouts {layout1} and {layout2} are not compatible up to replication."
  )


def join_layouts(
    layout1: fa.FragmentedLayout, layout2: fa.FragmentedLayout
) -> fa.FragmentedLayout:
  """Returns the "join" of two layouts that are compatible up to replication.

  The "join" of the two layouts is the least replicated layout that is compatible
  with both layouts and at least as replicated as the most replicated of the
  two layouts.

  Two layouts are compatible up to replication iff:
    - the two layouts are equal, or
    - one of the layout is a `WGSplatFragLayout` that can be broadcasted to the
      other layout's shape, or
    - the two layouts are `TiledLayout`s and are equal, if we ignore their
      replication dimensions.

  This is the dual of `meet_layouts`.

  Returns:
    The "join" of the two layouts if both layouts are compatible up to
    replication.

  Raises:
    ValueError: if the two layouts are not compatible up to replication.
  """
  if layout1 == layout2:
    return layout1

  match (layout1, layout2):
    case (fa.WGSplatFragLayout(), _):
      if isinstance(layout2, fa.TiledLayout):
        shape = layout2.base_tile_shape
      else:
        shape = layout2.shape
      if layout1.can_broadcast_to(shape):
        return layout1
    case (_, fa.WGSplatFragLayout()):
      if isinstance(layout1, fa.TiledLayout):
        shape = layout1.base_tile_shape
      else:
        shape = layout1.shape
      if layout2.can_broadcast_to(shape):
        return layout2
    case (fa.TiledLayout(), fa.TiledLayout()):
      # TODO(bchetioui): handle `TiledLayout` replication.
      raise NotImplementedError("TiledLayout replication not supported yet")

  raise ValueError(
      f"Layouts {layout1} and {layout2} are not compatible up to replication."
  )


def simplify_replicated_expression(
    expr: LeastReplicatedExpression | MostReplicatedExpression,
    assignments: dict[Variable, ConstantExpression],
    constructor: Callable[[tuple[Expression, ...]], Expression],
    reducer: Callable[[ConstantExpression, ConstantExpression], ConstantExpression]
) -> Expression | Unsatisfiable:
  assert len(expr.expressions) >= 1

  new_expressions: list[Expression] = []
  # Use a set to eliminate duplicates, but preserve the order.
  seen: set[Expression] = set()
  for expr in expr.expressions:
    simplified_expr = simplify_expression(expr, assignments)
    if simplified_expr in seen:
      continue
    new_expressions.append(simplified_expr)
    seen.add(simplified_expr)

  if len(new_expressions) == 1:
    return new_expressions[0]

  consts = [e for e in new_expressions if isinstance(e, ConstantExpression)]
  unknowns = [e for e in new_expressions if not isinstance(e, ConstantExpression)]

  if consts:
    const_red, *consts = consts
    red = const_red
    for cst in consts:
      try:
        red = ConstantExpression(reducer(red.value, cst.value))
      except ValueError:
        # The layouts are not compatible up to replication, this expression
        # cannot be simplified.
        return Unsatisfiable()
  else:
    red = None

  if red is not None:
    if unknowns:
      return constructor((red, *unknowns))
    return red

  return constructor(tuple(unknowns))


def simplify_expression(
    expr: Expression, assignments: dict[Variable, ConstantExpression]
) -> Expression | Unsatisfiable:
  """Simplifies an expression as much as is possible given a set of known variable assignments."""
  match expr:
    case ConstantExpression():
      return expr
    case Variable():
      return assignments.get(expr, expr)
    case MostReplicatedExpression():
      return simplify_replicated_expression(
          expr, assignments, MostReplicatedExpression, join_layouts
      )
    case LeastReplicatedExpression():
      return simplify_replicated_expression(
          expr, assignments, LeastReplicatedExpression, meet_layouts
      )
    case _:
      assert_never(expr)


@dataclasses.dataclass(frozen=True)
class Equation:
  lhs: Expression
  rhs: Expression

  def __str__(self):
    return f"{self.lhs} == {self.rhs}"


def reduce_equation(
    eq: Equation, assignments: dict[Variable, ConstantExpression]
) -> Solution | Unsatisfiable:
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
  lhs = simplify_expression(eq.lhs, assignments)
  rhs = simplify_expression(eq.rhs, assignments)
  match (lhs, rhs):
    case (Variable(), ConstantExpression()):
      return SatisfiedBy((lhs, rhs))
    case (ConstantExpression(), Variable()):
      return SatisfiedBy((rhs, lhs))
    case (ConstantExpression(), ConstantExpression()) if lhs != rhs:
      return Unsatisfiable()
    case _ if isinstance(lhs, Unsatisfiable) or isinstance(rhs, Unsatisfiable):
      return Unsatisfiable()
    case _ if lhs == rhs:
      return Tautological()
    case _:
      return Unknown(Equation(lhs, rhs))


@dataclasses.dataclass
class EquationSystem:
  """An equation system contains a set of equations and assignments.

  Assignments assign constant values to variables in the system (bound
  variables). Equations describe relationships between variables, and can be
  used to determine assignments for unknown (free) variables.
  """
  assignments: dict[Variable, ConstantExpression] = dataclasses.field(
      default_factory=dict
  )
  equations: list[Equation] = dataclasses.field(default_factory=list)

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
        case ConstantExpression():
          ...
        case MostReplicatedExpression(expressions=expressions):
          for e in expressions:
            extract_variables(e)
        case LeastReplicatedExpression(expressions=expressions):
          for e in expressions:
            extract_variables(e)
        case _:
          assert_never(expr)
    for equation in self.equations:
      extract_variables(equation.lhs)
      extract_variables(equation.rhs)
    return free_variables

  def __and__(self, other: EquationSystem) -> EquationSystem | Unsatisfiable:
    for variable, assignment in self.assignments.items():
      if variable in other.assignments and assignment != other.assignments[variable]:
        return Unsatisfiable()
    return EquationSystem(
        assignments=self.assignments | other.assignments,
        equations=self.equations + other.equations,
    )


class Unsatisfiable:
  ...


@dataclasses.dataclass(frozen=True)
class SatisfiedBy:
  assignment: tuple[Variable, ConstantExpression]


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


def _simplify_system_once(
    equation_system: EquationSystem,
) -> EquationSystem | Unsatisfiable | None:
  """Performs one simplification step over each equation in an equation system.

  Returns:
    - Unsatisfiable(): if the equation system is unsatisfiable.
    - A new equation system if any equation was simplified.
    - None: if the equation system is not known unsatisfiable, but hasn't been
      simplified.
  """
  changed = False
  assignments: dict[Variable, ConstantExpression] = dict()
  equations: list[Equation] = []
  for equation in equation_system.equations:
    match (result := reduce_equation(equation, equation_system.assignments)):
      case Unsatisfiable():
        return Unsatisfiable()
      case Tautological():
        changed = True
      case SatisfiedBy():
        variable, expression = result.assignment
        if variable in assignments and assignments[variable] != expression:
          return Unsatisfiable()
        assignments[variable] = expression
        changed = True
      case Unknown(equation=reduced_equation):
        equations.append(reduced_equation)
        changed |= reduced_equation != equation
      case _:
        assert_never(result)

  if changed:
    return EquationSystem(
        assignments=assignments | equation_system.assignments,
        equations=equations,
    )
  return None


def simplify(equation_system: EquationSystem) -> EquationSystem | Unsatisfiable:
  """Simplifies an equation system until it can no longer be simplified.

  Returns:
    - Unsatisfiable(): if the equation system is unsatisfiable.
    - The maximally simplified equation system otherwise.
  """
  while True:
    match (new_system := _simplify_system_once(equation_system)):
      case None:
        break
      case Unsatisfiable():
        return Unsatisfiable()
      case EquationSystem():
        equation_system = new_system
      case _:
        assert_never(new_system)

  return equation_system
