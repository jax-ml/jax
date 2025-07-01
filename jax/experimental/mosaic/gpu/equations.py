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
from typing import assert_never, Any

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


Expression = Variable | ConstantExpression


def simplify_expression(
    expr: Expression, assignments: dict[Variable, ConstantExpression]
) -> Expression:
  """Simplifies an expression as much as is possible given a set of known variable assignments."""
  match expr:
    case ConstantExpression():
      return expr
    case Variable():
      return assignments.get(expr, expr)
    case _:
      assert_never(expr)


@dataclasses.dataclass(frozen=True)
class Equation:
  lhs: Expression
  rhs: Expression

  def __str__(self):
    return f"{self.lhs} == {self.rhs}"


def simplify_equation(
    eq: Equation, assignments: dict[Variable, ConstantExpression]
) -> Equation:
  """Applies `reduce_expression` to both sides of an equation."""
  lhs = simplify_expression(eq.lhs, assignments)
  rhs = simplify_expression(eq.rhs, assignments)
  return Equation(lhs, rhs)


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

  def __post_init__(self):
    self.equations = [simplify_equation(e, self.assignments) for e in self.equations]

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


class Unknown:
  ...

class Tautological:
  ...


# The result of evaluating an equation---and by extension, a system of
# equations. An equation can either be unsatisfiable (i.e. there exists no
# assignment for which it holds), satisfied by an assignment, unknown (i.e.
# still undetermined), or tautological (i.e. the equation is guaranteed to
# hold for any assignment).
Solution = Unsatisfiable | SatisfiedBy | Unknown | Tautological


def evaluate_equation(eq: Equation) -> Solution:
  """Evaluates an equation.

  Args:
    eq: the equation to evaluate. The function does not reduce the equation
      before evaluating it, so it is assumed that the caller has already
      performed any necessary reduction.

  Returns:
    A Solution object representing the result of the evaluation. That is:
      - Unsatisfiable(): if the equation is unsatisfiable.
      - Tautological(): if the equation is tautological.
      - Satisfiable(): if the equation is satisfiable by assigning a value to
          a variable.
      - Unknown(): if the equation contains remaining unknown variables.
  """
  match (eq.lhs, eq.rhs):
    case (Variable(), ConstantExpression()):
      return SatisfiedBy((eq.lhs, eq.rhs))
    case (ConstantExpression(), Variable()):
      return SatisfiedBy((eq.rhs, eq.lhs))
    case (ConstantExpression(), ConstantExpression()) if eq.lhs != eq.rhs:
      return Unsatisfiable()
    case _ if eq.lhs == eq.rhs:
      return Tautological()
    case _:
      return Unknown()


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
    simplified_equation = simplify_equation(equation, equation_system.assignments)
    match (result := evaluate_equation(simplified_equation)):
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
      case Unknown():
        equations.append(simplified_equation)
        changed |= simplified_equation != equation
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
