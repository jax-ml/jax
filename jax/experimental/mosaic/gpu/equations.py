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
import enum
from typing import assert_never

from jax._src.lib.mlir import ir

from . import fragmented_array as fa


class VariableType(enum.IntEnum):
  """The type of a variable."""

  OPERAND = 0
  RESULT = 1

  def __str__(self) -> str:
    return self.name.lower()


@dataclasses.dataclass(frozen=True)
class Variable:
  """A variable corresponds to an operand or result of an MLIR operation."""
  operation: ir.OpView
  type: VariableType
  index: int

  def __str__(self):
    return f"Variable({self.operation.OPERATION_NAME}, {self.type.name}={self.index})"  # pytype: disable=attribute-error


@dataclasses.dataclass(frozen=True)
class ConstantExpression:
  """Wraps a known layout."""
  value: fa.FragmentedLayout


Expression = Variable | ConstantExpression


def reduce_expression(
    expr: Expression, known_variables: dict[Variable, Expression]
) -> Expression:
  """Reduces an expression as much as is possible given a set of known variables."""
  match expr:
    case ConstantExpression():
      return expr
    case Variable():
      return known_variables.get(expr, expr)
    case _:
      assert_never(f"Unknown expression type: {type(expr)}")


@dataclasses.dataclass(frozen=True)
class Equation:
  lhs: Expression
  rhs: Expression

  def __str__(self):
    return f"{self.lhs} == {self.rhs}"


def reduce_equation(
    eq: Equation, assignments: dict[Variable, ConstantExpression]
) -> Equation:
  """Applies `reduce_expression` to both sides of an equation."""
  return Equation(*[reduce_expression(e, assignments) for e in (eq.lhs, eq.rhs)])


@dataclasses.dataclass
class EquationSystem:
  solution: dict[Variable, ConstantExpression] = dataclasses.field(
      default_factory=dict
  )
  unknowns: set[Variable] = dataclasses.field(default_factory=set)
  equations: set[Equation] = dataclasses.field(default_factory=set)

  def __post_init__(self):
    for s in self.solution:
      self.unknowns.discard(s)
    self.equations = {reduce_equation(e, self.solution) for e in self.equations}

  def __or__(self, other: EquationSystem) -> EquationSystem:
    return EquationSystem(
        solution=self.solution | other.solution,
        unknowns=self.unknowns | other.unknowns,
        equations=self.equations | other.equations,
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
  solutions: dict[Variable, ConstantExpression] = dict()
  equations: set[Equation] = set()
  for equation in equation_system.equations:
    simplified_equation = reduce_equation(equation, equation_system.solution)
    match (result := evaluate_equation(simplified_equation)):
      case Unsatisfiable():
        return Unsatisfiable()
      case Tautological():
        changed = True
        continue
      case SatisfiedBy(assignment):
        variable, expression = assignment
        if variable in solutions and solutions[variable] != expression:
          return Unsatisfiable()
        solutions[variable] = expression
        changed = True
        continue
      case Unknown():
        equations.add(simplified_equation)
        changed |= simplified_equation != equation
        continue
      case _:
        assert_never(f"Unknown solution type: {type(result)}")

  if changed:
    return EquationSystem(
        solution=solutions | equation_system.solution,
        unknowns=equation_system.unknowns,
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
        assert_never(f"Unexpected result type: {type(new_system)}")

  return equation_system

