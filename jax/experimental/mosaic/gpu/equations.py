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

import abc
from collections.abc import Sequence
import dataclasses
import math
from typing import Any, Callable, assert_never, final

from . import fragmented_array as fa
from . import layouts as layouts_lib
from . import tcgen05


VariableKey = Any


@dataclasses.dataclass(frozen=True)
class Variable:
  """A variable is an abstract identifier.

  `key` is supposed to be hashable.
  """
  key: VariableKey

  def __str__(self):
    return f"V({self.key})"


class Constant(abc.ABC):
  """A constant is a known layout."""


@dataclasses.dataclass(frozen=True)
class RegisterLayout(Constant):
  """Wraps a known register layout."""

  value: fa.FragmentedLayout

  def __str__(self):
    return f"C({self.value})"


@dataclasses.dataclass(frozen=True)
class TMEMLayout(Constant):
  """Wraps a known TMEM layout."""

  value: tcgen05.TMEMLayout


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

  def __str__(self):
    return f"Reduce([{self.axes}], {self.expression})"


@dataclasses.dataclass(frozen=True)
class BroadcastInDim:
  expression: Expression
  axes: tuple[int, ...]
  shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Reshape:
  expression: Expression
  source_shape: tuple[int, ...]
  target_shape: tuple[int, ...]


Expression = (
    Variable
    | Constant
    | LeastReplicated
    | MostReplicated
    | Reduce
    | BroadcastInDim
    | Reshape
)


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

  consts = []
  unknowns = []
  for e in new_expressions:
    if not isinstance(e, Constant):
      unknowns.append(e)
      continue
    if not isinstance(e, RegisterLayout):
      raise ValueError(
          f"Reduction of non-register layout constant is not supported: {e}"
      )
    consts.append(e)

  if consts:
    const_red, *consts = consts
    red = const_red
    for cst in consts:
      red_value = reducer(red.value, cst.value)
      if red_value is None:
        # The layouts are not compatible up to replication, this expression
        # cannot be simplified.
        return Unsatisfiable()
      red = RegisterLayout(red_value)
  else:
    red = None

  constructor = type(input_expr)
  if red is not None:
    if unknowns:
      return constructor((red, *unknowns))
    return red

  return constructor(tuple(unknowns))


def reduce_broadcast_expression(
    broadcast: BroadcastInDim, assignments: dict[Variable, Constant]
) -> Expression | Unsatisfiable:
  def _check_shape_broadcast(shape: tuple[int, ...]) -> bool:
    for axis, s in zip(broadcast.axes, shape, strict=True):
      if broadcast.shape[axis] != s:
        return False
    return True

  reduced_expr = reduce_expression(broadcast.expression, assignments)
  match reduced_expr:
    case Unsatisfiable():
      return Unsatisfiable()
    case RegisterLayout(value=layout):
      match layout:
        case fa.WGSplatFragLayout(shape=shape):
          if not _check_shape_broadcast(shape):
            return Unsatisfiable()
          return RegisterLayout(fa.WGSplatFragLayout(shape=broadcast.shape))
        case _:
          return BroadcastInDim(
              expression=reduced_expr,
              axes=broadcast.axes,
              shape=broadcast.shape,
          )
    case _:
      return BroadcastInDim(
          expression=reduced_expr, axes=broadcast.axes, shape=broadcast.shape
      )


def reduce_reshape_expression(
    reshape: Reshape, assignments: dict[Variable, Constant]
) -> Expression | Unsatisfiable:
  reduced_expr = reduce_expression(reshape.expression, assignments)
  match reduced_expr:
    case Unsatisfiable():
      return Unsatisfiable()
    case RegisterLayout(value=layout):
      match layout:
        case fa.WGSplatFragLayout(shape=shape):
          assert math.prod(shape) == math.prod(reshape.target_shape)
          return RegisterLayout(
              fa.WGSplatFragLayout(shape=reshape.target_shape)
          )
        case fa.WGStridedFragLayout(shape=shape, vec_size=vec_size):
          assert math.prod(shape) == math.prod(reshape.target_shape)
          return RegisterLayout(
              fa.WGStridedFragLayout(
                  shape=reshape.target_shape, vec_size=vec_size
              )
          )
        case fa.TiledLayout() as tiled_layout:
          tile_shape = tiled_layout.base_tile_shape
          if len(reshape.target_shape) < len(tile_shape):
            return dataclasses.replace(reshape, expression=reduced_expr)
          # Even if the new shape is not perfectly tilable, it is possible that
          # we may be able to reshape the tiling itself in a way that is
          # compatible with the new shape. We do not handle this case at the
          # moment.
          for ts, s in zip(tile_shape, reshape.source_shape[-len(tile_shape):], strict=True):
            if s % ts != 0:
              return dataclasses.replace(reshape, expression=reduced_expr)

          # If minor tiled dimensions are modified, then reshaping is likely to
          # not be a no-op since the strides between tiles will change,
          # potentially mapping different elements to lanes and warps. We don't
          # attempt to handle this case at the moment.
          num_minor_tiled_dims = len(tile_shape) - 1
          source_minor_tiled_dims = reshape.source_shape[-num_minor_tiled_dims:]
          target_minor_tiled_dims = reshape.target_shape[-num_minor_tiled_dims:]
          major_tiled_dim = tile_shape[0]
          if (source_minor_tiled_dims != target_minor_tiled_dims or
              reshape.target_shape[-len(tile_shape)] % major_tiled_dim != 0):
            return dataclasses.replace(reshape, expression=reduced_expr)
          # At this point, we now that only non-tiled dimensions and/or the
          # majormost tiled dimensions may have changed. We also know that the
          # majormost tiled dimension is still tilable in the new shape.
          # Therefore, we can return the tiled layout as is.
          return RegisterLayout(tiled_layout)
    case _:
      return dataclasses.replace(reshape, expression=reduced_expr)


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
        case RegisterLayout(value=layout) if isinstance(layout, fa.TiledLayout):
          return RegisterLayout(layout.reduce(axes))
        case _:
          return Reduce(expression=reduced_expr, axes=axes)
    case BroadcastInDim():
      return reduce_broadcast_expression(expr, assignments)
    case Reshape():
      return reduce_reshape_expression(expr, assignments)
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

    if not isinstance(source, RegisterLayout) or not isinstance(
        target, RegisterLayout
    ):
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

  def __str__(self):
    return f"Relayout({self.source}  ⟶ {self.target})"


@dataclasses.dataclass(frozen=True)
class IsTransferable:
  """States that `source` layout must be transferable across memory spaces to `target` layout."""

  source: Expression
  target: Expression
  # TODO(allanrenucci): Can this be derived from the layouts?
  shape: tuple[int, int]

  def _is_valid_tmem_transfer(
      self, tmem_layout: tcgen05.TMEMLayout, reg_layout: fa.FragmentedLayout
  ) -> bool:
    columns = self.shape[1]
    packing = tmem_layout.vector_length
    if reg_layout in (fa.TCGEN05_LAYOUT, fa.TMEM_NATIVE_LAYOUT):
      return tmem_layout == tcgen05.tmem_default_layout(packing)
    if reg_layout == fa.WGMMA_LAYOUT:
      return tmem_layout == tcgen05.tmem_half_lane_layout(columns, packing)
    if reg_layout == tcgen05.fa_m64_collective_layout(columns):
      return tmem_layout == tcgen05.tmem_m64_collective_layout(columns, packing)
    return False

  def holds(self) -> bool | None:
    """Returns whether the constraint holds.

    Returns `None` if the constraint can't be checked.
    """
    source = self.source
    target = self.target

    if isinstance(source, TMEMLayout) and isinstance(target, RegisterLayout):
      return self._is_valid_tmem_transfer(source.value, target.value)
    if isinstance(target, TMEMLayout) and isinstance(source, RegisterLayout):
      return self._is_valid_tmem_transfer(target.value, source.value)
    if isinstance(source, TMEMLayout) and isinstance(target, TMEMLayout):
      return source == target
    if isinstance(target, Constant) and isinstance(source, Constant):
      source_type = type(source).__name__
      target_type = type(target).__name__
      raise NotImplementedError(f"Unsupported transfer: {source_type} -> {target_type}")

    return None

  def __str__(self):
    return f"IsTransferable({self.source}  ⟶ {self.target})"


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

  def __str__(self):
    return f"{self.lhs} ≠ {self.rhs}"


Constraint = Relayout | Distinct | IsTransferable


def reduce_constraint(
    constraint: Constraint, assignments: dict[Variable, Constant]
) -> Constraint | Tautological | Unsatisfiable:
  """Reduces a constraint."""
  match constraint:
    case Relayout(source=lhs, target=rhs):
      ...
    case Distinct(lhs=lhs, rhs=rhs):
      ...
    case IsTransferable(source=lhs, target=rhs, shape=_):
      ...
    case _ as never:
      assert_never(never)

  lhs_red = reduce_expression(lhs, assignments)
  rhs_red = reduce_expression(rhs, assignments)

  if isinstance(lhs_red, Unsatisfiable) or isinstance(rhs_red, Unsatisfiable):
    return Unsatisfiable()

  new_constraint: Constraint
  if isinstance(constraint, IsTransferable):
    new_constraint = IsTransferable(lhs_red, rhs_red, constraint.shape)
  else:
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
  constraints: Sequence[Constraint] = dataclasses.field(default_factory=list)

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
        case BroadcastInDim(expression=e):
          extract_variables(e)
        case Reshape(expression=e):
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
        case IsTransferable(source=source, target=target, shape=_):
          extract_variables(source)
          extract_variables(target)
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
        constraints=[*self.constraints, *other.constraints],
    )

  def __str__(self):
    r = "EquationSystem\n"
    r += "  assignments:\n"
    for assignment, constant in self.assignments.items():
      r += f"    {assignment} ⟵ {constant}\n"
    r += "  equations:\n"
    for equation in self.equations:
      r += f"    {equation}\n"
    r += "  constraints:\n"
    for constraint in self.constraints:
      r += f"    {constraint}\n"
    return r

@final
class Unsatisfiable:
  def __and__(self, other: EquationSystem | Unsatisfiable) -> Unsatisfiable:
    return self


@dataclasses.dataclass(frozen=True)
class SatisfiedBy:
  assignment: tuple[Variable, Constant]


@dataclasses.dataclass(frozen=True)
class Unknown:
  equation: Equation


class Tautological:
  ...


def non_splat_variables(
    constraints: Sequence[Constraint],
) -> dict[Variable, Constant]:
  """Returns a map var->splat_layout for all vars distinct from a splat."""
  result: dict[Variable, Constant] = {}
  for constraint in constraints:
    match constraint:
      case Distinct(lhs=lhs, rhs=rhs):
        if (
            isinstance(lhs, Variable)
            and isinstance(rhs, RegisterLayout)
            and isinstance(rhs.value, fa.WGSplatFragLayout)
        ):
          result[lhs] = rhs
        if (
            isinstance(rhs, Variable)
            and isinstance(lhs, RegisterLayout)
            and isinstance(lhs.value, fa.WGSplatFragLayout)
        ):
          result[rhs] = lhs
  return result


# The result of reducing an equation---and by extension, a system of
# equations. An equation can either be unsatisfiable (i.e. there exists no
# assignment for which it holds), satisfied by an assignment, unknown (i.e.
# still undetermined), or tautological (i.e. the equation is guaranteed to
# hold for any assignment).
Solution = Unsatisfiable | SatisfiedBy | Unknown | Tautological


def _has_relayout_of_non_splat_to_splat(constraints: Sequence[Constraint]) -> bool:
  """Returns whether the constraints imply a non-splat to splat relayout.

  Such relayouts are impossible and this helps shortcut the search.

  If this function returns False, this doesn't necessarily mean that there are
  no non-splat to splat relayouts, just that this is not known yet.
  """
  non_splat = non_splat_variables(constraints)
  if not non_splat:
    return False

  def is_constant_splat(e) -> bool:
    return isinstance(e, RegisterLayout) and isinstance(
        e.value, fa.WGSplatFragLayout
    )

  for constraint in constraints:
    match constraint:
      case Relayout(source=source, target=target):
        if source in non_splat and is_constant_splat(target):
          return True
      case _:
        pass
  return False


def saturate_distinct_from_splat(
    equation_system: EquationSystem,
) -> EquationSystem | Unsatisfiable:
  """Adds transitive Distinct constraints for all non-splat variables.

  Given `n` variables `l0`, ... `l{n-1}`, and a set of relayouts
  `{ Relayout(l{i}, l{i+1}) : 0 <= i < n }`, if we also know that
  `l{0}` is not splat, then we can automatically deduce that none of
  `l0`, ..., `l{n-1}` are splat either.

  This helps us quickly conclude that a system is unsatisfiable in cases where
  a non-splat variable is transitively relaid out into a splat layout.
  """
  non_splat = non_splat_variables(equation_system.constraints)
  new_constraints: list[Constraint] = []
  new_non_splat_found = len(non_splat) > 0

  while new_non_splat_found:
    new_non_splat_found = False
    for constraint in equation_system.constraints:
      match constraint:
        case Relayout(source=source, target=target):
          if isinstance(target, Variable) and source in non_splat and target not in non_splat:
            new_non_splat_found = True
            assert isinstance(source, Variable)
            splat_layout = non_splat[source]
            non_splat[target] = splat_layout
            new_constraints.append(Distinct(lhs=target, rhs=splat_layout))
        case _:
          pass
  return equation_system & EquationSystem(constraints=new_constraints)


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

  # Shortcut for a specific case of unsatisfiability. This shortcut
  # drastically reduces the size of the search space.
  if _has_relayout_of_non_splat_to_splat(equation_system.constraints):
    return Unsatisfiable()

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
