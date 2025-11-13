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

# mypy has been causing more problems than it solves here. Disable it for these
# files. We have pytype checks anyway.
# mypy: ignore-errors

from __future__ import annotations

import abc
from collections.abc import Sequence
import dataclasses
import math
from typing import Any, Callable, assert_never, final

from . import fragmented_array as fa
from . import launch_context as lc
from . import layouts as layouts_lib
from . import inference_utils
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

  def __str__(self):
    return f"C({self.value})"


@dataclasses.dataclass(frozen=True)
class SMEMTiling(Constant):
  """Wraps a known SMEM Tile Transform.

  If an SMEM reference may, in principle, have transforms but should not be
  tiled, then `value` is `None`.
  """

  value: lc.TileTransform | None

  def __str__(self):
    return f"C({self.value})"


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


@dataclasses.dataclass(frozen=True)
class Transpose:
  expression: Expression

  def __str__(self):
    return f"T({self.expression})"


Expression = (
    Variable
    | Constant
    | LeastReplicated
    | MostReplicated
    | Reduce
    | BroadcastInDim
    | Reshape
    | Transpose
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
      return dataclasses.replace(reshape, expression=reduced_expr)  # pytype: disable=bad-return-type


def reduce_transpose_expression(
    transpose: Transpose, assignments: dict[Variable, Constant]
) -> Expression | Unsatisfiable:
  reduced_expr = reduce_expression(transpose.expression, assignments)
  match reduced_expr:
    case Unsatisfiable():
      return Unsatisfiable()
    case SMEMTiling(value=tile_transform):
      if tile_transform is None:
        return SMEMTiling(None)
      tiling = tile_transform.tiling
      if len(tiling) != 2:
        raise NotImplementedError(
            f"Only 2D tilings are supported, got {len(tiling)}"
        )
      return SMEMTiling(lc.TileTransform(tiling[::-1]))
    case _:
      return Transpose(expression=reduced_expr)


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
    case Transpose():
      return reduce_transpose_expression(expr, assignments)
    case _:
      assert_never(expr)


_SUPPORTED_TILED_RELAYOUTS = frozenset([
    # Transposed layouts.
    (fa.WGMMA_LAYOUT, fa.WGMMA_TRANSPOSED_LAYOUT),
    (fa.WGMMA_TRANSPOSED_LAYOUT, fa.WGMMA_LAYOUT),
    (fa.TCGEN05_LAYOUT, fa.TCGEN05_TRANSPOSED_LAYOUT),
    (fa.TCGEN05_TRANSPOSED_LAYOUT, fa.TCGEN05_LAYOUT),
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
      case fa.WGSplatFragLayout() as splat, fa.WGStridedFragLayout() as strided:
        return splat.shape == strided.shape
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
  shape: tuple[int, ...]

  def supported_tmem_transfers(
      self, packing: int
  ) -> set[tuple[tcgen05.TMEMLayout, fa.FragmentedLayout]]:
    """Returns the set of supported TMEM <-> Register transfers."""
    assert len(self.shape) == 2
    columns = self.shape[1]
    tmem_default_layout = tcgen05.tmem_default_layout(packing)
    return {
        (tmem_default_layout, fa.TCGEN05_LAYOUT),
        (tmem_default_layout, fa.TMEM_NATIVE_LAYOUT),
        (tcgen05.tmem_half_lane_layout(columns, packing), fa.WGMMA_LAYOUT),
        (
            tcgen05.tmem_m64_collective_layout(columns, packing),
            tcgen05.fa_m64_collective_layout(columns),
        ),
    }

  def _is_valid_tmem_transfer(
      self, tmem_layout: tcgen05.TMEMLayout, reg_layout: fa.FragmentedLayout
  ) -> bool:
    packing = tmem_layout.vector_length
    return (tmem_layout, reg_layout) in self.supported_tmem_transfers(packing)

  def _is_valid_smem_transfer(
      self,
      smem_layout: lc.TileTransform | None,
      reg_layout: fa.FragmentedLayout,
  ) -> bool:
    # TODO(b/447079781): This is way too restrictive. We need to make it more
    # precise by:
    # - Consider whether the op is annotated with optimized copies or not.
    # - If copies do not have to be optimized, always return True.
    # - If copies have to be optimized, determine if the transfer is optimal by
    #   calling fragmented_array.plan_tiled_transfer.
    if inference_utils.is_mma_layout(reg_layout):
      return smem_layout is not None and len(smem_layout.tiling) == 2
    return smem_layout is None

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
    if isinstance(source, SMEMTiling) and isinstance(target, RegisterLayout):
      return self._is_valid_smem_transfer(source.value, target.value)
    if isinstance(target, SMEMTiling) and isinstance(source, RegisterLayout):
      return self._is_valid_smem_transfer(target.value, source.value)
    if isinstance(target, Constant) and isinstance(source, Constant):
      source_type = type(source).__name__
      target_type = type(target).__name__
      raise NotImplementedError(f"Unsupported transfer: {source_type} -> {target_type}")

    return None

  def __str__(self):
    return f"IsTransferable({self.source}  ⟶ {self.target})"


@dataclasses.dataclass(frozen=True)
class NotOfType:
  """States that `expr` is not an instance of `type`."""

  expr: Expression
  type: type[fa.FragmentedLayout]

  def holds(self) -> bool | None:
    """Whether the distinctiveness constraint holds.

    Returns `None` if the constraint can't be checked.
    """
    if not isinstance(self.expr, Constant):
      return None
    if not isinstance(self.expr, RegisterLayout):
      return True
    return not isinstance(self.expr.value, self.type)

  def __str__(self):
    return f"type({self.expr}) ≠ {self.type.__name__}"


@dataclasses.dataclass(frozen=True)
class Divides:
  """States that the `expr` tiling is a divisor of `tiling_multiple`.

  That is to say that, for each tiled dimension in `expr`, the dimension must
  divide its corresponding dimension in `tiling_multiple` starting from the
  tail.

  If `tiling_multiple` contains more dimensions than `expr`, then
  the extra dimensions in `tiling_multiple` are ignored for the purposes of the
  check.

  `expr` is not allowed to contain more dimensions than `tiling_multiple`, and
  this constraint therefore also constrains the rank of `expr`.
  """
  expr: Expression
  tiling_multiple: tuple[int, ...]

  def holds(self) -> bool | None:
    match self.expr:
      case SMEMTiling(value=None):
        # If there is no tiling, then this holds trivially.
        return True
      case SMEMTiling(value=lc.TileTransform(tiling=t)):
        tiling = t
      case RegisterLayout(value=fa.TiledLayout() as layout):
        tiling = layout.base_tile_shape
      case TMEMLayout(value):
        tiling = value.base_tile_shape
      case _:
        return None

    if len(tiling) > len(self.tiling_multiple):
      # The rank of the tiling is larger than the rank of the constraint. This
      # is not allowed.
      return False

    for size, multiple in zip(reversed(tiling), reversed(self.tiling_multiple)):
      if multiple % size:
        return False
    return True

  def __str__(self):
    return f"{self.tiling_multiple} % {self.expr} == 0"


Constraint = Relayout | NotOfType | IsTransferable | Divides


def reduce_constraint(
    constraint: Constraint, assignments: dict[Variable, Constant]
) -> Constraint | Tautological | Unsatisfiable:
  """Reduces a constraint."""

  new_constraint: Constraint
  match constraint:
    case Relayout(source=source, target=target):
      source_red = reduce_expression(source, assignments)
      target_red = reduce_expression(target, assignments)
      if isinstance(source_red, Unsatisfiable) or isinstance(
          target_red, Unsatisfiable
      ):
        return Unsatisfiable()
      new_constraint = Relayout(source_red, target_red)
    case NotOfType(expr=expr, type=type):
      expr_red = reduce_expression(expr, assignments)
      if isinstance(expr_red, Unsatisfiable):
        return Unsatisfiable()
      new_constraint = NotOfType(expr_red, type)
    case IsTransferable(source=source, target=target, shape=shape):
      source_red = reduce_expression(source, assignments)
      target_red = reduce_expression(target, assignments)
      if isinstance(source_red, Unsatisfiable) or isinstance(target_red, Unsatisfiable):
        return Unsatisfiable()
      new_constraint = IsTransferable(source_red, target_red, shape)
    case Divides(expr=expr, tiling_multiple=tiling_multiple):
      expr_red = reduce_expression(expr, assignments)
      if isinstance(expr_red, Unsatisfiable):
        return Unsatisfiable()
      new_constraint = Divides(expr_red, tiling_multiple)
    case _ as never:
      assert_never(never)

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
        case Transpose(expression=e):
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
        case NotOfType(expr=expr):
          extract_variables(expr)
        case IsTransferable(source=source, target=target, shape=_):
          extract_variables(source)
          extract_variables(target)
        case Divides(expr=expr):
          extract_variables(expr)
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
) -> set[Variable]:
  """Returns a all vars distinct from a splat."""
  vars: set[Variable] = set()
  for constraint in constraints:
    match constraint:
      case NotOfType(expr=Variable() as var, type=fa.WGSplatFragLayout):
        assert isinstance(var, Variable)  # make pytype happy
        vars.add(var)
  return vars


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
  """Adds transitive NotOfType constraints for all non-splat variables.

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
          if (
              isinstance(target, Variable)
              and source in non_splat
              and target not in non_splat
          ):
            new_non_splat_found = True
            non_splat.add(target)
            new_constraints.append(NotOfType(target, fa.WGSplatFragLayout))
        case _:
          pass
  return equation_system & EquationSystem(constraints=new_constraints)


def compute_transitively_equal_vars(
    system: EquationSystem,
) -> dict[Variable, list[Variable]]:
  """Computes all transitively equal variables in an equation system.

  The output dictionary maps each variable that appears in equations in the
  equation system to all the variables it is transitively equal to.
  """
  # The equality relations between variables form a graph where variables are
  # nodes and an equation `v1 == v2` forms an edge. All variables in a
  # connected component are transitively equal. We use a Union-Find data
  # structure with path compression to efficiently find these connected
  # components (i.e., equivalence classes).
  parent: dict[Variable, Variable] = {}
  def find(v: Variable) -> Variable:
    if v not in parent:
      parent[v] = v
    if parent[v] != v:
      parent[v] = find(parent[v])
    return parent[v]

  def union(v1: Variable, v2: Variable):
    root1 = find(v1)
    root2 = find(v2)
    if root1 != root2:
      parent[root2] = root1

  all_vars: set[Variable] = set()
  for eq in system.equations:
    if isinstance(eq.lhs, Variable) and isinstance(eq.rhs, Variable):
      all_vars.add(eq.lhs)
      all_vars.add(eq.rhs)
      union(eq.lhs, eq.rhs)

  # Group variables by their component representative.
  components: dict[Variable, list[Variable]] = {}
  for v in sorted(all_vars, key=str):
    root = find(v)
    components.setdefault(root, []).append(v)

  equal_vars: dict[Variable, list[Variable]] = {}
  for component_vars in components.values():
    for v in component_vars:
      equal_vars[v] = [other for other in component_vars if other != v]

  return equal_vars


def saturate_divides_constraints_for_equal_vars(
    system: EquationSystem,
) -> EquationSystem:
  """Saturates Divides constraints between all transitively equal vars.
  """
  equal_vars = compute_transitively_equal_vars(system)
  new_constraints: list[Constraint] = []
  for constraint in system.constraints:
    new_constraints.append(constraint)
    match constraint:
      case Divides(expr=expr, tiling_multiple=tiling_multiple):
        if isinstance(expr, Variable):
          for equal_var in equal_vars.get(expr, []):
            new_constraints.append(Divides(equal_var, tiling_multiple))
      case _:
        pass
  new_constraints = merge_divides_constraints(new_constraints)
  return dataclasses.replace(system, constraints=new_constraints)


# TODO(bchetioui): clean up API.
def merge_divides_constraints(constraints: Sequence[Constraint]) -> list[Constraint]:
  """Merges Divides constraints that can be merged."""
  result: list[Constraint] = []
  var_to_tiling_multiples : dict[Variable, tuple[int, ...]] = {}
  for constraint in constraints:
    match constraint:
      case Divides(expr=Variable() as v, tiling_multiple=tiling_multiple):
        assert isinstance(v, Variable)  # make pytype happy
        if (previous_tiling_multiple := var_to_tiling_multiples.get(v)) is None:
          var_to_tiling_multiples[v] = tiling_multiple
          continue
        # If the two tuples are of different lengths, the larger tuple will
        # be truncated (removing initial multiples) to the length of the
        # smaller tuple. This preserves the semantics of the Divides constraints
        # where a tiling's rank cannot exceed the size of tiling_multiple.
        min_len = min(len(tiling_multiple), len(previous_tiling_multiple))
        new_tiling_multiple = []
        if min_len > 0:
          for x, y in zip(tiling_multiple[-min_len:], previous_tiling_multiple[-min_len:], strict=True):
            new_tiling_multiple.append(math.gcd(x, y))
        var_to_tiling_multiples[v] = tuple(new_tiling_multiple)
      case _:
        result.append(constraint)
  for expr, tiling_multiple in var_to_tiling_multiples.items():
    result.append(Divides(expr, tiling_multiple))
  return result


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
  assignments: dict[Variable, Constant] = {}
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

  new_constraints = merge_divides_constraints(constraints)
  changed |= len(new_constraints) != len(constraints)
  constraints = new_constraints

  # Shortcut for a specific case of unsatisfiability. This shortcut
  # drastically reduces the size of the search space.
  if _has_relayout_of_non_splat_to_splat(constraints):
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
