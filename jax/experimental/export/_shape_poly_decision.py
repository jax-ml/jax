# Copyright 2022 The JAX Authors.
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
"""Shape polymorphism support for deciding inequalities of symbolic dimensions.

"""

from __future__ import annotations

import collections
from collections.abc import Sequence
import itertools
import math
from typing import Callable

import numpy as np

from jax.experimental.export import _shape_poly
from jax.experimental.export._shape_poly import (
  _DimExpr, _DimMon, _DimAtom,
  SymbolicScope,
  DimSize,
  InconclusiveDimensionOperation,
)


def geq_decision(e1: DimSize, e2: DimSize,
                 cmp_str: Callable[[], str]) -> bool:
  """Implements `e1 >= e2`.

  Args:
    e1, e2: the expressions to compare for greater-equal
    cmp_str: a callable such that `cmp_str()` describes the comparison
      for error messages, e.g., "a <= b". Without this all comparisions would
      be reported as ">=".

  Raises InconclusiveDimensionOperation if the result is not conclusive.
  """
  if isinstance(e1, _DimExpr):
    scope = e1.scope
    if isinstance(e2, _DimExpr):
      scope._check_same_scope(e2, f"when comparing {cmp_str()}")
  elif isinstance(e2, _DimExpr):
    scope = e2.scope
  else:
    return int(e1) >= int(e2)
  decision = _DecisionByElimination(scope)
  lb, ub = decision.bounds(e1 - e2, _stop_early_for_geq0)
  if lb >= 0:
    return True
  if ub < 0:
    return False

  if scope._explicit_constraints:
    describe_scope = f"\nUsing symbolic scope {scope}"
  else:
    describe_scope = ""
  raise InconclusiveDimensionOperation(
      f"Symbolic dimension comparison {cmp_str()} is inconclusive.{describe_scope}")

_shape_poly._geq_decision = geq_decision

def _stop_early_for_geq0(lb, ub):
  return lb >= 0 or ub < 0

def bounds_decision(e: DimSize,
                    stop_early: Callable[[float, float], bool] | None) -> tuple[float, float]:
  if not isinstance(e, _DimExpr):
    return (int(e), int(e))
  decision = _DecisionByElimination(e.scope)
  return decision.bounds(e, stop_early)

_shape_poly._bounds_decision = bounds_decision


class _DecisionByElimination:
  """A decision procedure based on elimination of terms.

  Given an expression `e = m_k*m + rest_e` for which we want to compute bounds,
  and a constraint `c = m_c*m + rest_c >= 0`,

  Let `e0 = abs(m_c)*e - sgn(m_c)*m_k*c`. (Note that we eliminated `m` from
  `e0`, since `abs(m_c)*m_k = sgn(m_c)*m_k*m_c`.)

  Since `c >= 0`,
  if `sgn(m_c)*m_k > 0`:
     then `abs(m_c)*e >= e0`, hence, `LB(e) >= ceil(LB(e0) / abs(m_c))`,

  if `sgn(m_c)*m_k < 0`
    then `abs(m_c)*e <= e0`, hence, `UB(e) <= floor(UB(e0) / abs(m_c))`,
  """
  def __init__(self, scope: SymbolicScope):
    self.scope = scope
    self._processed_for_internal_constraints: set[_DimMon] = set()
    # The other fields are for keeping an efficient representation of
    # the explicit constraints.
    self._term_bounds: dict[_DimMon, tuple[float, float]] = {}
    # The _expr_constraints represents a set of constraints that are not
    # just simple monomials. The set is represented as a mapping from a
    # monomial "m" to tuples (k, c) where "c >= 0" represents a constraint that
    # has "m" as the leading monomial with coefficient "k".
    self._expr_constraints: dict[_DimMon, set[tuple[int, _DimExpr]]] = collections.defaultdict(set)

    # TODO: find a way to reuse the state reflecting the explicit constraints
    # We sort the constraints, so that the results of the heuristics do not
    # depend on the order in which the user writes the constraints.
    for c, c_str in sorted(scope._explicit_constraints,
                           key=lambda c: c[0]._monomials_sorted):
      self.add_constraint(c, 0, c_str)

  def add_constraint(self,
                     e1: _DimExpr | int | float,
                     e2: _DimExpr | int | float,
                     constraint_str: str | None = None):
    """Adds a constraint "e1 >= e2" to the internal state."""
    if isinstance(e1, float):
      if np.isinf(e1) and e1 >= 0: return
      assert e1 == np.floor(e1)
      e1 = int(e1)
    if isinstance(e2, float):
      if np.isinf(e2) and e2 <= 0: return
      e2 = int(e2)
    e = e1 if isinstance(e2, (int, float)) and e2 == 0 else e1 - e2
    if constraint_str is None:
      constraint_str = f"{e1} >= {e2}"
    if (const := _DimExpr.to_constant(e)) is not None:
      if const < 0:
        raise ValueError(f"Unsatisfiable constraint: {constraint_str}")
      return
    assert isinstance(e, _DimExpr)
    self._add_to_state(e, constraint_str)
    combinations = self._combine_with_existing_constraints(e, constraint_str)
    for a in combinations:
      self._add_to_state(a, f"{a} >= 0")


  def _combine_with_existing_constraints(self,
                                         e: _DimExpr,
                                         debug_str: str) -> set[_DimExpr]:
    # This combines `e` with those constraints already present. The resulting
    # constraints are not scanned for new internal constraints (because there
    # are no new monomials), but they are also not combined further.
    # TODO: this results in incompleteness, but it is probably a good
    # compromise.
    combinations: set[_DimExpr] = set()
    def acc_combination(e: _DimExpr | int):
      if (const := _DimExpr.to_constant(e)) is not None:
        if const < 0:
          raise ValueError(f"Unsatisfiable constraints: {debug_str}")
      else:
        combinations.add(e)  # type: ignore

    # First combine with the existing monomial constraints
    for e_m, e_c in e.monomials():
      if e_m.degree == 0: continue
      m_lb, m_ub = self._term_bounds.get(e_m, (-np.inf, np.inf))
      if e_c > 0:
        if m_ub < np.inf:
          e_minus_m = _DimExpr._merge_sorted_terms(e._monomials_sorted, 0, 1,
                                                   [(e_m, e_c)], 0, -1)
          e_minus_m_ub = _DimExpr._merge_sorted_terms(e_minus_m, 0, 1,
                                                      [(_DimMon(), 1)], 0, e_c * int(m_ub))
          acc_combination(_DimExpr(dict(e_minus_m_ub), e.scope))
      else:
        if m_lb > -np.inf:
          e_minus_m = _DimExpr._merge_sorted_terms(e._monomials_sorted, 0, 1,
                                                   [(e_m, e_c)], 0, -1)
          e_minus_m_lb = _DimExpr._merge_sorted_terms(e_minus_m, 0, 1,
                                                      [(_DimMon(), 1)], 0, e_c * int(m_lb))
          acc_combination(_DimExpr(dict(e_minus_m_lb), e.scope))

    for prev_constraints in self._expr_constraints.values():
      for _, prev in prev_constraints:
        # Compose "e" with "prev" if they have one monomial with different
        # signs
        for e_m, e_c in e.monomials():
          if e_m.degree == 0: continue
          prev_c = prev._coeffs.get(e_m)
          if prev_c is not None and prev_c * e_c < 0:
            new_constraint = _DimExpr(
                dict(_DimExpr._merge_sorted_terms(e._monomials_sorted, 0, abs(prev_c),
                                                  prev._monomials_sorted, 0, abs(e_c))),
                e.scope)
            acc_combination(new_constraint)
            break

    return combinations

  def _add_to_state(self, e: _DimExpr,
                    constraint_str: str):
    """Updates the internal state to reflect "e >= 0". """
    assert _DimExpr.to_constant(e) is None
    for m, m_c in e.monomials():
      if m.degree == 0: continue
      _add_internal_constraints(self, m, e.scope)

    if (mon_factors := e.to_single_term()) is not None:
      n, mon_c, mon = mon_factors
      bounds = self._term_bounds.get(mon, (- np.inf, np.inf))
      if mon_c > 0:
        mon_ge = int(np.ceil(- n / mon_c))
        new_bounds = (max(mon_ge, bounds[0]), bounds[1])
      else:
        le = int(np.floor(-n / mon_c))
        new_bounds = (bounds[0], min(le, bounds[1]))
      if new_bounds[0] > new_bounds[1]:
        raise ValueError(f"Unsatisfiable constraint: {constraint_str}")

      self._term_bounds[mon] = new_bounds
      return

    lead_m, lead_m_c = e.leading_term
    self._expr_constraints[lead_m].add((lead_m_c, e))

  def bounds(self, e: DimSize,
             stop_early: Callable[[float, float], bool] | None
             ) -> tuple[float, float]:
    """Returns the lower and upper bounds, or -+inf.

    See more details in `_shape_poly.bounds_decision`.
    """
    if (const := _DimExpr.to_constant(e)) is not None:
      return (const, const)
    assert isinstance(e, _DimExpr)
    cache_key = (e, stop_early)
    if (res := self.scope._bounds_cache.get(cache_key)) is not None: return res
    res = self._bounds_for_sorted_terms(e.scope, e._monomials_sorted, 0, stop_early)
    self.scope._bounds_cache[cache_key] = res
    return res

  def _bounds_for_sorted_terms(self,
                               scope: SymbolicScope,
                               e: Sequence[tuple[_DimMon, int]],
                               i: int,
                               stop_early: Callable[[float, float], bool] | None) -> tuple[float, float]:
    """The lower and upper bounds of e[i:].

    See comments about soundness and `cmp_with` in the `_shape_poly.bounds_decision`` method.
    """
    if i >= len(e): return (0, 0)

    m, m_c = e[i]
    if len(m) == 0:  # A constant
      assert i == len(e) - 1  # Must be last
      return (m_c, m_c)

    _add_internal_constraints(self, m, scope)
    lb = -np.inf
    ub = np.inf

    # Look among the term bounds
    if m in self._term_bounds:
      m_lb, m_ub = self._term_bounds.get(m, (- np.inf, np.inf))
      rest_lb, rest_ub = self._bounds_for_sorted_terms(scope, e, i + 1, None)
      if m_c > 0:
        lb = max(lb, m_c * m_lb + rest_lb)
        ub = min(ub, m_c * m_ub + rest_ub)
      else:
        lb = max(lb, m_c * m_ub + rest_lb)
        ub = min(ub, m_c * m_lb + rest_ub)

      if stop_early is not None and stop_early(lb, ub): return (lb, ub)

    # Now look through the _expr_constraints
    if m in self._expr_constraints:
      for m_k, c in self._expr_constraints[m]:
        # A complex expression. See comments from top of class.
        sgn_m_k = 1 if m_k > 0 else -1
        abs_m_k = m_k * sgn_m_k
        # The recursive call has a smaller leading monomial, because we are only
        # looking at the tail of e, and in c the largest monomial is m, and the
        # merging will cancel the m.
        rest = _DimExpr._merge_sorted_terms(e, i, abs_m_k,
                                            c._monomials_sorted, 0, - sgn_m_k * m_c)
        rest_lb, rest_ub = self._bounds_for_sorted_terms(scope, rest, 0, None)
        if m_c / m_k > 0:
          lb = max(lb, np.ceil(rest_lb / abs_m_k))
        else:
          ub = min(ub, np.floor(rest_ub / abs_m_k))
        if stop_early is not None and stop_early(lb, ub): return (lb, ub)

    # Now look for special rules for atoms
    if (m_a := m.to_atom()) is not None:
      if m_a.operation in [_DimAtom.MAX, _DimAtom.MIN]:
        # m_c*MAX(op1, op2) + rest_e >= max(m_c * op1 + rest_e, m_c * op2 + rest_e)
        #   if m_c > 0. Similar rules for when m_c < 0 and for MIN.
        op1, op2 = m_a.operands
        rest1 = _DimExpr._merge_sorted_terms(e, i + 1, 1,
                                             op1._monomials_sorted, 0, m_c)
        rest2 = _DimExpr._merge_sorted_terms(e, i + 1, 1,
                                             op2._monomials_sorted, 0, m_c)
        rest1_lb, rest1_ub = self._bounds_for_sorted_terms(scope, rest1, 0, None)
        rest2_lb, rest2_ub = self._bounds_for_sorted_terms(scope, rest2, 0, None)
        like_max = (m_c > 0 if m_a.operation == _DimAtom.MAX else m_c < 0)
        if like_max:
          lb = max(lb, max(rest1_lb, rest2_lb))
          ub = min(ub, max(rest1_ub, rest2_ub))
        else:
          lb = max(lb, min(rest1_lb, rest2_lb))
          ub = min(ub, min(rest1_ub, rest2_ub))
        if stop_early is not None and stop_early(lb, ub): return (lb, ub)

    return lb, ub

def _add_internal_constraints(decision: _DecisionByElimination, m: _DimMon, scope: SymbolicScope):
  """Adds the internal constraints for the monomial `m`."""
  if m in decision._processed_for_internal_constraints: return
  decision._processed_for_internal_constraints.add(m)
  m_e = _DimExpr.from_monomial(m, 1, scope)  # m as a _DimExpr
  a = m.to_atom()
  if a is None:
    # This is a multiplication of atoms. Try to compute bounds based on
    # the bounds of the atoms.
    bounds = []
    for a, exp in m.items():
      a_l, a_u = decision.bounds(_DimExpr.from_monomial(_DimMon.from_atom(a, 1),
                                                        1, scope), None)
      assert a_l <= a_u
      bounds.append((a_l ** exp, a_u ** exp))

    candidate_bounds = [math.prod(atom_bounds)
                        for atom_bounds in itertools.product(*bounds)]
    m_l = min(*candidate_bounds)
    m_u = max(*candidate_bounds)
    decision.add_constraint(m_e, m_l)
    decision.add_constraint(m_u, m_e)
    return

  # It is an atom, is it a variable?
  if (v := a.to_var()) is not None:
    decision.add_constraint(m_e, 1)  # v >= 1
    return

  if a.operation == _DimAtom.MOD:
    op1, op2 = a.operands
    op2_b_l, op2_b_u = decision.bounds(op2, _stop_early_for_geq0)
    if op2_b_l > 0:  # positive divisor
      decision.add_constraint(m_e, 0)  # m >= 0
      decision.add_constraint(op2 - 1, m_e)  # m <= op2 - 1
      decision.add_constraint(op2_b_u - 1, m_e)
    elif op2_b_u < 0:  # negative divisor
      decision.add_constraint(m_e, op2 + 1)  # m >= op2 + 1
      decision.add_constraint(m_e, op2_b_l + 1)
      decision.add_constraint(0, m_e)  # m <= 0
    return

  if a.operation == _DimAtom.FLOORDIV:
    op1, op2 = a.operands
    (op1_l, op1_u) = decision.bounds(op1, None)
    (op2_l, op2_u) = decision.bounds(op2, None)

    def math_floor_with_inf(a: float, b: float):  # math.floor, but aware of inf
      # When either a or b are infinite, the results represent the limit
      # of "a // b".
      assert b != 0
      if not np.isinf(b):  # divisor b is finite
        if not np.isinf(a):
          return math.floor(a / b)
        # a is infinite, b is finite
        return -np.inf if (a >= 0) != (b >= 0) else np.inf
      elif not np.isinf(a):  # dividend a is finite and divisor b is infinite
        return -1 if (a >= 0) != (b >= 0) else 0
      else:  # both dividend and divisor are infinite
        return -np.inf if (a >= 0) != (b >= 0) else np.inf

    # Same reasoning as for multiplication: the bounds are among the cross-product
    # of the bounds.
    candidate_bounds = [math_floor_with_inf(op1_l, op2_l),
                        math_floor_with_inf(op1_l, op2_u),
                        math_floor_with_inf(op1_u, op2_l),
                        math_floor_with_inf(op1_u, op2_u)]
    m_l = min(*candidate_bounds)
    m_u = max(*candidate_bounds)
    decision.add_constraint(m_e, m_l)
    decision.add_constraint(m_u, m_e)
    if op2_l >= 0:
      if decision.bounds(op1, _stop_early_for_geq0)[0] >= 0:
        decision.add_constraint(m_e, 0)
      mod_e = _DimExpr.from_operation(_DimAtom.MOD, op1, op2,
                                      scope=scope)
      combined = op2 * m_e + mod_e
      decision.add_constraint(op1, combined)
      decision.add_constraint(combined, op1)
    return

  if a.operation == _DimAtom.MAX:
    op1, op2 = a.operands
    op1_b_l, op1_b_u = decision.bounds(op1, None)
    op2_b_l, op2_b_u = decision.bounds(op2, None)
    decision.add_constraint(m_e, max(op1_b_l, op2_b_l))
    decision.add_constraint(max(op1_b_u, op2_b_u), m_e)
    decision.add_constraint(m_e, op1)
    decision.add_constraint(m_e, op2)

  if a.operation == _DimAtom.MIN:
    op1, op2 = a.operands
    op1_b_l, op1_b_u = decision.bounds(op1, None)
    op2_b_l, op2_b_u = decision.bounds(op2, None)
    decision.add_constraint(m_e, min(op1_b_l, op2_b_l))
    decision.add_constraint(min(op1_b_u, op2_b_u), m_e)
    decision.add_constraint(op1, m_e)
    decision.add_constraint(op2, m_e)
