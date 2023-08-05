# Copyright 2023 The JAX Authors.
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

"""
Linear sum assignment. Direct lax translation of scipy.optimize.linear_sum_assignment.
"""

from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

def linear_sum_assignment(
    cost : jax.Array,
    maximize : bool = False
  ) -> Tuple[jax.Array, jax.Array]:
  """
  Solve the linear sum assignment problem.

  Supports :func:`~jax.vmap` and :func:`~jax.jit`.

  Args:
    cost_matrix: The cost matrix of the bipartite graph.
    maximize: Calculates a maximum weight matching if true.

  Returns:
    An array of row indices and one of corresponding column indices giving the
    optimal assignment. The cost of the assignment can be computed as
    ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be sorted; in
    the case of a square cost matrix they will be equal to ``numpy.arange
    (cost_matrix.shape[0])``.

  See Also:
    scipy.optimize.linear_sum_assignment

  """
  transpose = cost.shape[1] < cost.shape[0]

  if cost.shape[0] == 0 or cost.shape[1] == 0:
    return jnp.array([]), jnp.array([])

  if transpose:
    cost = cost.T

  cost = lax.cond(maximize, lambda: -cost, lambda: cost).astype(float)

  u = jnp.full(cost.shape[0], 0., cost.dtype)
  v = jnp.full(cost.shape[1], 0., cost.dtype)
  path = jnp.full(cost.shape[1], -1, dtype=int)
  col4row = jnp.full(cost.shape[0], -1, dtype=int)
  row4col = jnp.full(cost.shape[1], -1, dtype=int)

  init = cost, u, v, path, row4col, col4row
  cost, u, v, path, row4col, col4row = lax.fori_loop(
    0,
    cost.shape[0],
    _lsa_body,
    init
  )

  if transpose:
    v = col4row.argsort()
    return col4row[v], v
  else:
    return jnp.arange(cost.shape[0]), col4row

def _find_short_augpath_while_body_inner_for(it, val):
  remaining, min_value, cost, i, u, v, shortest_path_costs, path, lowest, row4col, index = val

  j = remaining[it]

  r = min_value + cost[i, j] - u[i] - v[j]

  path = lax.cond(
    r < shortest_path_costs[j],
    lambda: path.at[j].set(i),
    lambda: path
  )
  shortest_path_costs = shortest_path_costs.at[j].min(r)

  index = lax.cond(
    (shortest_path_costs[j] < lowest) |
    ((shortest_path_costs[j] == lowest) & (row4col[j] == -1)),
    lambda: it,
    lambda: index
  )
  lowest = jnp.minimum(lowest, shortest_path_costs[j])

  outval = remaining, min_value, cost, i, u, v, shortest_path_costs, path, lowest, row4col, index
  return outval

def _find_short_augpath_while_body_tail(val):
  """
  Tail of _find_short_augpath_while_body, only to be executed if min_value != jnp.inf
  """
  remaining, index, row4col, sink, i, SC, num_remaining = val

  j = remaining[index]
  pred = row4col[j] == -1
  sink = lax.cond(pred, lambda: j, lambda: sink)
  i = lax.cond(jnp.logical_not(pred), lambda: row4col[j], lambda: i)

  SC = SC.at[j].set(True)
  num_remaining -= 1
  remaining = remaining.at[index].set(remaining[num_remaining])

  outval = remaining, index, row4col, sink, i, SC, num_remaining
  return outval

def _find_short_augpath_while_body(val):
  (cost, u, v, path, row4col, current_row, min_value, num_remaining, remaining, SR, SC, shortest_path_costs, sink) = val

  index = -1
  lowest = jnp.inf
  SR = SR.at[current_row].set(True)

  init = (remaining, min_value, cost, current_row, u, v, shortest_path_costs, path, lowest, row4col, index)
  output = lax.fori_loop(
    0,
    num_remaining,
    _find_short_augpath_while_body_inner_for,
    init
  )
  remaining, min_value, cost, current_row, u, v, shortest_path_costs, path, lowest, row4col, index = output

  min_value = lowest
  # infeasible cost matrix
  sink = lax.cond(min_value == jnp.inf, lambda:-1, lambda:sink)

  output = lax.cond(
    sink == -1,
    _find_short_augpath_while_body_tail,
    lambda a: a,
    (remaining, index, row4col, sink, current_row, SC, num_remaining)
  )
  remaining, index, row4col, sink, current_row, SC, num_remaining = output

  outval = (cost, u, v, path, row4col, current_row, min_value, num_remaining, remaining, SR, SC, shortest_path_costs, sink)
  return outval

def _find_short_augpath_while_cond(val):
  sink = val[-1]
  return sink == -1

def _find_augmenting_path(cost, u, v, path, row4col, current_row):
  min_value = 0
  num_remaining = cost.shape[1]
  remaining = jnp.arange(cost.shape[1])[::-1]

  SR = jnp.full(cost.shape[0], False, dtype=bool)
  SC = jnp.full(cost.shape[1], False, dtype=bool)

  shortest_path_costs = jnp.full(cost.shape[1], jnp.inf)

  sink = -1

  init = (cost, u, v, path, row4col, current_row, min_value, num_remaining, remaining, SR, SC, shortest_path_costs, sink)
  output = lax.while_loop(
    _find_short_augpath_while_cond,
    _find_short_augpath_while_body,
    init
  )
  (cost, u, v, path, row4col, current_row, min_value, num_remaining, remaining, SR, SC, shortest_path_costs, sink) = output

  return sink, min_value, SR, SC, shortest_path_costs, path

def _augment_previous_while_body(val):
  path, sink, row4col, col4row, current_row, _ = val

  i = path[sink]
  row4col = row4col.at[sink].set(i)

  col4row, sink = col4row.at[i].set(sink), col4row[i]
  breakvar = (i == current_row)

  outval = path, sink, row4col, col4row, current_row, breakvar
  return outval

def _augment_previous_while_cond(val):
  breakvar = val[-1]
  return jnp.logical_not(breakvar)

def _lsa_body(current_row, val):
  cost, u, v, path, row4col, col4row = val

  sink, min_value, SR, SC, shortest_path_costs, path = _find_augmenting_path(cost, u, v, path, row4col, current_row)

  u = u.at[current_row].add(min_value)

  mask = (SR & (jnp.arange(cost.shape[0]) != current_row)).astype(float)

  u = u + mask*(min_value - shortest_path_costs[col4row])

  mask = SC.astype(float)

  v = v + mask*(shortest_path_costs - min_value)

  breakvar = False
  init = path, sink, row4col, col4row, current_row, breakvar
  output = lax.while_loop(
    _augment_previous_while_cond,
    _augment_previous_while_body,
    init
  )
  path, sink, row4col, col4row, current_row, breakvar = output

  outval = cost, u, v, path, row4col, col4row
  return outval
