# Copyright 2020 Google LLC
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
"""Module for the propagate custom Jaxpr interpreter.
The propagate Jaxpr interpreter converts a Jaxpr to a directed graph where
vars are nodes and primitives are edges. It initializes invars and outvars with
Cells (an interface defined below), where a Cell encapsulates a value (or a set
of values) that a node in the graph can take on, and the Cell is computed from
neighboring Cells, using a set of propagation rules for each primtive.Each rule
indicates whether the propagation has been completed for the given edge.
If so, the propagate interpreter continues on to that primitive's neighbors
in the graph. Propagation continues until there are Cells for every node, or
when no further progress can be made. Finally, Cell values for all nodes in the
graph are returned.
"""
import abc
import collections
import functools
import itertools as it
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Type, Union

import dataclasses
from jax import core as jax_core
from jax import linear_util as lu
from jax import tree_util
from jax import util as jax_util

__all__ = [
    'Cell',
    'Equation',
    'Environment',
    'propagate'
]


VarOrLiteral = Union[jax_core.Var, jax_core.Literal]
safe_map = jax_core.safe_map


class Pytree(metaclass=abc.ABCMeta):
  """Class that registers objects as Jax pytree_nodes."""

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    tree_util.register_pytree_node(
        cls,
        cls.flatten,
        cls.unflatten
    )

  @abc.abstractmethod
  def flatten(self):
    pass

  @abc.abstractclassmethod
  def unflatten(cls, data, xs):
    pass


class Cell(Pytree):
  """Base interface for objects used during propagation.
  A Cell represents a member of a lattice, defined by the `top`, `bottom`
  and `join` methods. Conceptually, a "top" cell represents complete information
  about a value and a "bottom" cell represents no information about a value.
  Cells that are neither top nor bottom thus have partial information.
  The `join` method is used to combine two cells to create a cell no less than
  the two input cells. During the propagation, we hope to join cells until
  all cells are "top".
  Transformations that use propagate need to pass in objects that are Cell-like.
  A Cell needs to specify how to create a new default cell from a literal value,
  using the `new` class method. A Cell also needs to indicate if it is a known
  value with the `is_unknown` method, but by default, Cells are known.
  """

  def __init__(self, aval):
    self.aval = aval

  def __lt__(self, other: Any) -> bool:
    raise NotImplementedError

  def top(self) -> bool:
    raise NotImplementedError

  def bottom(self) -> bool:
    raise NotImplementedError

  def join(self, other: 'Cell') -> 'Cell':
    raise NotImplementedError

  @property
  def shape(self) -> Tuple[int]:
    return self.aval.shape

  @property
  def ndim(self) -> int:
    return len(self.shape)

  def is_unknown(self):
    # Convenient alias
    return self.bottom()

  @classmethod
  def new(cls, value):
    """Creates a new instance of a Cell from a value."""
    raise NotImplementedError

  @classmethod
  def unknown(cls, aval):
    """Creates an unknown Cell from an abstract value."""
    raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class Equation:
  """Hashable wrapper for jax_core.Jaxprs."""
  invars: Tuple[jax_core.Var]
  outvars: Tuple[jax_core.Var]
  primitive: jax_core.Primitive
  params_flat: Tuple[Any]
  params_tree: Any

  @classmethod
  def from_jaxpr_eqn(cls, eqn):
    params_flat, params_tree = tree_util.tree_flatten(eqn.params)
    return Equation(tuple(eqn.invars), tuple(eqn.outvars), eqn.primitive,
                    tuple(params_flat), params_tree)

  @property
  def params(self):
    return tree_util.tree_unflatten(self.params_tree, self.params_flat)

  def __hash__(self):
    # Override __hash__ to use Literal object IDs because Literals are not
    # natively hashable
    hashable_invars = tuple(id(invar) if isinstance(invar, jax_core.Literal)
                            else invar for invar in self.invars)
    return hash((hashable_invars, self.outvars, self.primitive,
                 self.params_tree))

  def __str__(self):
    return '{outvars} = {primitive} {invars}'.format(
        invars=' '.join(map(str, self.invars)),
        outvars=' '.join(map(str, self.outvars)),
        primitive=self.primitive,
    )


class Environment(Pytree):
  """Keeps track of variables and their values during propagation."""

  def __init__(self, cell_type, jaxpr):
    self.cell_type = cell_type
    self.env: Dict[jax_core.Var, Cell] = {}
    self.subenvs: Dict[Equation, 'Environment'] = {}
    self.jaxpr: jax_core.Jaxpr = jaxpr

  def copy(self) -> 'Environment':
    env = Environment(self.cell_type, self.jaxpr)
    env.env = self.env.copy()
    env.subenvs = {k: subenv.copy() for k, subenv in self.subenvs.items()}
    return env

  def join(self, other: 'Environment') -> 'Environment':
    env = Environment(self.cell_type, self.jaxpr)
    for var, val in self.env.items():
      env.env[var] = val.join(other.env[var])
    for eqn, subenv in self.subenvs.items():
      env.subenvs[eqn] = subenv.join(other.subenvs[eqn])
    return env

  def assert_same_type(self, other_env) -> None:
    """Raises an error if environments do not have matching Jaxprs."""
    error = ValueError('Cannot compare environments of different types.')
    if self.cell_type != other_env.cell_type:
      raise error
    elif self.jaxpr != other_env.jaxpr:
      raise error
    elif self.env.keys() != other_env.env.keys():
      raise error

  def read(self, var: VarOrLiteral) -> Cell:
    if isinstance(var, jax_core.Literal):
      return self.cell_type.new(var.val)
    else:
      return self.env.get(var, self.cell_type.unknown(var.aval))

  def write(self, var: VarOrLiteral, cell: Cell) -> Cell:
    if isinstance(var, jax_core.Literal):
      return cell
    cur_cell = self.read(var)
    if var is jax_core.dropvar:
      return cur_cell
    self.env[var] = cur_cell.join(cell)
    return self.env[var]

  def __getitem__(self, var: VarOrLiteral) -> Cell:
    return self.read(var)

  def __setitem__(self, key, val):
    raise ValueError('Environments do not support __setitem__. Please use the '
                     '`write` method instead.')

  def __contains__(self, var: VarOrLiteral):
    if isinstance(var, jax_core.Literal):
      return True
    return var in self.env

  def write_subenv(self, eqn: Equation, subenv: 'Environment') -> None:
    if eqn not in self.subenvs:
      self.subenvs[eqn] = subenv
    else:
      self.subenvs[eqn] = self.subenvs[eqn].join(subenv)

  def flatten(self):
    env_keys, env_values = jax_util.unzip2(self.env.items())
    subenv_keys, subenv_values = jax_util.unzip2(self.subenvs.items())
    return (env_values, subenv_values), (env_keys, subenv_keys, self.cell_type,
                                         self.jaxpr)

  @classmethod
  def unflatten(cls, data, xs):
    env_keys, subenv_keys, cell_type, jaxpr = data
    env_values, subenv_values = xs
    env = Environment(cell_type, jaxpr)
    env.env = dict(zip(env_keys, env_values))
    env.subenvs = dict(zip(subenv_keys, subenv_values))
    return env


def construct_graph_representation(eqns):
  """Constructs a graph representation of a Jaxpr."""
  neighbors = collections.defaultdict(set)
  for eqn in eqns:
    for var in it.chain(eqn.invars, eqn.outvars):
      if isinstance(var, jax_core.Literal):
        continue
      neighbors[var].add(eqn)

  def get_neighbors(var):
    if isinstance(var, jax_core.Literal):
      return set()
    return neighbors[var]
  return get_neighbors


def update_queue_state(queue, cur_eqn, get_neighbor_eqns,
                       incells, outcells, new_incells, new_outcells):
  """Updates the queue from the result of a propagation."""
  all_vars = cur_eqn.invars + cur_eqn.outvars
  old_cells = incells + outcells
  new_cells = new_incells + new_outcells

  for var, old_cell, new_cell in zip(all_vars, old_cells, new_cells):
    # If old_cell is less than new_cell, we know the propagation has made
    # progress.
    if old_cell < new_cell:
      # Extend left as a heuristic because in graphs corresponding to
      # chains of unary functions, we immediately want to pop off these
      # neighbors in the next iteration
      neighbors = get_neighbor_eqns(var) - set(queue) - {cur_eqn}
      queue.extendleft(neighbors)


PropagationRule = Callable[
    ...,
    Tuple[List[Union[Cell]], List[Cell], Optional[Environment]],
]


def propagate(cell_type: Type[Cell],
              rules: Dict[jax_core.Primitive, PropagationRule],
              jaxpr: jax_core.Jaxpr,
              constcells: List[Cell],
              incells: List[Cell],
              outcells: List[Cell]) -> Environment:
  """Propagates cells in a Jaxpr using a set of rules.
  Args:
    cell_type: used to instantiate literals into cells
    rules: maps JAX primitives to propagation rule functions
    jaxpr: used to construct the propagation graph
    constcells: used to populate the Jaxpr's constvars
    incells: used to populate the Jaxpr's invars
    outcells: used to populate the Jaxpr's outcells
  Returns:
    The Jaxpr environment after propagation has terminated
  """
  env = Environment(cell_type, jaxpr)
  safe_map(env.write, jaxpr.constvars, constcells)
  safe_map(env.write, jaxpr.invars, incells)
  safe_map(env.write, jaxpr.outvars, outcells)

  eqns = safe_map(Equation.from_jaxpr_eqn, jaxpr.eqns)

  get_neighbor_eqns = construct_graph_representation(eqns)
  # Initialize propagation queue with equations neighboring constvars, invars,
  # and outvars.
  out_eqns = set()
  for jaxpr_eqn in jaxpr.eqns:
    for var in it.chain(jaxpr_eqn.invars, jaxpr_eqn.outvars):
      env.write(var, cell_type.unknown(var.aval))

  for var in it.chain(jaxpr.outvars, jaxpr.invars, jaxpr.constvars):
    out_eqns.update(get_neighbor_eqns(var))
  queue = Deque[Equation](out_eqns)
  while queue:
    eqn = queue.popleft()

    incells = safe_map(env.read, eqn.invars)
    outcells = safe_map(env.read, eqn.outvars)

    rule = rules[eqn.primitive]
    call_jaxpr, params = jax_core.extract_call_jaxpr(eqn.primitive, eqn.params)
    subfuns = []
    if call_jaxpr:
      subfuns.append(lu.wrap_init(functools.partial(propagate, cell_type, rules,
          call_jaxpr, ())))

    new_incells, new_outcells, subenv = rule(
        list(it.chain(subfuns, incells)), outcells, **params)
    if subenv:
      env.write_subenv(eqn, subenv)

    new_incells = safe_map(env.write, eqn.invars, new_incells)
    new_outcells = safe_map(env.write, eqn.outvars, new_outcells)

    update_queue_state(queue, eqn, get_neighbor_eqns, incells, outcells,
                       new_incells, new_outcells)
  return env
