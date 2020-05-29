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

from functools import partial
import itertools
from typing import Any, Dict, Iterable, Iterator, List, Tuple, TypeVar

import numpy as np

from .. import core
from .. import linear_util as lu
from ..util import prod, safe_map as map, safe_zip as zip
from ..tree_util import tree_structure, tree_flatten, tree_unflatten


TRIVIAL_TREEDEF = tree_structure(1)

TreeDef = Any
Array = Any
PyTree = Any


@lu.transformation
def undo_tree_fun(trees):
  with core.new_master(TreeTrace) as master:
    out_trees = yield (master, trees), {}
    del master
  yield out_trees


@lu.transformation
def undo_tree_subtrace(master, trees):
  trace = TreeTrace(master, core.cur_sublevel())
  in_tracers = map(partial(convert_vectorized_tree, trace), trees)
  ans = yield in_tracers, {}
  out_tracers = map(trace.full_raise, ans)
  out_trees = tuple(restore_tree(t.treedefs, t.leaves) for t in out_tracers)
  yield out_trees


def is_trivial_axis(
    treedef: TreeDef, leafshape: List[Tuple[int, ...]],
) -> bool:
  return (treedef is TRIVIAL_TREEDEF
          and len(leafshapes) == 1
          and len(leafshapes[0]) == 1)


def _iter_leaf_coords(treedefs: List[TreeDef]) -> Iterator[Tuple[int, ...]]:
  return itertools.product(*[range(treedef.num_leaves) for treedef in treedefs])


T = TypeVar("T")

def _concat_tuples(tuples: Iterable[Tuple[T, ...]]) -> Tuple[T, ...]:
  return tuple(itertools.chain.from_iterable(tuples))


def _leafshape(
    leafshapes: List[List[Tuple[int, ...]]],
    coords: Tuple[int, ...],
) -> Tuple[int, ...]:
  return _concat_tuples([leafshapes[i][j] for i, j in enumerate(coords)])


class TreeTracer(core.Tracer):
  __slots__ = ["treedefs", "leafshapes", "leaves"]

  treedefs: List[TreeDef]
  leafshapes: List[List[Tuple[int, ...]]]
  leaves: Dict[Tuple[int, ...], Array]

  def __init__(self, trace, treedefs, leafshapes, leaves):
    assert len(treedefs) == len(leafshapes)
    assert leaves
    for coords in _iter_leaf_coords(treedefs):
      expected_shape = _leafshape(leafshapes, coords)
      actual_shape = leaves[coords].shape
      assert actual_shape == expected_shape, (actual_shape, expected_shape)
    self._trace = trace
    self.treedefs = treedefs
    self.leafshapes = leafshapes
    self.leaves = leaves

  @property
  def aval(self):
    shape = tuple(sum(map(prod, shapes)) for shapes in self.leafshapes)
    dtype = next(iter(self.leaves.values())).dtype
    return core.ShapedArray(shape, dtype)

  def full_lower(self):
    if all(map(is_trivial_axis, self.treedefs, self.leafshapes)):
      value, = self.leaves.values()
      return core.full_lower(value)
    else:
      return self


class TreeTrace(core.Trace):

  def pure(self, val):
    # constant array/scalar, no tracers
    return convert_leaf_array(self, val)

  def lift(self, tracer):
    # called for tracers of a lower priority
    return convert_leaf_array(self, tracer)

  def sublift(self, tracer):
    # specifically called for transformations of functions that involve
    # jit/pmap via lexical closure -- called for tracers of your trace type
    return TreeTracer(self, tracer.treedefs, tracer.leafshapes, tracer.leaves)

  def process_primitive(self, primitive, tracers, params):
    assert not primitive.multiple_results  # TODO
    rule = tree_rules[primitive]
    treedefs, leafshapes, leaves = rule(tracers, **params)
    return TreeTracer(self, treedefs, leafshapes, leaves)


def convert_vectorized_tree(trace: TreeTrace, tree: PyTree) -> TreeTracer:
  import jax.numpy as jnp
  xs, treedef = tree_flatten(tree)
  leafshape = [np.shape(x) for x in xs]
  dtype = jnp.result_type(*xs)
  leaves = {(i,): jnp.asarray(leaf, dtype) for i, leaf in enumerate(xs)}
  return TreeTracer(trace, [treedef], [leafshape], leaves)


def convert_leaf_array(trace: TreeTrace, leaf) -> TreeTracer:
  import jax.numpy as jnp
  treedef = tree_structure(leaf)
  if treedef != TRIVIAL_TREEDEF:
    raise ValueError(
        f"argument to from_array must be a leaf already, got {treedef}")
  leaf = jnp.asarray(leaf)
  shape = [[(s,)] for s in leaf.shape]
  leaves = {(0,) * leaf.ndim: leaf}
  return TreeTracer(trace, [TRIVIAL_TREEDEF] * leaf.ndim, shape, leaves)


def restore_tree(
    treedefs: List[TreeDef], leaves: Dict[Tuple[int, ...], Array]) -> PyTree:
  while treedefs:
    flattened_leaves = {}
    for coords in _iter_leaf_coords(treedefs[:-1]):
      leaf_list = [leaves[coords + (i,)] for i in range(treedefs[-1].num_leaves)]
      flattened_leaves[coords] = tree_unflatten(treedefs[-1], leaf_list)
    treedefs = treedefs[:-1]
    leaves = flattened_leaves
  return leaves[()]


### rule definitions

tree_rules = {}

def defvectorized(prim):
  tree_rules[prim] = partial(vectorized_tree_rule, prim)

def vectorized_tree_rule(prim, args, **params):
  arg, = args
  leaves = {coords: prim.bind(arg.leaves[coords], **params)
            for coords in _iter_leaf_coords(arg.treedefs)}
  return arg.treedefs, arg.leafshapes, leaves


def defnaryop(prim):
  tree_rules[prim] = partial(naryop_tree_rule, prim)

def naryop_tree_rule(prim, args, **params):
  # TODO: needs broadcasting
  ndim, = {t.ndim for t in args}

  out_treedefs = []
  out_leafshapes = []

  for axis in range(ndim):
    # check treedefs
    non_trivial_treedefs = {arg.treedefs[axis] for arg in args
                            if arg.treedefs[axis] != TRIVIAL_TREEDEF}
    if len(non_trivial_treedefs) > 1:
      raise ValueError(
          f"conflicting treedefs along axis={axis}: {non_trivial_treedefs}"
      )
    elif len(non_trivial_treedefs) == 1:
      treedef, = non_trivial_treedefs
      out_treedefs.append(treedef)
    else:
      out_treedefs.append(TRIVIAL_TREEDEF)

    # check shapes
    non_trivial_shapes = {tuple(arg.leafshapes[axis]) for arg in args
                          if arg.leafshapes[axis] != [(arg.shape[axis],)]}
    if len(non_trivial_shapes) > 1:
      raise ValueError(
          f"conflicting shapes along axis={axis}: {non_trivial_shapes}"
      )
    elif len(non_trivial_shapes) == 1:
      leafshapes, = non_trivial_shapes
      out_leafshapes.append(leafshapes)
    else:
      axis_size = max(arg.shape[axis] for arg in args)
      out_leafshapes.append([(axis_size,)])

  out_leaves = {}
  for coords in _iter_leaf_coords(out_treedefs):
    leaves = []
    for arg in args:
      # TODO(shoyer): use lax.expand_dims for inserting dimensions rather
      # than reshape!
      in_coords = tuple(k if arg.shape[i] != 1 else 0
                        for i, k in enumerate(coords))
      leaf = arg.leaves[in_coords]
      shape_pieces = [
          arg.leafshapes[i][j]
          if arg.shape[i] != 1
          else (1,) * len(out_leafshapes[i][j])  # broadcasting
          for i, j in enumerate(coords)
      ]
      shape = _concat_tuples(shape_pieces)
      leaves.append(leaf.reshape(shape))

    out_leaves[coords] = prim.bind(*leaves, **params)

  return out_treedefs, out_leafshapes, out_leaves


def broadcast_in_dim_tree_rule(prim, arg, *, shape, broadcast_dimensions):
  treedefs = [flattree.TRIVIAL_TREEDEF] * len(shape)
  leafshapes = [[(s,)] for s in shape]

  for input_dim, output_dim in enumerate(broadcast_dimensions):
    treedef = treedefs[output_dim] = arg.treedefs[input_dim]
    leafshape = arg.leafshapes[input_dim]
    if not flattree.is_trivial_axis(treedef, leafshape):
      if shape[input_dim] != shape[output_dim]:
        raise ValueError(f"cannot resize dimension {input_dim} because it "
                         f"corresponds to a non-trivial pytree: {treedef}")
      leafshapes[output_dim] = leafshape

  new_dims = sorted(set(range(len(shape))) - set(broadcast_dimensions))

  out_leaves = {}
  for in_coords in _iter_leaf_coords(treedefs):
    leaf = arg.leaves[in_coords]

    out_coords = list(coords)
    for i in new_dims:
      out_coords.insert(i, 0)

    leaf_shape = ...
    leaf_bdims = ...

    out_leaves[out_coords] = prim.bind(arg.leaves, leaf_shape, leaf_bdims)
