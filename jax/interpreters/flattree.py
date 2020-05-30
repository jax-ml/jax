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
from ..util import prod, safe_map as map, unzip3
from ..tree_util import tree_structure, tree_flatten, tree_unflatten


TRIVIAL_TREEDEF = tree_structure(1)

TreeDef = Any
ArrayLike = Any
PyTree = Any


@lu.transformation
def tree_fun(trees):
  with core.new_master(TreeTrace) as master:
    out_trees = yield (master, trees), {}
    del master
  yield out_trees

@lu.transformation
def tree_subtrace(master, trees):
  trace = TreeTrace(master, core.cur_sublevel())
  in_tracers = [TreeTracer(trace, *convert_vectorized_tree(t)) for t in trees]
  ans = yield in_tracers, {}
  out_tracers = map(trace.full_raise, ans)
  out_trees = tuple(restore_tree(t.treedefs, t.leaves) for t in out_tracers)
  yield out_trees

@lu.transformation_with_aux
def traceable(in_tree_def, *states):
  state_in = tree_unflatten(in_tree_def, states)
  state_out = yield state_in, {}
  out_flat, out_tree_def = tree_flatten(state_out)
  yield out_flat, out_tree_def


def is_trivial_axis(
    treedef: TreeDef, leafshapes: List[Tuple[int, ...]],
) -> bool:
  return treedef is TRIVIAL_TREEDEF and len(leafshapes) == 1 and len(leafshapes[0]) == 1


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


def _axis_length(shapes: Iterable[Tuple[int, ...]]) -> int:
  return sum(map(prod, shapes))


class TreeTracer(core.Tracer):
  __slots__ = ["treedefs", "leafshapes", "leaves"]

  treedefs: List[TreeDef]
  leafshapes: List[List[Tuple[int, ...]]]
  leaves: Dict[Tuple[int, ...], ArrayLike]

  def __init__(self, trace, treedefs, leafshapes, leaves):
    assert len(treedefs) == len(leafshapes)
    assert leaves
    for coords in _iter_leaf_coords(treedefs):
      expected_shape = _leafshape(leafshapes, coords)
      actual_shape = leaves[coords].shape
      assert actual_shape == expected_shape, (coords, actual_shape, expected_shape)
    self._trace = trace
    self.treedefs = treedefs
    self.leafshapes = leafshapes
    self.leaves = leaves

  @property
  def aval(self):
    shape = tuple(map(_axis_length, self.leafshapes))
    dtype = core.concrete_aval(next(iter(self.leaves.values()))).dtype
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
    return TreeTracer(self, *convert_leaf_array(val))

  def lift(self, tracer):
    # called for tracers of a lower priority
    return TreeTracer(self, *convert_leaf_array(val))

  def sublift(self, tracer):
    # specifically called for transformations of functions that involve
    # jit/pmap via lexical closure -- called for tracers of your trace type
    return TreeTracer(self, tracer.treedefs, tracer.leafshapes, tracer.leaves)

  def process_primitive(self, primitive, tracers, params):
    assert not primitive.multiple_results  # TODO
    rule = tree_rules[primitive]
    treedefs_in, leafshapes_in, leaves_in = unzip3(
        (t.treedefs, t.leafshapes, t.leaves) for t in tracers)
    treedefs, leafshapes, leaves = rule(
        treedefs_in, leafshapes_in, leaves_in, **params)
    return TreeTracer(self, treedefs, leafshapes, leaves)

  def process_call(self, call_primitive, f, tracers, params):
    treedefs_in, leafshapes_in, leaves_in = unzip3(
        (t.treedefs, t.leafshapes, t.leaves) for t in tracers)
    state_in, in_tree_def = tree_flatten((treedefs_in, leafshapes_in, leaves_in))
    f_tree, out_tree_def = traceable(tree_subtrace(f, self.master), in_tree_def)
    # This breaks, because state_in contains a treedef which isn't a value JAX
    # value.
    result = call_primitive.bind(f_tree, *state_in, **params)
    state_out = tree_unflatten(out_tree_def(), result)
    return [TreeTracer(self, *args) for args in zip(state_out)]


TreeState = Tuple[
    List[TreeDef],
    List[List[Tuple[int, ...]]],
    Dict[Tuple[int, ...], ArrayLike],
]


def convert_vectorized_tree(tree: PyTree) -> TreeState:
  import jax.numpy as jnp
  xs, treedef = tree_flatten(tree)
  leafshape = [np.shape(x) for x in xs]
  dtype = jnp.result_type(*xs)
  leaves = {(i,): jnp.asarray(leaf, dtype) for i, leaf in enumerate(xs)}
  return [treedef], [leafshape], leaves


def convert_leaf_array(leaf: ArrayLike) -> TreeState:
  import jax.numpy as jnp
  treedef = tree_structure(leaf)
  if treedef != TRIVIAL_TREEDEF:
    raise ValueError(
        f"argument to from_array must be a leaf already, got {treedef}")
  ndim = np.ndim(leaf)
  treedefs = [TRIVIAL_TREEDEF] * ndim
  leafshapes = [[(s,)] for s in np.shape(leaf)]
  leaves = {(0,) * ndim: leaf}
  return treedefs, leafshapes, leaves


def restore_tree(
    treedefs: List[TreeDef], leaves: Dict[Tuple[int, ...], ArrayLike]) -> PyTree:
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

def vectorized_tree_rule(prim, treedefs_in, leafshapes_in, leaves_in, **params):
  treedefs, = treedefs_in
  leafshapes, = leafshapes_in
  leaves, = leaves_in
  out_leaves = {coords: prim.bind(leaves[coords], **params)
                for coords in _iter_leaf_coords(treedefs)}
  return treedefs, leafshapes, out_leaves


def defnaryop(prim):
  tree_rules[prim] = partial(naryop_tree_rule, prim)


def _filter_scalar_leaves(treedefs_in, leafshapes_in, leaves_in):
  treedefs_out = []
  leafshapes_out = []
  leaves_out = []
  scalars = []
  for i, (treedefs, leafshapes, leaves) in enumerate(
      zip(treedefs_in, leafshapes_in, leaves_in)):
    if treedefs:
      treedefs_out.append(treedefs)
      leafshapes_out.append(leafshapes)
      leaves_out.append(leaves)
    else:
      scalars.append((i, leaves[()]))
  return treedefs_out, leafshapes_out, leaves_out, scalars


def _is_broadcasting_axis(shapes: List[Tuple[int, ...]]) -> bool:
  return _axis_length(shapes) == 1


def naryop_tree_rule(prim, treedefs_in, leafshapes_in, leaves_in, **params):
  from jax import lax

  treedefs_in, leafshapes_in, leaves_in, scalars = _filter_scalar_leaves(
      treedefs_in, leafshapes_in, leaves_in)

  if not treedefs_in:
    args = [scalar for _, scalar in scalars]
    return [], [], {(): prim.bind(*args, **params)}

  ndim, = {len(treedefs) for treedefs in treedefs_in}

  out_treedefs = []
  out_leafshapes = []

  for axis in range(ndim):
    # check treedefs
    non_trivial_treedefs = {treedefs[axis] for treedefs in treedefs_in
                            if treedefs[axis] != TRIVIAL_TREEDEF}
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
    non_trivial_shapes = {tuple(leafshapes[axis]) for leafshapes in leafshapes_in
                          if leafshapes[axis] != [(1,)]}
    using_broadcasting = any(
        leafshapes[axis] == [(1,)] for leafshapes in leafshapes_in)
    if len(non_trivial_shapes) > 1:
      raise ValueError(
          f"conflicting shapes along axis={axis}: {non_trivial_shapes}"
      )
    elif len(non_trivial_shapes) == 1:
      shapes, = non_trivial_shapes
      shapes = [(1,) if shape == () and using_broadcasting else shape for shape in shapes]
      out_leafshapes.append(shapes)
    else:
      # axis_size = max(len(treedefs) for treedefs in treedefs_in)
      out_leafshapes.append([(1,)])
  # print(f"out_leafshapes: {out_leafshapes}")

  out_leaves = {}
  for coords in _iter_leaf_coords(out_treedefs):
    args = []

    rank_per_axis = []
    for axis, coord in enumerate(coords):
      # print(f"axis={axis}")
      rank = 0
      # found_broadcasting_dim = False
      for leafshapes in leafshapes_in:
        # print(f"leafshape: {leafshapes[axis]}")
        if len(leafshapes[axis]) == 1:
          shape, = leafshapes[axis]
        else:
          shape = leafshapes[axis][coord]
        rank = max(rank, len(shape))
      rank_per_axis.append(rank)

    # print(f'rank_per_axis: {rank_per_axis}')

    for leafshapes, leaves in zip(leafshapes_in, leaves_in):
      in_coords = tuple(coord if len(leafshapes[axis]) != 1 else 0
                        for axis, coord in enumerate(coords))
      leaf = leaves[in_coords]

      insert_dims = []
      dim = 0
      for axis, coord in enumerate(coords):
        if len(leafshapes[axis]) == 1:
          shape, = leafshapes[axis]  # broadcasting
        else:
          shape = leafshapes[axis][coord]
        insert_dims.extend(
            range(dim, dim + rank_per_axis[axis] - len(shape)))
        dim += rank_per_axis[axis]

      # print(f'at {coords}, expanding {tuple(insert_dims)}')
      args.append(lax.expand_dims(leaf, tuple(insert_dims)))

    for i, scalar in scalars:
      args.insert(i, scalar)

    out_leaves[coords] = prim.bind(*args, **params)

  return out_treedefs, out_leafshapes, out_leaves


def broadcast_in_dim_tree_rule(prim, treedefs_in, leafshapes_in, leaves_in,
                               *, shape, broadcast_dimensions):
  treedefs, = treedefs_in
  leafshapes, = leafshapes_in
  leaves, = leaves_in

  out_treedefs = [TRIVIAL_TREEDEF] * len(shape)
  out_leafshapes = [[(s,)] for s in shape]

  for input_dim, output_dim in enumerate(broadcast_dimensions):
    treedef = out_treedefs[output_dim] = treedefs[input_dim]
    leafshape = leafshapes[input_dim]
    if not is_trivial_axis(treedef, leafshape):
      if _axis_length(leafshape) != shape[output_dim]:
        raise ValueError(f"cannot resize dimension {input_dim} because it "
                         f"corresponds to a non-trivial pytree: {treedef}")
      out_leafshapes[output_dim] = leafshape

  out_leaves = {}
  for in_coords, out_coords in zip(
      _iter_leaf_coords(treedefs), _iter_leaf_coords(out_treedefs)):

    leaf = leaves[in_coords]

    out_shape = _concat_tuples(
        [out_leafshapes[i][j] for i, j in enumerate(out_coords)])

    out_bdims = []
    bdim_delta = 0
    for shapes, coord, bdim in zip(
        leafshapes, in_coords, broadcast_dimensions,
    ):
      leaf_ndim = len(shapes[coord])
      out_bdims.extend(range(bdim + bdim_delta, bdim + bdim_delta + leaf_ndim))
      bdim_delta += leaf_ndim - 1

    out_leaves[out_coords] = prim.bind(
        leaf, shape=out_shape, broadcast_dimensions=tuple(out_bdims))

  return out_treedefs, out_leafshapes, out_leaves


def defreducer(prim, binop_prim):
  tree_rules[prim] = partial(reducer_tree_rule, prim, binop_prim.bind)

def reducer_tree_rule(prim, binop, treedefs_in, leafshapes_in, leaves_in,
                      *, axes, **params):
  treedefs, = treedefs_in
  leafshapes, = leafshapes_in
  leaves, = leaves_in

  out_treedefs = [t for i, t in enumerate(treedefs) if i not in axes]
  out_leafshapes = [s for i, s in enumerate(leafshapes) if i not in axes]

  out_leaves = {coords: None for coords in _iter_leaf_coords(out_treedefs)}

  for in_coords in _iter_leaf_coords(treedefs):
    out_coords = tuple(c for i, c in enumerate(in_coords) if i not in axes)

    leaf_axes = []
    num_axes = 0
    for axis, coord in enumerate(in_coords):
      leaf_ndim = len(leafshapes[axis][coord])
      if axis in axes:
        leaf_axes.extend(range(num_axes, num_axes + leaf_ndim))
      num_axes += leaf_ndim

    reduced_leaf = prim.bind(leaves[in_coords], axes=tuple(leaf_axes), **params)
    out_leaves[out_coords] = (
        reduced_leaf
        if out_leaves[out_coords] is None
        else binop(reduced_leaf, out_leaves[out_coords])
    )
  return out_treedefs, out_leafshapes, out_leaves
