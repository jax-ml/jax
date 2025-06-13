# Copyright 2019 The JAX Authors.
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
Parallelization primitives.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import itertools
import math

import jax
from jax import tree_util
from jax._src import core
from jax._src import config
from jax._src import dispatch
from jax._src import dtypes
from jax._src.sharding_impls import (SPMDAxisContext, ShardingContext,
                                     NamedSharding, PartitionSpec as P)
from jax._src.core import AxisName, ShapedArray
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.mesh import get_abstract_mesh
from jax._src.core import pvary
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.util import (canonicalize_axis, moveaxis, safe_map, safe_zip,
                           unzip2)
import jax.numpy as jnp
import numpy as np

unsafe_map, map = map, safe_map  # type: ignore
unsafe_zip, zip = zip, safe_zip  # type: ignore


### parallel traceables

def psum(x, axis_name, *, axis_index_groups=None):
  """Compute an all-reduce sum on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Inputs of boolean dtype are converted to integers before the reduction.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would perform psums over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce sum along the axis ``axis_name``.

  Examples:
    For example, with 4 XLA devices available:

    >>> x = np.arange(4)
    >>> y = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(x)
    >>> print(y)
    [6 6 6 6]
    >>> y = jax.pmap(lambda x: x / jax.lax.psum(x, 'i'), axis_name='i')(x)
    >>> print(y)
    [0.         0.16666667 0.33333334 0.5       ]

    Suppose we want to perform ``psum`` among two groups, one with ``device0`` and ``device1``, the other with ``device2`` and ``device3``,

    >>> y = jax.pmap(lambda x: jax.lax.psum(x, 'i', axis_index_groups=[[0, 1], [2, 3]]), axis_name='i')(x)
    >>> print(y)
    [1 1 5 5]

    An example using 2D-shaped x. Each row is data from one device.

    >>> x = np.arange(16).reshape(4, 4)
    >>> print(x)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]

    Full ``psum`` across all devices:

    >>> y = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(x)
    >>> print(y)
    [[24 28 32 36]
     [24 28 32 36]
     [24 28 32 36]
     [24 28 32 36]]

    Perform ``psum`` among two groups:

    >>> y = jax.pmap(lambda x: jax.lax.psum(x, 'i', axis_index_groups=[[0, 1], [2, 3]]), axis_name='i')(x)
    >>> print(y)
    [[ 4  6  8 10]
     [ 4  6  8 10]
     [20 22 24 26]
     [20 22 24 26]]
  """
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  if not axis_name:
    return x
  if any(isinstance(axis, int) for axis in axis_name) and axis_index_groups is not None:
    raise ValueError("axis_index_groups only supported for sums over just named axes")
  _validate_reduce_axis_index_groups(axis_index_groups)
  leaves, treedef = tree_util.tree_flatten(x)
  leaves = [lax.convert_element_type(l, np.int32)
            if dtypes.dtype(l) == np.bool_ else l for l in leaves]
  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  # handle the constant case specially
  if all(not isinstance(leaf, core.Tracer) for leaf in leaves):
    named_axes, pos_axes = axes_partition = [], []
    for axis in axis_name:
      axes_partition[isinstance(axis, int)].append(axis)
    def pos_reduce(x):
      if not pos_axes:
        return x
      return lax.reduce_sum(x, [canonicalize_axis(axis, getattr(x, 'ndim', 0))
                                for axis in pos_axes])
    if axis_index_groups is not None:
      assert not pos_axes
      size = len(axis_index_groups[0])
    else:
      size = math.prod([core.get_axis_env().axis_size(name) for name in named_axes])
    out_flat = tuple(lax._const(leaf, size) * pos_reduce(leaf) for leaf in leaves)
  else:
    if config._check_vma.value:
      out_flat = bind_psum_invariant(
          leaves, axes=tuple(axis_name), axis_index_groups=axis_index_groups)
    else:
      out_flat = psum_p.bind(
          *leaves, axes=tuple(axis_name), axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, out_flat)

def bind_psum_invariant(leaves, *, axes, axis_index_groups):
  if axis_index_groups is not None:
    raise NotImplementedError
  axes_ = frozenset(axes)
  args_ = []
  for x in leaves:
    in_vma = core.get_aval(x).vma
    args_.append(pvary(x, tuple(pbroadcast_names))
                 if (pbroadcast_names := axes_ - in_vma) else x)
  return psum_invariant_p.bind(*args_, axes=axes,
                               axis_index_groups=axis_index_groups)


def pmean(x, axis_name, *, axis_index_groups=None):
  """Compute an all-reduce mean on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would perform pmeans over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and on TPUs all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce mean along the axis ``axis_name``.

  For example, with 4 XLA devices available:

  >>> x = np.arange(4)
  >>> y = jax.pmap(lambda x: jax.lax.pmean(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [1.5 1.5 1.5 1.5]
  >>> y = jax.pmap(lambda x: x / jax.lax.pmean(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [0.        0.6666667 1.3333334 2.       ]
  """
  x = psum(x, axis_name=axis_name, axis_index_groups=axis_index_groups)
  n = _axis_size(axis_name, axis_index_groups)
  return tree_util.tree_map(lambda v: v / n, x)

def pmax(x, axis_name, *, axis_index_groups=None):
  """Compute an all-reduce max on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would perform pmaxes over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and on TPUs all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce max along the axis ``axis_name``.
  """
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  if any(isinstance(axis, int) for axis in axis_name) and axis_index_groups is not None:
    raise ValueError("axis_index_groups only supported for sums over just named axes")
  _validate_reduce_axis_index_groups(axis_index_groups)
  leaves, treedef = tree_util.tree_flatten(x)
  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  leaves = map(partial(insert_collective_pvary, axis_name), leaves)
  out_flat = pmax_p.bind(*leaves, axes=axis_name,
                         axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, out_flat)

def pmin(x, axis_name, *, axis_index_groups=None):
  """Compute an all-reduce min on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would perform pmins over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and on TPUs all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce min along the axis ``axis_name``.
  """
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  if any(isinstance(axis, int) for axis in axis_name) and axis_index_groups is not None:
    raise ValueError("axis_index_groups only supported for sums over just named axes")
  _validate_reduce_axis_index_groups(axis_index_groups)
  leaves, treedef = tree_util.tree_flatten(x)
  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  leaves = map(partial(insert_collective_pvary, axis_name), leaves)
  out_flat = pmin_p.bind(*leaves, axes=axis_name,
                         axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, out_flat)

# TODO(mattjj): add a pargmin_p, or add named axis support to lax.argmin_p
def pargmin(x, axis_name):
  if isinstance(axis_name, (tuple, list)):
    raise TypeError(f"pargmin only accepts a single axis, got {axis_name}")
  return _axis_index_of_val(x, pmin(x, axis_name), axis_name)

# TODO(mattjj): add a pargmax_p, or add named axis support to lax.argmax_p
def pargmax(x, axis_name):
  if isinstance(axis_name, (tuple, list)):
    raise TypeError(f"pargmin only accepts a single axis, got {axis_name}")
  return _axis_index_of_val(x, pmax(x, axis_name), axis_name)

def _axis_index_of_val(x, val, axis_name):
  idx = axis_index(axis_name)
  mask = (val == x)
  validx = lax.select(mask,
                      lax.full(mask.shape, idx),
                      lax.full(mask.shape, dtypes.iinfo(dtypes.dtype(idx)).max, dtypes.dtype(idx)))
  return pmin(validx, axis_name)

def _validate_reduce_axis_index_groups(axis_index_groups):
  if axis_index_groups is None:
    return
  axis_space = range(sum(len(group) for group in axis_index_groups))
  if {i for g in axis_index_groups for i in g} != set(axis_space):
    raise ValueError("axis_index_groups must cover all indices exactly once")

def _canonicalize_axis_index_groups(axis_index_groups):
  if axis_index_groups is None:
    return
  return tuple(map(tuple, axis_index_groups))


def pbroadcast(x, axis_name, source):
  """Perform a collective broadcast and replicate from ``source``.

  This is equivalent to
  ```
  def pbroadcast(x, axis_name, source):
    masked = jnp.where(axis_index(axis_name) == source, x, zeros_like(x))
    return psum(masked, axis_name)
  ```
  but implemented in a hardware optimized way.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  This function is an analog of the CollectiveBroadcast HLO.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    source: int, representing which index into ``axis_name`` that should be copied.

  Returns:
    Array(s) with ``x`` being copied from the ``source`` index slice of ``axis_name``.
  """
  return tree_util.tree_map(
      partial(pbroadcast_p.bind, axis_name=axis_name, source=source), x)


def ppermute(x, axis_name, perm):
  """Perform a collective permutation according to the permutation ``perm``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  This function is an analog of the CollectivePermute HLO.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    perm: list of pairs of ints, representing
      ``(source_index, destination_index)``
      pairs that encode how the mapped axis named ``axis_name`` should be
      shuffled. The integer values are treated as indices into the mapped axis
      ``axis_name``. Any two pairs should not have the same source index or the
      same destination index. For each index of the axis ``axis_name`` that does
      not correspond to a destination index in ``perm``, the corresponding
      values in the result are filled with zeros of the appropriate type.

  Returns:
    Array(s) with the same shape as ``x`` with slices along the axis
    ``axis_name`` gathered from ``x`` according to the permutation ``perm``.
  """
  if not isinstance(axis_name, (list, tuple)):
    axis_name = (axis_name,)
  def bind(leaf):
    leaf = insert_collective_pvary(axis_name, leaf)
    return ppermute_p.bind(leaf, axis_name=axis_name, perm=tuple(map(tuple, perm)))
  return tree_util.tree_map(bind, x)

def pshuffle(x, axis_name, perm):
  """Convenience wrapper of jax.lax.ppermute with alternate permutation encoding

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    perm: list of ints encoding sources for the permutation to be applied to
      the axis named ``axis_name``, so that the output at axis index i
      comes from the input at axis index perm[i]. Every integer in [0, N) should
      be included exactly once for axis size N.

  Returns:
    Array(s) with the same shape as ``x`` with slices along the axis
    ``axis_name`` gathered from ``x`` according to the permutation ``perm``.
  """
  if set(perm) != set(range(len(perm))):
    raise ValueError(f"`perm` does not represent a permutation: {perm}")
  return ppermute(x, axis_name, list(zip(perm, range(len(perm)))))


def pswapaxes(x, axis_name, axis, *, axis_index_groups=None):
  """Swap the pmapped axis ``axis_name`` with the unmapped axis ``axis``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  The group size of the mapped axis size must be equal to the size of the
  unmapped axis; that is, we must have
  ``lax.psum(1, axis_name, axis_index_groups=axis_index_groups) == x.shape[axis]``.
  By default, when ``axis_index_groups=None``, this encompasses all the devices.

  This function is a special case of ``all_to_all`` where the pmapped axis of
  the input is placed at the position ``axis`` in the output. That is, it is
  equivalent to ``all_to_all(x, axis_name, axis, axis)``.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis: int indicating the unmapped axis of ``x`` to map with the name
      ``axis_name``.
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would run pswapaxes over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x``.
  """
  return all_to_all(x, axis_name, axis, axis, axis_index_groups=axis_index_groups)

def all_to_all(x, axis_name, split_axis, concat_axis, *, axis_index_groups=None, tiled=False):
  """Materialize the mapped axis and map a different axis.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  In the output, the input mapped axis ``axis_name`` is materialized at the
  logical axis position ``concat_axis``, and the input unmapped axis at position
  ``split_axis`` is mapped with the name ``axis_name``.

  The group size of the mapped axis size must be equal to the size of the
  unmapped axis; that is, we must have
  ``lax.psum(1, axis_name, axis_index_groups=axis_index_groups) == x.shape[axis]``.
  By default, when ``axis_index_groups=None``, this encompasses all the devices.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    split_axis: int indicating the unmapped axis of ``x`` to map with the name
      ``axis_name``.
    concat_axis: int indicating the position in the output to materialize the
      mapped axis of the input with the name ``axis_name``.
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would run all_to_all over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and all groups must be the same size.
    tiled: when True, all_to_all will divide split_axis into chunks and concatenate
      them along concat_axis. In particular, no dimensions are added or removed.
      False by default.

  Returns:
    When tiled is False, array(s) with shape given by the expression::

      np.insert(np.delete(x.shape, split_axis), concat_axis, axis_size)

    where ``axis_size`` is the size of the mapped axis named ``axis_name`` in
    the input ``x``.

    Otherwise array with shape similar to the input shape, except with split_axis
    divided by axis size and concat_axis multiplied by axis size.
  """
  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  def bind(x, split_axis=split_axis, concat_axis=concat_axis):
    group_size = _axis_size(axis_name, axis_index_groups)
    if tiled:
      if x.shape[split_axis] % group_size != 0:
        raise ValueError(f"The size of all_to_all split_axis ({x.shape[split_axis]}) "
                         f"has to be divisible by the size of the named axis "
                         f"{axis_name} ({group_size})")
    else:
      if group_size != x.shape[split_axis]:
        msg = ("all_to_all requires the size of the mapped axis axis_name to "
               "equal x.shape[split_axis], but they are {} and {} respectively.")
        raise ValueError(msg.format(group_size, x.shape[split_axis]))
      if split_axis < concat_axis:
        concat_axis += 1  # concat_axis gives a position _after_ split_axis is removed
        x = lax.expand_dims(x, (concat_axis,))  # insert the new axis
      elif split_axis == concat_axis:
        pass
      else:  # concat_axis < split_axis
        x = lax.expand_dims(x, (concat_axis,))  # insert the new axis
        split_axis += 1   # we have a new axis before split_axis now
    x = insert_collective_pvary(axis_name, x)
    result = all_to_all_p.bind(x, split_axis=split_axis, concat_axis=concat_axis,
                               axis_name=axis_name,
                               axis_index_groups=axis_index_groups,
                               tiled=tiled)
    if not tiled and split_axis != concat_axis:
      result = lax.squeeze(result, (split_axis,))
    return result

  return tree_util.tree_map(bind, x)

def ragged_all_to_all(
    operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *,
    axis_name, axis_index_groups = None):
  """Ragged version of :func:`all_to_all` collective.

  We say data are "ragged" when they can be represented as a list of arrays
  whose shapes differ only in the size of the leading axis. For example, these
  data are ragged, comprising four component arrays::

    ragged_data = [jnp.arange(3), jnp.arange(1), jnp.arange(4), jnp.arange(1)]

  We often instead want a contiguous representation, e.g. for batching. But
  because the shapes of the components differ, we can't apply ``jnp.stack`` to
  represent these data by a single rectangular array with the leading axis
  indexing the component arrays. So instead of stacking, we concatenate along
  the leading axis and keep track of offsets and sizes.

  That is, we can represent ragged data contiguously using a triple of dense
  arrays ``(data, offsets, sizes)``:
    * ``data``: the concatenated component arrays,
    * ``offsets``: 1D array of indices into the leading axis of ``data``
      indicating where the data for each component array begins,
    * ``sizes``: 1D array of sizes of the leading axis of each component array.
  We refer to this triple as a ragged array. (Offsets can't be computed from
  sizes in general to allow for internal padding.)

  For example::
    data: f32[8,3] = jnp.array([
        [a,b,c], [d,e,f], [g,h,i], [j,k,l], [m,n,o], [p,q,r], [s,t,u], [v,w,x],
    ])
    offsets: i32[3] = jnp.array([0, 1, 4])
    sizes: i32[3] = jnp.array([1, 3, 4])

    # To extract the first component array, of type f32[1,3]
    data[offsets[0]:offsets[0]+sizes[0]]

    # To extract the second component array, of type f32[3,3]
    data[offsets[1]:offsets[1]+sizes[1]]

    # To extract the third component array, of type f32[4,3]
    data[offsets[2]:offsets[2]+sizes[2]]

  The ``ragged_all_to_all`` collective operation communicates slices of ragged
  arrays between devices. Each caller is both a sender and a receiver. The
  ``input_offsets`` and ``send_sizes`` arguments indicate the slices of the
  caller's ``operand`` to be sent. Received results are returned in an array
  that has the same value of the argument ``output`` except with received values
  written at some slices. The ``output_offsets`` argument does *not* indicate
  the offsets at which all the received results are written; instead,
  ``output_offsets`` indicates the offsets at which the *sent* slices are
  written on their corresponding receivers. The sizes of received slices are
  indicated by ``recv_sizes``. See below for details.

  The arrays ``input_offsets``, ``send_sizes``,``output_offsets``, and
  ``recv_sizes`` must all be the same length, and that length must be divisible
  by the size of the mapped axis ``axis_name``. Moreover, ``send_sizes`` and
  ``recv_sizes`` must satisfy::

    jnp.all(send_sizes == jax.lax.all_to_all(recv_sizes, axis_name, 0, 0, tiled=True))

  Specifically, given a call::

    result = ragged_all_to_all(operand, output, input_offsets, send_sizes,
                               output_offsets, recv_sizes, axis_name)

  the caller sends data like::

    assert len(input_offsets) == len(send_sizes) == len(output_offsets) == len(recv_sizes)
    N = len(input_offsets)
    slices_per_device, leftover = divmod(N, lax.axis_size(axis_name))
    assert not leftover

    for i in range(N):
      dst_idx = i // slices_per_device
      SEND(data=operand[input_offsets[i]:input_offsets[i]+send_sizes[i]],
           axis_name=axis_name, to_axis_index=dst_idx)

  and receives data in ``result`` like::

    result = output
    output_offsets_ = jax.lax.all_to_all(output_offsets, axis_name, 0, 0, tiled=True)
    for i in range(N):
      src_idx = i // slices_per_device
      result = result.at[output_offsets_[i]:output_offsets_[i]+recv_sizes[i]
                    ].set(RECEIVE(axis_name=axis_name, from_axis_index=src_idx))

  where ``SEND`` and ``RECEIVE`` are pseudocode. Notice that a caller's local
  ``output_offsets`` does not indicate the offsets at which its local ``result``
  is updated; instead, it indicates where the corresponding sent slices are
  written on their destination instances. To compute the local offsets at which
  received data are written, we apply an ``all_to_all`` on ``output_offsets``.

  For example, if we apply a ``ragged_all_to_all`` along an axis of size 2, with
  these arguments in each mapped function instance::

    axis index 0:
      operand = [1, 2, 2]
      output = [0, 0, 0, 0]
      input_offsets = [0, 1]
      send_sizes = [1, 2]
      output_offsets = [0, 0]
      recv_sizes = [1, 1]

    axis index 1:
      operand = [3, 4, 0]
      output = [0, 0, 0, 0]
      input_offsets = [0, 1]
      send_sizes = [1, 1]
      output_offsets = [1, 2]
      recv_sizes = [2, 1]

  then::

    axis index 0:
      result = [1, 3, 0, 0]

    axis index 1:
      result = [2, 2, 4, 0]

  Args:
    operand: data array of shape (N, A, B, ...) representing concatenated
      (possibly padded) ragged data to be sent.
    output: data array of shape (M, A, B, ...) to update with received data.
    input_offsets: 1D integer array of shape (K,) representing the offsets of
      leading-axis slices into ``operand`` to be sent.
    send_sizes: 1D integer array array of shape (K,) representing the sizes of
      leading-axis slices into ``operand`` to be sent.
    output_offsets: 1D integer array of shape (K,) representing where the
      corresponding sent data is written on each corresponding receiver.
    recv_sizes: 1D integer array of shape (K,) representing sizes of
      leading-axis slices into ``output`` to update with received data.
    axis_name: name of the mapped axis over which to perform the communication.
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would run ragged all to all over the
      first two and last two replicas). Groups must cover all axis indices
      exactly once, and all groups must be the same size. Otherwise, the
      behavior is undefined.

  Returns:
    Array of shape (M, A, B, ...) with the same value as the ``output`` except
    with received data written into slices starting at
    ``all_to_all(output_offsets, axis_name, 0, 0, tiled=True)`` and with size
    ``recv_sizes``.
  """

  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)

  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  return ragged_all_to_all_p.bind(operand, output, input_offsets, send_sizes,
                                  output_offsets, recv_sizes,
                                  axis_name=axis_name,
                                  axis_index_groups=axis_index_groups)


def axis_index(axis_name: AxisName) -> jax.Array:
  """Return the index along the mapped axis ``axis_name``.

  Args:
    axis_name: hashable Python object used to name the mapped axis.

  Returns:
    An integer representing the index.

  For example, with 8 XLA devices available:

  >>> from functools import partial
  >>> @partial(jax.pmap, axis_name='i')
  ... def f(_):
  ...   return lax.axis_index('i')
  ...
  >>> f(jnp.zeros(4))
  Array([0, 1, 2, 3], dtype=int32)
  >>> f(jnp.zeros(8))
  Array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int32)
  >>> @partial(jax.pmap, axis_name='i')
  ... @partial(jax.pmap, axis_name='j')
  ... def f(_):
  ...   return lax.axis_index('i'), lax.axis_index('j')
  ...
  >>> x, y = f(jnp.zeros((4, 2)))
  >>> print(x)
  [[0 0]
  [1 1]
  [2 2]
  [3 3]]
  >>> print(y)
  [[0 1]
  [0 1]
  [0 1]
  [0 1]]
  """
  if not isinstance(axis_name, (tuple, list)):
    return axis_index_p.bind(axis_name=axis_name)
  else:
    inner_size = 1
    index = jnp.asarray(0)
    for name in reversed(axis_name):
      index += axis_index(name) * inner_size
      inner_size *= axis_size(name)
    return index


def axis_size(axis_name: AxisName) -> int:
  """Return the size of the mapped axis ``axis_name``.

  Args:
    axis_name: hashable Python object used to name the mapped axis.

  Returns:
    An integer representing the size.

  For example, with 8 XLA devices available:

  >>> from functools import partial
  >>> from jax.sharding import PartitionSpec as P
  >>> mesh = jax.make_mesh((8,), 'i')
  >>> @partial(jax.shard_map, mesh=mesh, in_specs=P('i'), out_specs=P())
  ... def f(_):
  ...   return lax.axis_size('i')
  ...
  >>> f(jnp.zeros(16))
  Array(8, dtype=int32, weak_type=True)
  >>> mesh = jax.make_mesh((4, 2), ('i', 'j'))
  >>> @partial(jax.shard_map, mesh=mesh, in_specs=P('i', 'j'), out_specs=P())
  ... def f(_):
  ...   return lax.axis_size(('i', 'j'))
  ...
  >>> f(jnp.zeros((16, 8)))
  Array(8, dtype=int32, weak_type=True)
  """
  return _axis_size(axis_name)


def _axis_size(
    axis_name: AxisName,
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    /,
) -> int:
  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  return psum(1, axis_name, axis_index_groups=axis_index_groups)


def pgather(src, idx, axes: int | AxisName):
  """Uses the last positional axis of idx to index into src's axes."""
  if not isinstance(axes, (tuple, list)):
    axes = (axes,)
  # TODO: Canonicalize exes!
  return pgather_p.bind(src, idx, axes=tuple(axes))

### parallel primitives

def _names_in_param(pname: str, params: core.ParamDict) -> tuple[str]:
  axis_names = params[pname]
  if isinstance(axis_names, (tuple, list)):
    return tuple(axis_names)
  else:
    return (axis_names,)

def _constant_reduction(prim, axis_data, args, axes, axis_index_groups):
  assert axis_data.name in axes
  if axis_index_groups: raise NotImplementedError
  new_axes = tuple(n for n in axes if n != axis_data.name)
  if new_axes:
    args = prim.bind(*args, axes=new_axes, axis_index_groups=axis_index_groups)
  if prim is psum_p:
    outs = [lax._const(x, axis_data.size) * x for x in args]
  elif prim in (pmin_p, pmax_p):
    outs = args
  else:
    raise Exception(f"Unrecognized reducer: {prim}")

  return outs, [None] * len(outs)

def _reduction_with_positional_batcher(
    prim, vals_in, dims_in, axis_index_groups,
    transform_unmapped, transform_mapped):
  if axis_index_groups is not None:
    raise NotImplementedError("axis_index_groups not supported in vmap collectives. "
                              "Please open a feature request!")
  vals_in = [val if d is batching.not_mapped or d == 0 else _moveaxis(d, 0, val)
             for val, d in zip(vals_in, dims_in)]
  mapped_vals_in, unmapped_vals_in = partitioned_vals_in = [], []
  mapped_idxs, unmapped_idxs = partitioned_idxs = [], []
  for i, (val, d) in enumerate(zip(vals_in, dims_in)):
    partitioned_vals_in[d is batching.not_mapped].append(val)
    partitioned_idxs[d is batching.not_mapped].append(i)
  vals_out = [None] * len(vals_in)
  if unmapped_vals_in:
    unmapped_axes, unmapped_vals_in = transform_unmapped(0, unmapped_vals_in)
    unmapped_vals_out = prim.bind(*unmapped_vals_in, axes=unmapped_axes, axis_index_groups=None)
    for i, val in zip(unmapped_idxs, unmapped_vals_out):
      vals_out[i] = val
  if mapped_vals_in:
    mapped_axes, mapped_vals_in = transform_mapped(0, mapped_vals_in)
    mapped_vals_out = prim.bind(*mapped_vals_in, axes=mapped_axes, axis_index_groups=None)
    for i, val in zip(mapped_idxs, mapped_vals_out):
      vals_out[i] = val
  assert all(v is not None for v in vals_out)
  return vals_out

def _reduction_batcher(prim, vals_in, dims_in, *, axes, axis_index_groups):
  assert prim.multiple_results
  if not any(isinstance(axis, int) for axis in axes):
    return prim.bind(*vals_in, axes=axes, axis_index_groups=axis_index_groups), dims_in
  vals_out = _reduction_with_positional_batcher(
      prim, vals_in, dims_in, axis_index_groups,
      lambda d, d_vals_in: (axes, d_vals_in),
      lambda d, d_vals_in: (tuple(axis + (axis >= d) if isinstance(axis, int) else axis
                                  for axis in axes),
                            d_vals_in))
  # _reduction_with_positional_batcher moves all map dims to 0
  return vals_out, [d if d is batching.not_mapped else 0 for d in dims_in]

def _batched_reduction_collective(
    prim, if_unmapped, axis_data, vals_in, dims_in, axes,
    axis_index_groups):
  assert prim.multiple_results
  if all(d is None for d in dims_in):
    if axis_data.name in axes:
      return _constant_reduction(prim, axis_data, vals_in, axes, axis_index_groups)
    else:
      return prim.bind(*vals_in, axes=axes, axis_index_groups=axis_index_groups), dims_in

  if axis_data.name not in axes:
    return _reduction_batcher(prim, vals_in, dims_in, axes=axes,
                              axis_index_groups=axis_index_groups)

  # Note that we have a choice here. We can either unfuse the reduction into one
  # that handles the batched dims and then another one that handles the rest.
  # Alternatively, we can keep the dimension reduction fused with the rest, but
  # we have to split the primitive into one for unmapped inputs and another
  # one for mapped, because they differ in their `axes` parameter.
  # We choose the second strategy here.
  vals_out = _reduction_with_positional_batcher(
      prim, vals_in, dims_in, axis_index_groups,
      lambda d, d_vals_in: (tuple(axis for axis in axes if axis != axis_data.name),
                            [if_unmapped(v, axis_data.size) for v in d_vals_in]),
      lambda d, d_vals_in: (tuple(axis + (axis >= d) if isinstance(axis, int) else
                                  axis if axis != axis_data.name else
                                  d for axis in axes),
                            d_vals_in))
  return vals_out, [batching.not_mapped] * len(vals_out)

def _replica_groups(axis_env, axis_name, axis_index_groups):
  replica_groups = pxla.axis_groups(axis_env, axis_name)
  if axis_index_groups is not None:
    replica_groups = [[axis_group[i] for i in axis_index_group]
                      for axis_group in replica_groups
                      for axis_index_group in axis_index_groups]
  return replica_groups

def _replica_groups_hlo(replica_groups: Sequence[Sequence[int]]
                        ) -> ir.DenseElementsAttr:
  # Uneven replica groups are padded with -1.
  groups = np.array(list(itertools.zip_longest(*replica_groups, fillvalue=-1)),
                    dtype=np.int64).T
  return ir.DenseIntElementsAttr.get(np.ascontiguousarray(groups))

def _allreduce_impl(prim, pos_reducer, *args, axes, axis_index_groups):
  assert axis_index_groups is None
  if not all(isinstance(axis, int) for axis in axes):
     return dispatch.apply_primitive(prim, *args, axes=axes,
                                     axis_index_groups=axis_index_groups)
  assert all(isinstance(axis, int) for axis in axes)
  return [pos_reducer(arg, axes) for arg in args]

def _allreduce_effectful_abstract_eval(*args, axes, axis_index_groups):
  _check_axis_names(axes)
  named_axes = tuple(axis for axis in axes if not isinstance(axis, int))
  pos_axes = tuple(axis for axis in axes if isinstance(axis, int))
  if axis_index_groups is not None:
    if len(pos_axes) != 0:
      raise ValueError(f"axis_index_groups can only be used with reductions over "
                       f"named axes, but got: {axes}")
  core.check_avals_context_mesh(args, 'all_reduce')
  out_avals = [
      ShapedArray(lax._reduce_op_shape_rule(arg, axes=pos_axes), arg.dtype,
                  sharding=lax._reduce_op_sharding_rule(arg, axes=pos_axes))
      for arg in args
  ]
  return out_avals, {core.NamedAxisEffect(axis) for axis in named_axes}

def _psum_invariant_abstract_eval(name, *args, axes, axis_index_groups):
  if not config._check_vma.value:
    return psum_p.abstract_eval(
        *args, axes=axes, axis_index_groups=axis_index_groups)

  assert isinstance(axes, tuple)
  _check_axis_names(axes)
  arg_vma = [a.vma for a in args]
  # If intersection between arg_vma and axes is empty, error
  if any(not set(axes) & a for a in arg_vma):
    raise ValueError(
        f"Collective {name} must be applied to a device-varying "
        f"type, but got {arg_vma} for collective acting "
        f"over axis name {axes}. Please open an issue at "
        "https://github.com/jax-ml/jax/issues, and as a temporary "
        "workaround pass the check_vma=False argument to `jax.shard_map`")

  named_axes = tuple(axis for axis in axes if not isinstance(axis, int))
  pos_axes = tuple(axis for axis in axes if isinstance(axis, int))
  if axis_index_groups is not None:
    if len(pos_axes) != 0:
      raise ValueError(
          "axis_index_groups can only be used with reductions over "
          f"named axes, but got: {axes}")
  core.check_avals_context_mesh(args, 'all_reduce')
  out_avals = [
      core.ShapedArray(
          lax._reduce_op_shape_rule(arg, axes=pos_axes), arg.dtype,
          sharding=lax._reduce_op_sharding_rule(arg, axes=pos_axes),
          vma=frozenset(a for a in arg.vma if a not in named_axes))
      for arg in args
  ]
  return out_avals, {core.NamedAxisEffect(axis) for axis in named_axes}

# TODO(yashkatariya): Replace this with _psum_invariant_abstract_eval
def _pmin_pmax_abstract_eval(name, *args, axes, axis_index_groups):
  if not config._check_vma.value:
    return _allreduce_effectful_abstract_eval(
        *args, axes=axes, axis_index_groups=axis_index_groups)
  return _psum_invariant_abstract_eval(
      name, *args, axes=axes, axis_index_groups=axis_index_groups)

def _check_axis_names(axes):
  named_axes = tuple(axis for axis in axes if not isinstance(axis, int))
  axis_env = core.get_axis_env()
  for name in named_axes:
    if not axis_env.axis_exists(name):
      raise NameError(f"unbound axis name: {name}")

def _allreduce_lowering(prim, pos_fn, ctx, *args, axes, axis_index_groups):
  if axis_index_groups is not None and ("tpu" in ctx.module_context.platforms):
    len_0 = len(axis_index_groups[0])
    if any(len(g) != len_0 for g in axis_index_groups):
      raise ValueError("axis_index_groups must all be the same size for TPU lowering")
  named_axes, positional_axes = axes_partition = [], []
  for axis in axes:
    axes_partition[isinstance(axis, int)].append(axis)

  if positional_axes:
    reducer = mlir.lower_fun(pos_fn, multiple_results=False)
    def _positional_reduce(aval, arg):
      aval_out = aval.update(
          shape=np.delete(np.array(aval.shape, dtype=np.int64),
                          positional_axes))
      reducer_ctx = ctx.replace(primitive=None, avals_in=[aval], avals_out=[aval_out])
      out, = reducer(reducer_ctx, arg, axes=tuple(positional_axes))
      return out
    args = map(_positional_reduce, ctx.avals_in, args)
  if not named_axes:
    return args

  replica_groups = _replica_groups_hlo(
      _replica_groups(ctx.module_context.axis_env, named_axes,
                      axis_index_groups))
  axis_context = ctx.module_context.axis_context
  is_spmd = isinstance(axis_context, (SPMDAxisContext, ShardingContext))

  def all_reduce(aval, x):
    if is_spmd:
      channel = ctx.module_context.new_channel()
      other_args = dict(
          channel_handle=hlo.ChannelHandle.get(
              channel, mlir.DEVICE_TO_DEVICE_TYPE),
          use_global_device_ids=ir.BoolAttr.get(True))
    else:
      other_args = {}

    if hlo.get_api_version() < 8:
      op = hlo.AllReduceOp(
          x.type, x, replica_groups=replica_groups, **other_args)
    else:
      op = hlo.AllReduceOp(
          [x.type], [x], replica_groups=replica_groups, **other_args)
    scalar_aval = core.ShapedArray(
        (), aval.dtype, sharding=NamedSharding(aval.sharding.mesh, P()))
    scalar_type = mlir.aval_to_ir_type(scalar_aval)
    reducer_block = op.regions[0].blocks.append(scalar_type, scalar_type)
    with ir.InsertionPoint(reducer_block):
      lower_reducer = mlir.lower_fun(prim.bind, multiple_results=False)
      reducer_ctx = ctx.replace(primitive=None,
                                avals_in=[scalar_aval] * 2, avals_out=[scalar_aval])
      out_nodes = lower_reducer(reducer_ctx, *reducer_block.arguments)
      hlo.return_(mlir.flatten_ir_values(out_nodes))
    return op.result

  return [all_reduce(aval, x) for aval, x in zip(ctx.avals_in, args)]


def _psum_transpose_rule(cts, *args, axes, axis_index_groups):
  named_axes, pos_axes = axes_partition = [], []
  for axis in axes:
    axes_partition[isinstance(axis, int)].append(axis)

  if pos_axes:
    def broadcast_positional(ct, arg):
      assert ad.is_undefined_primal(arg)
      if type(ct) is ad.Zero: return ad.Zero(arg.aval)
      return lax._reduce_sum_transpose_rule(ct, arg, axes=pos_axes)[0]
    cts = map(broadcast_positional, cts, args)

  # We treat psum as psum + pbroadcast, which is why the transpose reduces
  # over the named axes again (unlike for positional axes).
  nonzero_out_cts, treedef = tree_util.tree_flatten(cts)
  nonzero_in_cts = psum_p.bind(*nonzero_out_cts, axes=tuple(named_axes),
                               axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, nonzero_in_cts)

psum_p = core.Primitive('psum')
psum_p.multiple_results = True
psum_p.def_impl(partial(_allreduce_impl, psum_p, lax.reduce_sum))
psum_p.def_effectful_abstract_eval(_allreduce_effectful_abstract_eval)
mlir.register_lowering(
    psum_p, partial(_allreduce_lowering, lax.add_p, lax.reduce_sum))
ad.deflinear2(psum_p, _psum_transpose_rule)
batching.fancy_primitive_batchers[psum_p] = \
  partial(_batched_reduction_collective, psum_p, lambda v, axis_size: axis_size * v)
batching.skippable_batchers[psum_p] = partial(_names_in_param, 'axes')

pmax_p = core.Primitive('pmax')
pmax_p.multiple_results = True
pmax_p.def_impl(partial(_allreduce_impl, pmax_p, lax.reduce_max))
pmax_p.def_effectful_abstract_eval(partial(_pmin_pmax_abstract_eval, 'pmax'))
mlir.register_lowering(
    pmax_p, partial(_allreduce_lowering, lax.max_p, lax.reduce_max))
batching.fancy_primitive_batchers[pmax_p] = \
  partial(_batched_reduction_collective, pmax_p, lambda v, axis_size: v)
batching.skippable_batchers[pmax_p] = partial(_names_in_param, 'axes')


pmin_p = core.Primitive('pmin')
pmin_p.multiple_results = True
pmin_p.def_impl(partial(_allreduce_impl, pmin_p, lax.reduce_min))
pmin_p.def_effectful_abstract_eval(partial(_pmin_pmax_abstract_eval, 'pmin'))
mlir.register_lowering(
    pmin_p, partial(_allreduce_lowering, lax.min_p, lax.reduce_min))
batching.fancy_primitive_batchers[pmin_p] = \
  partial(_batched_reduction_collective, pmin_p, lambda v, axis_size: v)
batching.skippable_batchers[pmin_p] = partial(_names_in_param, 'axes')


def _ppermute_lowering(ctx, x, *, axis_name, perm):
  replica_groups = _replica_groups(ctx.module_context.axis_env, axis_name, None)
  group_size = len(replica_groups[0])
  srcs, dsts = unzip2((src % group_size, dst % group_size) for src, dst in perm)
  if not (len(srcs) == len(set(srcs)) and len(dsts) == len(set(dsts))):
    msg = "ppermute sources and destinations must be unique, got {}."
    raise ValueError(msg.format(perm))

  full_perm = np.zeros((len(replica_groups), len(perm), 2), np.int64)
  for i, grp in enumerate(replica_groups):
    grp = sorted(grp)
    for j, (src, dst) in enumerate(perm):
      full_perm[i, j, 0] = grp[src]
      full_perm[i, j, 1] = grp[dst]
  full_perm = full_perm.reshape((-1, 2))

  axis_context = ctx.module_context.axis_context
  is_manual = (
      isinstance(axis_context, SPMDAxisContext)
      and axis_context.manual_axes
  )
  if is_manual:
    channel = ctx.module_context.new_channel()
    other_args = dict(
        channel_handle=hlo.ChannelHandle.get(channel, mlir.DEVICE_TO_DEVICE_TYPE))
  else:
    other_args = {}

  return hlo.CollectivePermuteOp(
      x, mlir.dense_int_elements(full_perm), **other_args).results

def _ppermute_transpose_rule(t, x, perm, axis_name):
  srcs, dsts = unzip2(perm)
  inverse_perm = list(zip(dsts, srcs))
  return [ppermute(t, axis_name=axis_name, perm=inverse_perm)]

def _ppermute_batcher(axis_data, vals_in, dims_in, axis_name, perm):
  axis_size, frame_name = axis_data.size, axis_data.name
  (v,), (d,) = vals_in, dims_in
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  if axis_data.name not in axis_name:
    return ppermute_p.bind(v, perm=perm, axis_name=axis_name), d
  remaining_axes = tuple(axis for axis in axis_name if axis != frame_name)
  if remaining_axes:
    return ppermute_p.bind(v, perm=perm, axis_name=remaining_axes), d
  assert axis_name[0] == frame_name, "ppermute batcher called with a wrong axis!"
  assert len(perm) == axis_size, "Permutation doesn't match the axis size!"
  if d is batching.not_mapped:
    return v, d
  perm_indices = np.zeros(axis_size, dtype=int)
  for src, dst in perm:
    perm_indices[dst] = src
  return v.take(perm_indices, d), d

def _raise_to_shaped_abstract_eval(x, *, axis_name, **params):
  _check_axis_names(axis_name)
  collective_vma_rule('ppermute', axis_name, x)
  return x

ppermute_p = core.Primitive('ppermute')
ppermute_p.def_abstract_eval(_raise_to_shaped_abstract_eval)
ad.deflinear2(ppermute_p, _ppermute_transpose_rule)
mlir.register_lowering(ppermute_p, _ppermute_lowering)
batching.fancy_primitive_batchers[ppermute_p] = _ppermute_batcher
batching.skippable_batchers[ppermute_p] = partial(_names_in_param, 'axis_name')

def _pbroadcast_transpose_rule(t, x, source, axis_name):
  is_source = axis_index(axis_name) == source
  tsum = psum(t, axis_name)
  return [lax.select(is_source, lax.full_like(t, tsum), lax.full_like(t, 0))]

def _pbroadcast_batcher(axis_data, vals_in, dims_in, axis_name, source):
  axis_size = axis_data.size
  (v,), (d,) = vals_in, dims_in
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  if axis_data.name not in axis_name:
    return pbroadcast_p.bind(v, axis_name=axis_name, source=source), d
  remaining_axes = tuple(axis for axis in axis_name if axis != axis_data.name)
  if remaining_axes:
    raise NotImplementedError("pbroadcast batcher only supports a single axis")
  assert axis_name[0] == axis_data.name, "pbroadcast batcher called with a wrong axis!"
  assert source >= 0 and source < axis_size, "collective broadcast doesn't fit in the axis size!"
  if axis_size == 1 and remaining_axes:
    return pbroadcast_p.bind(v, source=source, axis_name=remaining_axes), d
  if d is batching.not_mapped:
    return v, d
  return v.take([source] * axis_size, d), d

def _pbroadcast_lowering(ctx, x, *, axis_name, source):
  replica_groups = _replica_groups(ctx.module_context.axis_env, axis_name, None)
  def source_to_front(group):
    return [group[source]] + list(group[:source]) + list(group[source + 1:])
  replica_groups = [source_to_front(group) for group in replica_groups]
  is_spmd = isinstance(
      ctx.module_context.axis_context,
      (SPMDAxisContext, ShardingContext),
  )
  if is_spmd:
    # We want to emit the collective-broadcast with global device IDs and a unique
    # channel ID, as otherwise it interprets the devices as replicas instead
    # of partitions - and XLA is configured with only a single replica.
    channel = ctx.module_context.new_channel()
    channel_handle = hlo.ChannelHandle.get(channel, mlir.DEVICE_TO_DEVICE_TYPE)
    other_args = dict(channel_handle=channel_handle)
  else:
    other_args = {}
  return hlo.CollectiveBroadcastOp(
      x, replica_groups=_replica_groups_hlo(replica_groups), **other_args
  ).results

pbroadcast_p = core.Primitive('pbroadcast')
pbroadcast_p.def_abstract_eval(_raise_to_shaped_abstract_eval)
ad.deflinear2(pbroadcast_p, _pbroadcast_transpose_rule)
mlir.register_lowering(pbroadcast_p, _pbroadcast_lowering, platform='gpu')
batching.fancy_primitive_batchers[pbroadcast_p] = _pbroadcast_batcher
batching.skippable_batchers[pbroadcast_p] = partial(_names_in_param, 'axis_name')


def _moveaxis(src, dst, x):
  perm = [i for i in range(x.ndim) if i != src]
  perm.insert(dst, src)
  return lax.transpose(x, perm)

def _splitaxis(axis, factor, x):
  new_shape = list(x.shape)
  assert new_shape[axis] % factor == 0, (new_shape[axis], factor)
  new_shape[axis:axis+1] = [factor, new_shape[axis] // factor]
  return x.reshape(new_shape)

def _foldaxis(axis, x):
  new_shape = list(x.shape)
  new_shape[axis:axis+2] = [x.shape[axis] * x.shape[axis + 1]]
  return x.reshape(new_shape)

def _all_to_all_lowering(
    ctx, x, *, split_axis, concat_axis, axis_name, axis_index_groups, tiled
):
  del tiled  # expand_dims and squeeze is done in `all_to_all` if `True`
  # Workaround for AllToAll not being implemented on CPU.
  replica_groups = _replica_groups(ctx.module_context.axis_env, axis_name,
                                   axis_index_groups)
  if len(replica_groups[0]) == 1:
    return [x]
  split_count = len(replica_groups[0])
  if not all(split_count == len(g) for g in replica_groups):
    raise ValueError('Replica groups must be equally sized')
  is_spmd = isinstance(
      ctx.module_context.axis_context,
      (SPMDAxisContext, ShardingContext),
  )
  if is_spmd:
    # We want to emit the all-gather with global device IDs and a unique
    # channel ID, as otherwise it interprets the devices as replicas instead
    # of partitions - and XLA is configured with only a single replica.
    channel = ctx.module_context.new_channel()
    channel_handle = hlo.ChannelHandle.get(channel, mlir.DEVICE_TO_DEVICE_TYPE)
    other_args = dict(channel_handle=channel_handle)
  else:
    other_args = {}
  if hlo.get_api_version() < 8:
    return hlo.AllToAllOp(
        x,
        split_dimension=mlir.i64_attr(split_axis),
        concat_dimension=mlir.i64_attr(concat_axis),
        split_count=mlir.i64_attr(split_count),
        replica_groups=_replica_groups_hlo(replica_groups),
        **other_args).results
  return hlo.AllToAllOp(
    [x],
    split_dimension=mlir.i64_attr(split_axis),
    concat_dimension=mlir.i64_attr(concat_axis),
    split_count=mlir.i64_attr(split_count),
    replica_groups=_replica_groups_hlo(replica_groups),
    **other_args).results

def _all_to_all_transpose_rule(
    cts, x, axis_name, split_axis, concat_axis, axis_index_groups, tiled
):
  return (all_to_all(
      cts,
      axis_name=axis_name,
      split_axis=concat_axis,
      concat_axis=split_axis,
      axis_index_groups=axis_index_groups,
      tiled=tiled),)

def _all_to_all_batcher(vals_in, dims_in, *, axis_name, split_axis, concat_axis, axis_index_groups,
                        tiled):
  x, = vals_in
  d, = dims_in
  result = all_to_all_p.bind(
      x,
      axis_name=axis_name,
      split_axis=split_axis + (d <= split_axis),
      concat_axis=concat_axis + (d <= concat_axis),
      axis_index_groups=axis_index_groups,
      tiled=tiled,
  )
  return result, d

def _all_to_all_batched_collective(axis_data, vals_in, dims_in,
                                   axis_name, split_axis, concat_axis,
                                   axis_index_groups, tiled):
  if axis_index_groups is not None:
    raise NotImplementedError("Please open a feature request!")

  axis_size, frame_name = axis_data.size, axis_data.name
  if isinstance(axis_name, (list, tuple)):
    axes_names = axis_name
  else:
    axes_names = [axis_name]
  if frame_name not in axes_names:
    return _all_to_all_batcher(
      vals_in, dims_in, axis_name=axis_name, split_axis=split_axis,
      concat_axis=concat_axis, axis_index_groups=axis_index_groups, tiled=tiled)

  x, = vals_in
  d, = dims_in
  if d is batching.not_mapped:
    # TODO(sharadmv,apaszke): Remove this broadcast that comes from
    # all_gather_transpose and instead avoid using all_to_all in
    # all_gather_transpose.
    x = lax.broadcast(x, (axis_size, *x.shape))
    d = 0
  if isinstance(axis_name, (list, tuple)):
    pos = axis_name.index(frame_name)
    major_axes, minor_axes = axis_name[:pos], axis_name[pos + 1:]
  else:
    major_axes, minor_axes = (), ()
  # Optimized case when no splitting is necessary
  if not major_axes and not minor_axes:
    if split_axis == concat_axis:
      axis = split_axis + (d <= split_axis)
      d_pre_split = d
      x = _splitaxis(axis, axis_size, x)
      d += (axis <= d)
      return _foldaxis(axis, moveaxis(x, (d, axis), (axis, d))), d_pre_split
    else:
      x_concat = _foldaxis(concat_axis, _moveaxis(d, concat_axis, x))
      return _splitaxis(split_axis, axis_size, x_concat), split_axis
  # Here we have to handle either the major or the minor dimensions
  # We will be accumulating chunks into the three leading dims: [Major, Current, Minor, ...]
  x, d = lax.expand_dims(_moveaxis(d, 0, x), (0, 2)), 1
  split_axis += 3; concat_axis += 3  # Offset by extra three leading dims

  if major_axes:
    x = all_to_all_p.bind(x, axis_name=major_axes,
                          split_axis=split_axis, concat_axis=0,
                          axis_index_groups=axis_index_groups,
                          tiled=tiled)
  # Split out the local part into axis new_d (NOTE: d is already in axis 1)
  assert d == 1
  x = _splitaxis(split_axis, axis_size, x)
  new_d = split_axis
  concat_axis += (split_axis <= concat_axis)  # Offset the existing axes by the new batch axis
  split_axis += 1
  if minor_axes:
    x = all_to_all_p.bind(x, axis_name=minor_axes,
                          split_axis=split_axis, concat_axis=2,
                          axis_index_groups=axis_index_groups,
                          tiled=tiled)

  # Fold the chunk axes into a single one
  x = _foldaxis(0, _foldaxis(0, x))
  split_axis -= 2; concat_axis -= 2; new_d -= 2
  # Fold gathered axes into concat_axis
  x = _foldaxis(concat_axis - 1, _moveaxis(0, concat_axis - 1, x))
  new_d -= 1  # We've removed 0th dimension, so new_d needs to be adjusted
  return x, new_d


def _all_to_all_effectful_abstract_eval(
    input_aval, axis_name, split_axis, concat_axis, axis_index_groups, tiled
):
  del tiled  # expand_dims and squeeze is done in `all_to_all` if `True`
  if not isinstance(axis_name, (list, tuple)):
    axis_name = (axis_name,)
  _check_axis_names(axis_name)
  shape = list(input_aval.shape)
  axis_size = (
      _axis_size(axis_name)
      if axis_index_groups is None
      else len(axis_index_groups[0])
  )
  assert shape[split_axis] % axis_size == 0, (shape[split_axis], axis_size)
  shape[split_axis] //= axis_size
  shape[concat_axis] *= axis_size
  vma = collective_vma_rule('all_to_all', axis_name, input_aval)
  out_aval = input_aval.update(shape=tuple(shape), weak_type=False, vma=vma)
  effects = {*map(core.NamedAxisEffect, axis_name)}
  return out_aval, effects


all_to_all_p = core.Primitive('all_to_all')
all_to_all_p.def_effectful_abstract_eval(_all_to_all_effectful_abstract_eval)
mlir.register_lowering(all_to_all_p, _all_to_all_lowering)
ad.deflinear2(all_to_all_p, _all_to_all_transpose_rule)
batching.fancy_primitive_batchers[all_to_all_p] = _all_to_all_batched_collective
batching.skippable_batchers[all_to_all_p] = partial(_names_in_param, 'axis_name')


def _ragged_all_to_all_lowering(
    ctx, operand, output, input_offsets, send_sizes, output_offsets, recv_sizes,
    *, axis_name, axis_index_groups
):
  replica_groups = _replica_groups(ctx.module_context.axis_env, axis_name,
                                   axis_index_groups)

  # Assumes all groups are the same size
  split_count = len(replica_groups[0])
  if not all(split_count == len(g) for g in replica_groups):
    raise ValueError('Replica groups must be equally sized')

  ragged_all_to_all_attrs = {
      "replica_groups": _replica_groups_hlo(replica_groups)
  }
  is_spmd = isinstance(
      ctx.module_context.axis_context, (SPMDAxisContext, ShardingContext))
  if is_spmd:
    ragged_all_to_all_attrs['channel_id'] = ir.IntegerAttr.get(
        ir.IntegerType.get_signless(64), ctx.module_context.new_channel()
    )

  return hlo.CustomCallOp(
      result=[output.type],
      inputs=[operand, output, input_offsets, send_sizes, output_offsets,
              recv_sizes],
      call_target_name=ir.StringAttr.get('ragged_all_to_all'),
      backend_config=ir.DictAttr.get(ragged_all_to_all_attrs),
      api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 4),
  ).results

def _ragged_all_to_all_effectful_abstract_eval(
    operand, output, input_offsets, send_sizes, output_offsets, recv_sizes,
    axis_name, axis_index_groups
):
  del operand, axis_index_groups
  if not dtypes.issubdtype(input_offsets.dtype, np.integer):
    raise ValueError("ragged_all_to_all input_offsets must be integer type.")
  if not dtypes.issubdtype(send_sizes.dtype, np.integer):
    raise ValueError("ragged_all_to_all send_sizes must be integer type.")
  if not dtypes.issubdtype(output_offsets.dtype, np.integer):
    raise ValueError("ragged_all_to_all output_offsets must be integer type.")
  if not dtypes.issubdtype(recv_sizes.dtype, np.integer):
    raise ValueError("ragged_all_to_all recv_sizes must be integer type.")
  if len(input_offsets.shape) != 1 or input_offsets.shape[0] < 1:
    raise ValueError(
        "ragged_all_to_all input_offsets must be rank 1 with positive dimension"
        " size, but got shape {}".format(input_offsets.shape)
    )
  if len(send_sizes.shape) != 1 or send_sizes.shape[0] < 1:
    raise ValueError(
        "ragged_all_to_all send_sizes must be rank 1 with positive dimension"
        " size, but got shape {}".format(send_sizes.shape)
    )
  if len(output_offsets.shape) != 1 or output_offsets.shape[0] < 1:
    raise ValueError(
        "ragged_all_to_all output_offsets must be rank 1 with positive"
        " dimension size, but got shape {}".format(output_offsets.shape)
    )
  if len(recv_sizes.shape) != 1 or recv_sizes.shape[0] < 1:
    raise ValueError(
        "ragged_all_to_all recv_sizes must be rank 1 with positive dimension"
        " size, but got shape {}".format(recv_sizes.shape)
    )

  _check_axis_names(axis_name)
  out_aval = output.update(shape=output.shape, weak_type=False)
  effects = {*map(core.NamedAxisEffect, axis_name)}
  return out_aval, effects

def _ragged_all_to_all_jvp(primals, tangents, **params):
  operand, output, *sizes_and_offsets = primals
  operand_dot, output_dot, *_ = tangents
  result = ragged_all_to_all_p.bind(
      operand, output, *sizes_and_offsets, **params)
  if type(operand_dot) is type(output_dot) is ad.Zero:
    result_dot = ad.Zero.from_primal_value(result)
  else:
    operand_dot = ad.instantiate_zeros(operand_dot)
    output_dot = ad.instantiate_zeros(output_dot)
    result_dot = ragged_all_to_all_p.bind(
        operand_dot, output_dot, *sizes_and_offsets, **params)
  return result, result_dot

def _ragged_all_to_all_transpose(
    t, operand, output, input_offsets, send_sizes, output_offsets, recv_sizes,
    *, axis_name, axis_index_groups):
  if type(t) is ad.Zero:
    operand_t = ad.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
    output_t = ad.Zero(output.aval) if ad.is_undefined_primal(output) else None
  else:
    zero = ad.zeros_like_aval(operand.aval)
    output_offsets_ = all_to_all(output_offsets, axis_name, 0, 0, tiled=True)
    input_offsets_ = all_to_all(input_offsets, axis_name, 0, 0, tiled=True)
    operand_t = ragged_all_to_all_p.bind(
        t, zero, output_offsets_, recv_sizes, input_offsets_, send_sizes,
        axis_name=axis_name, axis_index_groups=axis_index_groups)
    mask = jax.numpy.cumsum(
        jax.numpy.zeros(t.shape[0], dtype='int32').at[output_offsets_].set(1)\
        .at[output_offsets_ + recv_sizes].add(-1))
    mask = jax.numpy.expand_dims(mask, (*range(1, t.ndim),))
    output_t = jax.numpy.where(mask, 0, t)
  return [operand_t, output_t] + [None] * 4

def _ragged_all_to_all_batched_collective(axis_data, vals_in, dims_in,
                                          axis_name, axis_index_groups):
  del axis_data
  if axis_index_groups:
    raise NotImplementedError("Please open a feature request!")

  operand, output, input_offsets, send_sizes, output_offsets, recv_sizes = vals_in
  operand_dim, output_dim, input_offsets_dim, send_sizes_dim, output_offsets_dim, recv_sizes_dim = dims_in
  if not (operand.shape[operand_dim] == output.shape[output_dim] == input_offsets.shape[input_offsets_dim] == send_sizes.shape[send_sizes_dim] == output_offsets.shape[output_offsets_dim] == recv_sizes.shape[recv_sizes_dim]):
    raise ValueError("all operands must have the same batch sizes")

  sliced_results = []
  for i in range(operand.shape[operand_dim]):
    sliced_operand = slicing.slice_in_dim(operand, start_index=i, limit_index=i+1, axis=operand_dim).flatten()
    sliced_output = slicing.slice_in_dim(output, start_index=i, limit_index=i+1, axis=output_dim)
    sliced_output_shape = sliced_output.shape
    sliced_output = sliced_output.flatten()
    sliced_input_offsets = slicing.slice_in_dim(input_offsets, start_index=i, limit_index=i+1, axis=input_offsets_dim).flatten()
    sliced_send_sizes = slicing.slice_in_dim(send_sizes, start_index=i, limit_index=i+1, axis=send_sizes_dim).flatten()
    sliced_output_offsets = slicing.slice_in_dim(output_offsets, start_index=i, limit_index=i+1, axis=output_offsets_dim).flatten()
    sliced_recv_sizes = slicing.slice_in_dim(recv_sizes, start_index=i, limit_index=i+1, axis=recv_sizes_dim).flatten()
    sliced_result = ragged_all_to_all(sliced_operand, sliced_output, sliced_input_offsets, sliced_send_sizes, sliced_output_offsets, sliced_recv_sizes, axis_name=axis_name, axis_index_groups=axis_index_groups)
    sliced_result = lax.expand_dims(sliced_result.reshape(sliced_output_shape), dimensions=(output_dim,))
    sliced_results.append(sliced_result)

  concat_result = lax.concatenate(sliced_results, dimension=output_dim)
  return concat_result, operand_dim

ragged_all_to_all_p = core.Primitive('ragged_all_to_all')
ragged_all_to_all_p.def_effectful_abstract_eval(_ragged_all_to_all_effectful_abstract_eval)
ad.primitive_jvps[ragged_all_to_all_p] = _ragged_all_to_all_jvp
ad.primitive_transposes[ragged_all_to_all_p] = _ragged_all_to_all_transpose
mlir.register_lowering(ragged_all_to_all_p, _ragged_all_to_all_lowering)
batching.fancy_primitive_batchers[ragged_all_to_all_p] = _ragged_all_to_all_batched_collective
batching.skippable_batchers[ragged_all_to_all_p] = partial(_names_in_param, 'axis_name')

def insert_collective_pvary(axis_name, x):
  if not config._check_vma.value:
    return x

  axis_name = (axis_name,) if not isinstance(axis_name, tuple) else axis_name
  aval = core.get_aval(x)
  names_union = set(axis_name) | aval.vma
  x = pvary(x, tuple(n for n in names_union if n not in aval.vma))
  return x

def all_gather(x, axis_name, *, axis_index_groups=None, axis=0, tiled=False):
  """Gather values of x across all replicas.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  This is equivalent to, but faster than, all_to_all(broadcast(x)).

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would run all gather over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and all groups must be the same size.
    axis: a positional axis into which the chunks along ``axis_name`` will be
      concatenated.
    tiled: when ``False``, the chunks will be stacked into a fresh positional
      axis at index ``axis`` in the output. When ``True``, ``axis`` has to
      refer to an existing positional dimension and the chunks will be
      concatenated into that dimension.

  Returns:
    Array(s) representing the result of an all-gather along the axis
    ``axis_name``. Shapes are the same as ``x.shape``, but:

    - when ``tiled`` is ``False``, there is a new dimension equal to the
      size of axis ``axis_name`` in position ``axis``,
    - when ``tiled`` is ``True``, the size of dimension in position ``axis``
      is multiplied by the size of axis ``axis_name``.

  For example, with 4 XLA devices available:

  >>> x = np.arange(4)
  >>> y = jax.pmap(lambda x: jax.lax.all_gather(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [[0 1 2 3]
   [0 1 2 3]
   [0 1 2 3]
   [0 1 2 3]]

  An example of using axis_index_groups, groups split by even & odd device ids:

  >>> x = np.arange(16).reshape(4, 4)
  >>> print(x)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
  >>> def f(x):
  ...   return jax.lax.all_gather(
  ...       x, 'i', axis_index_groups=[[0, 2], [3, 1]])
  >>> y = jax.pmap(f, axis_name='i')(x)
  >>> print(y)
  [[[ 0  1  2  3]
    [ 8  9 10 11]]
   [[12 13 14 15]
    [ 4  5  6  7]]
   [[ 0  1  2  3]
    [ 8  9 10 11]]
   [[12 13 14 15]
    [ 4  5  6  7]]]
  """
  if not isinstance(axis_name, tuple):
    axis_name = axis_name,
  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  axis_size = _axis_size(axis_name, axis_index_groups)
  def bind(leaf):
    leaf = insert_collective_pvary(axis_name, leaf)
    return all_gather_p.bind(
        leaf,
        all_gather_dimension=canonicalize_axis(
            axis, np.ndim(leaf) if tiled else np.ndim(leaf) + 1),
        axis_name=axis_name, axis_index_groups=axis_index_groups,
        axis_size=axis_size, tiled=tiled)
  return tree_util.tree_map(bind, x)

def _all_gather_impl(x, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled):
  raise AssertionError("Unexpected call to _all_gather_impl")

def _all_gather_lowering(ctx, x, *, all_gather_dimension, axis_name,
                         axis_index_groups, axis_size, tiled,
                         platform=None):
  x_aval, = ctx.avals_in
  out_aval, = ctx.avals_out
  axis_context = ctx.module_context.axis_context
  is_spmd = isinstance(axis_context, (SPMDAxisContext, ShardingContext))
  if not tiled:
    new_shape = list(x_aval.shape)
    new_shape.insert(all_gather_dimension, 1)
    broadcast_dimensions = [i for i in range(len(new_shape)) if i != all_gather_dimension]
    x = hlo.broadcast_in_dim(
        mlir.aval_to_ir_type(x_aval.update(shape=new_shape)), x,
        mlir.dense_int_array(broadcast_dimensions))
  replica_groups = _replica_groups(ctx.module_context.axis_env, axis_name,
                                    axis_index_groups)
  if is_spmd:
    # We want to emit the all-gather with global device IDs and a unique
    # channel ID, as otherwise it interprets the devices as replicas instead
    # of partitions - and XLA is configured with only a single replica.
    channel = ctx.module_context.new_channel()
    other_args = dict(
        channel_handle=hlo.ChannelHandle.get(
            channel, mlir.DEVICE_TO_DEVICE_TYPE),
        use_global_device_ids=ir.BoolAttr.get(True))
  else:
    other_args = {}

  if hlo.get_api_version() < 8:
    return hlo.AllGatherOp(
        mlir.aval_to_ir_type(out_aval),
        x, all_gather_dim=mlir.i64_attr(all_gather_dimension),
        replica_groups=_replica_groups_hlo(replica_groups),
        **other_args).results
  return hlo.AllGatherOp(
      [mlir.aval_to_ir_type(out_aval)],
      [x], all_gather_dim=mlir.i64_attr(all_gather_dimension),
      replica_groups=_replica_groups_hlo(replica_groups),
      **other_args).results


def collective_vma_rule(prim_name, axis_name, x_aval):
  if not config._check_vma.value:
    return frozenset()
  axis_name = (axis_name,) if not isinstance(axis_name, tuple) else axis_name
  if any(a not in x_aval.vma for a in axis_name):
    raise ValueError(
        f"Collective {prim_name} must be applied to a device-varying "
        f" type, but got {x_aval.vma} for collective acting "
        f"over axis name {axis_name}. Please open an issue at "
        "https://github.com/jax-ml/jax/issues and as a temporary "
        "workaround pass the check_vma=False argument to `jax.shard_map`")
  return x_aval.vma

def _all_gather_effectful_abstract_eval(
    x_aval, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled
):
  if not isinstance(axis_name, (list, tuple)):
    axis_name = (axis_name,)
  _check_axis_names(axis_name)
  new_shape = list(x_aval.shape)
  if tiled:
    new_shape[all_gather_dimension] *= axis_size
  else:
    new_shape.insert(all_gather_dimension, axis_size)
  out_vma = collective_vma_rule('all_gather', axis_name, x_aval)
  return (x_aval.update(shape=new_shape, vma=out_vma),
          {*map(core.NamedAxisEffect, axis_name)})

def _all_gather_transpose_rule(cts, x, *, all_gather_dimension, axis_name,
                               axis_index_groups, axis_size, tiled):
  return (psum_scatter(cts, axis_name=axis_name,
                       scatter_dimension=all_gather_dimension,
                       axis_index_groups=axis_index_groups,
                       tiled=tiled),)

def _all_gather_batcher(prim, vals_in, dims_in, *, all_gather_dimension, axis_name,
                        axis_index_groups, axis_size, tiled):
  (x,), (d,) = vals_in, dims_in
  if d is not batching.not_mapped:
    if d <= all_gather_dimension:
      all_gather_dimension += 1
    elif not tiled:  # Tiled all-gather doesn't modify the set of dimensions
      d += 1
  if prim is all_gather_p:
    result = all_gather_p.bind(
        x, all_gather_dimension=all_gather_dimension, axis_name=axis_name,
        axis_index_groups=axis_index_groups, axis_size=axis_size,
        tiled=tiled)
    return result, d
  else:
    assert prim is all_gather_invariant_p
    result = all_gather_invariant_p.bind(
        x, all_gather_dimension=all_gather_dimension, axis_name=axis_name,
        axis_size=axis_size, tiled=tiled)
    return result, d

def _all_gather_batched_collective(prim, axis_data, vals_in, dims_in,
                                   all_gather_dimension, axis_name,
                                   axis_index_groups, axis_size, tiled):
  frame_size, frame_name = axis_data.size, axis_data.name
  if frame_name not in axis_name:
    return _all_gather_batcher(
        prim, vals_in, dims_in, all_gather_dimension=all_gather_dimension,
        axis_name=axis_name, axis_index_groups=axis_index_groups,
        axis_size=axis_size, tiled=tiled)
  if axis_index_groups is not None:
    raise NotImplementedError("axis_index_groups not supported in vmap")
  assert axis_size == frame_size, "axis size doesn't match"
  if not isinstance(axis_name, tuple):
    axis_name = (axis_name,)
  if len(axis_name) > 1:
    raise NotImplementedError("Please open a feature request!")
  assert axis_name == (frame_name,), "batcher called with wrong axis name"
  (x,), (d,) = vals_in, dims_in
  if d is batching.not_mapped:
    out_shape = list(np.shape(x))
    out_shape.insert(all_gather_dimension, axis_size)
    broadcast_dims = [i for i in range(len(out_shape)) if i != all_gather_dimension]
    y = lax.broadcast_in_dim(x, out_shape, broadcast_dims)
  else:
    y = _moveaxis(d, all_gather_dimension, x)
  if tiled:
    y = _foldaxis(all_gather_dimension, y)
  return y, batching.not_mapped

all_gather_p = core.Primitive('all_gather')
all_gather_p.def_effectful_abstract_eval(_all_gather_effectful_abstract_eval)
all_gather_p.def_impl(_all_gather_impl)
mlir.register_lowering(all_gather_p, _all_gather_lowering)
for p in ("cuda", "rocm", "tpu"):
  mlir.register_lowering(all_gather_p,
                         partial(_all_gather_lowering, platform=p),
                         platform=p)
ad.deflinear2(all_gather_p, _all_gather_transpose_rule)
batching.fancy_primitive_batchers[all_gather_p] = partial(
    _all_gather_batched_collective, all_gather_p)
batching.skippable_batchers[all_gather_p] = partial(_names_in_param, 'axis_name')


def all_gather_invariant(x, axis_name, *, axis: int = 0, tiled: bool = False):
  """Gather values of x across all replicas.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  all_gather_invariant differs from all_gather in the following ways:

  * all_gather_invariant is Varying -> Invariant.
    For example: `out: f32[8] = all_gather_invariant(inp: f32[4]{V: x}, 'x')`
    where the size of mesh axis `x` is 2.
    While all_gather is Varying -> Varying.

  * all_gather_invariant transposes to dynamic_slice which is
    Invariant -> Varying. While all_gather transposes to reduce_scatter
    which is Varying -> Varying.
  """
  if not isinstance(axis_name, tuple):
    axis_name = axis_name,
  axis_size = _axis_size(axis_name, None)
  axes_ = frozenset(axis_name)
  def bind(leaf):
    in_vma = core.typeof(leaf).vma
    if vary_names := axes_ - in_vma:
      leaf = pvary(leaf, tuple(vary_names))
    return all_gather_invariant_p.bind(
        leaf,
        all_gather_dimension=canonicalize_axis(axis, np.ndim(leaf) if tiled else
                                               np.ndim(leaf) + 1),
        axis_name=axis_name, axis_size=axis_size, tiled=tiled)
  return tree_util.tree_map(bind, x)

all_gather_invariant_p = core.Primitive('all_gather_invariant')

def _all_gather_invariant_effectful_abstract_eval(
    x_aval, *, all_gather_dimension, axis_name, axis_size, tiled
):
  _check_axis_names(axis_name)
  new_shape = list(x_aval.shape)
  if tiled:
    new_shape[all_gather_dimension] *= axis_size
  else:
    new_shape.insert(all_gather_dimension, axis_size)
  out_vma = frozenset(v for v in x_aval.vma if v not in axis_name)
  return (x_aval.update(shape=new_shape, vma=out_vma),
          {*map(core.NamedAxisEffect, axis_name)})

all_gather_invariant_p.def_effectful_abstract_eval(
    _all_gather_invariant_effectful_abstract_eval)

def _all_gather_invariant_impl(x, *, all_gather_dimension, axis_name, axis_size,
                               tiled):
  raise NotImplementedError
all_gather_invariant_p.def_impl(_all_gather_invariant_impl)


def _all_gather_invariant_lowering(
    ctx, x, *, all_gather_dimension, axis_name, axis_size, tiled, platform=None):
  return _all_gather_lowering(
      ctx, x, all_gather_dimension=all_gather_dimension, axis_name=axis_name,
      axis_index_groups=None, axis_size=axis_size, tiled=tiled,
      platform=platform)

mlir.register_lowering(all_gather_invariant_p, _all_gather_invariant_lowering)
for p in ("cuda", "rocm", "tpu"):
  mlir.register_lowering(all_gather_invariant_p,
                         partial(_all_gather_invariant_lowering, platform=p),
                         platform=p)

def _all_gather_invariant_transpose_rule(
    cts, x, *, all_gather_dimension, axis_name, axis_size, tiled):
  slice_size, rem = divmod(cts.shape[all_gather_dimension], axis_size)
  assert not rem
  idx = axis_index(axis_name) * slice_size
  out = slicing.dynamic_slice_in_dim(
      cts, idx, slice_size=slice_size, axis=all_gather_dimension)
  return (out,) if tiled else (lax.squeeze(out, [all_gather_dimension]),)
ad.deflinear2(all_gather_invariant_p, _all_gather_invariant_transpose_rule)

def _all_gather_invariant_batched_collective(
    axis_data, vals_in, dims_in, all_gather_dimension, axis_name, axis_size,
    tiled):
  return _all_gather_batched_collective(
      all_gather_invariant_p, axis_data, vals_in, dims_in, all_gather_dimension,
      axis_name, None, axis_size, tiled)
batching.fancy_primitive_batchers[all_gather_invariant_p] = _all_gather_invariant_batched_collective
batching.skippable_batchers[all_gather_invariant_p] = partial(_names_in_param, 'axis_name')


def _reduce_scatter_lowering(
    prim, ctx, x,
    *, scatter_dimension, axis_name,
    axis_index_groups, axis_size, tiled):
  x_aval, = ctx.avals_in
  aval_out, = ctx.avals_out
  scalar_aval = x_aval.update(shape=())
  replica_groups = _replica_groups(ctx.module_context.axis_env, axis_name,
                                   axis_index_groups)
  scatter_out_shape = list(x_aval.shape)
  scatter_out_shape[scatter_dimension] //= axis_size
  axis_context = ctx.module_context.axis_context
  is_spmd = isinstance(
      axis_context,
      (SPMDAxisContext, ShardingContext),
  )
  if is_spmd:
    # We want to emit the all-gather with global device IDs and a unique
    # channel ID, as otherwise it interprets the devices as replicas instead
    # of partitions - and XLA is configured with only a single replica.
    channel = ctx.module_context.new_channel()
    other_args = dict(
        channel_handle=hlo.ChannelHandle.get(
            channel, mlir.DEVICE_TO_DEVICE_TYPE),
        use_global_device_ids=ir.BoolAttr.get(True))
  else:
    other_args = {}
  op = hlo.ReduceScatterOp(
      mlir.aval_to_ir_type(x_aval.update(shape=scatter_out_shape)),
      x,
      scatter_dimension=mlir.i64_attr(scatter_dimension),
      replica_groups=_replica_groups_hlo(replica_groups),
      **other_args)
  scalar_type = mlir.aval_to_ir_type(scalar_aval)
  reducer_block = op.regions[0].blocks.append(scalar_type, scalar_type)
  with ir.InsertionPoint(reducer_block):
    lower_reducer = mlir.lower_fun(prim.bind, multiple_results=False)
    reducer_ctx = ctx.replace(primitive=None,
                              avals_in=[scalar_aval] * 2,
                              avals_out=[scalar_aval])
    out_nodes = lower_reducer(reducer_ctx, *reducer_block.arguments)
    hlo.return_(mlir.flatten_ir_values(out_nodes))

  if tiled:
    return op.results
  else:
    return [hlo.reshape(mlir.aval_to_ir_type(aval_out), op.result)]


def _reduce_scatter_effectful_abstract_eval(
    x_aval, *, axis_name, scatter_dimension, axis_index_groups, axis_size, tiled
):
  if not isinstance(axis_name, (list, tuple)):
    axis_name = (axis_name,)
  _check_axis_names(axis_name)
  new_shape = list(x_aval.shape)
  scatter_dim_input_size = x_aval.shape[scatter_dimension]
  if tiled:
    if scatter_dim_input_size % axis_size != 0:
      raise ValueError(f"tiled reduce_scatter operand scatter dimension size "
                       f"{scatter_dim_input_size} must be divisible by "
                       f"shard_count {axis_size}")
    new_shape[scatter_dimension] = scatter_dim_input_size // axis_size
  else:
    if scatter_dim_input_size != axis_size:
      raise ValueError(f"reduce_scatter operand scatter dimension size "
                       f"{scatter_dim_input_size} must match shard count "
                       f"{axis_size}")
    del new_shape[scatter_dimension]
  vma = collective_vma_rule('reduce_scatter', axis_name, x_aval)
  return (x_aval.update(shape=new_shape, vma=vma),
          {*map(core.NamedAxisEffect, axis_name)})


def _reduce_scatter_transpose_rule(cts, x, *, axis_name, scatter_dimension,
                                   axis_index_groups, axis_size, tiled):
  return (all_gather(cts, axis_name=axis_name,
                     axis_index_groups=axis_index_groups,
                     axis=scatter_dimension, tiled=tiled),)


def _reduce_scatter_batcher(vals_in, dims_in, *, scatter_dimension, axis_name,
                            axis_index_groups, axis_size, tiled):
  (x,), (d,) = vals_in, dims_in
  if d <= scatter_dimension:
    scatter_dimension += 1
  elif not tiled:  # Tiled all-scatter doesn't change the rank
    d += 1
  result = reduce_scatter_p.bind(
      x,
      scatter_dimension=scatter_dimension,
      axis_name=axis_name,
      axis_index_groups=axis_index_groups,
      axis_size=axis_size,
      tiled=tiled)
  return result, d

def _reduce_scatter_collective(axis_data, vals_in, dims_in,
                               scatter_dimension, axis_name,
                               axis_index_groups, axis_size, tiled):
  frame_size, frame_name = axis_data.size, axis_data.name
  if frame_name not in axis_name:
    return _reduce_scatter_batcher(
        vals_in, dims_in, scatter_dimension=scatter_dimension,
        axis_name=axis_name, axis_index_groups=axis_index_groups,
        axis_size=axis_size, tiled=tiled)
  if axis_index_groups is not None:
    raise NotImplementedError("axis_index_groups not supported in vmap")
  assert axis_size == frame_size, "axis size doesn't match"
  if not isinstance(axis_name, tuple):
    axis_name = (axis_name,)
  if len(axis_name) > 1:
    raise NotImplementedError("Please open a feature request!")
  assert axis_name == (frame_name,), "batcher called with wrong axis name"
  (x,), (d,) = vals_in, dims_in
  if d is batching.not_mapped:
    y, dy = x * axis_size, scatter_dimension
  else:
    y, dy = lax.reduce(x, 0., lax.add, (d,)), scatter_dimension
  if tiled:
    y = _splitaxis(dy, axis_size, y)
  return y, dy


reduce_scatter_p = core.Primitive("reduce_scatter")
reduce_scatter_p.def_effectful_abstract_eval(
    _reduce_scatter_effectful_abstract_eval
)
ad.deflinear2(reduce_scatter_p, _reduce_scatter_transpose_rule)
batching.fancy_primitive_batchers[reduce_scatter_p] = _reduce_scatter_collective
batching.skippable_batchers[reduce_scatter_p] = partial(_names_in_param, 'axis_name')

mlir.register_lowering(reduce_scatter_p,
                       partial(_reduce_scatter_lowering, lax.add_p))

def psum_scatter(x, axis_name, *, scatter_dimension=0, axis_index_groups=None,
                 tiled=False):
  """
  Like ``psum(x, axis_name)`` but each device retains only part of the result.

  For example, ``psum_scatter(x, axis_name, scatter_dimension=0, tiled=False)``
  computes the same value as ``psum(x, axis_name)[axis_index(axis_name)]``, but
  it is more efficient. Thus the ``psum`` result is left scattered along the
  mapped axis.

  One efficient algorithm for computing ``psum(x, axis_name)`` is to perform a
  ``psum_scatter`` followed by an ``all_gather``, essentially evaluating
  ``all_gather(psum_scatter(x, axis_name))``. So we can think of
  ``psum_scatter`` as "the first half" of a ``psum``.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a mapped axis (see the
      :func:`jax.pmap` documentation for more details).
    scatter_dimension: a positional axis into which the all-reduce result along
      ``axis_name`` will be scattered.
    axis_index_groups: optional list of lists of integers containing axis
      indices. For example, for an axis of size 4,
      ``axis_index_groups=[[0, 1], [2, 3]]`` would run reduce-scatter over the
      first two and the last two axis indices. Groups must cover all axis
      indices exactly once, and all groups must be the same size.
    tiled: boolean representing whether to use rank-preserving 'tiled' behavior.
      When ``False`` (the default value), the size of dimension in
      ``scatter_dimension`` must match the size of axis ``axis_name`` (or the
      group size if ``axis_index_groups`` is given). After scattering the
      all-reduce result along ``scatter_dimension``, the output is squeezed by
      removing ``scatter_dimension``, so the result has lower rank than the
      input. When ``True``, the size of dimension in ``scatter_dimension`` must
      be divisible by the size of axis ``axis_name`` (or the group size if
      ``axis_index_groups`` is given), and the ``scatter_dimension`` axis is
      preserved (so the result has the same rank as the input).

  Returns:
    Array(s) with the similar shape as ``x``, except the size of dimension in
    position ``scatter_dimension`` is divided by the size of axis ``axis_name``
    (when ``tiled=True``), or the dimension in position ``scatter_dimension`` is
    eliminated (when ``tiled=False``).

  For example, with 4 XLA devices available:

  >>> x = np.arange(16).reshape(4, 4)
  >>> print(x)
  [[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]
   [12 13 14 15]]
  >>> y = jax.pmap(lambda x: jax.lax.psum_scatter(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [24 28 32 36]

  if using tiled:

  >>> y = jax.pmap(lambda x: jax.lax.psum_scatter(x, 'i', tiled=True), axis_name='i')(x)
  >>> print(y)
  [[24]
   [28]
   [32]
   [36]]

  An example of using axis_index_groups:

  >>> def f(x):
  ...   return jax.lax.psum_scatter(
  ...       x, 'i', axis_index_groups=[[0, 2], [3, 1]], tiled=True)
  >>> y = jax.pmap(f, axis_name='i')(x)
  >>> print(y)
  [[ 8 10]
   [20 22]
   [12 14]
   [16 18]]
  """
  if not isinstance(axis_name, tuple):
    axis_name = axis_name,
  axis_size = _axis_size(axis_name, axis_index_groups)
  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  def bind(leaf):
    leaf = insert_collective_pvary(axis_name, leaf)
    return reduce_scatter_p.bind(
        leaf, axis_name=axis_name, scatter_dimension=scatter_dimension,
        axis_index_groups=axis_index_groups, axis_size=axis_size, tiled=tiled)
  return tree_util.tree_map(bind, x)


def _build_axis_index_lowering_hlo(ctx, axis_name, axis_env):
  if isinstance(axis_name, tuple):
    assert axis_name, 'empty axis name'
    if len(axis_name) > 1:
      raise NotImplementedError(
          '`axis_index` translation rule does not support multiple axis names.')
    axis_name, = axis_name
  if axis_name not in axis_env.names:
    raise NameError(f"unbound axis name: {axis_name}")
  axis_context = ctx.module_context.axis_context
  axis_pos = list(axis_env.names).index(axis_name)

  # For partial auto, enter into a fully manual shard_map.
  if (isinstance(axis_context, SPMDAxisContext) and
      axis_context.manual_axes and
      axis_context.manual_axes != frozenset(axis_context.mesh.axis_names)):
    if axis_env.sizes[axis_pos] == 1:
      return hlo.constant(ir.DenseElementsAttr.get(np.asarray(0, dtype=np.int32)))
    def f():
      return axis_index_p.bind(axis_name=axis_name)
    return mlir.lower_fun(
        lambda: [jax.shard_map(f, check_vma=False, in_specs=(),
                               out_specs=P())()])(ctx)[0]

  nreplicas = axis_env.nreps // math.prod(axis_env.sizes)
  div = mlir.ir_constant(
      np.array(
          nreplicas * math.prod(axis_env.sizes[axis_pos + 1 :]), dtype=np.uint32
      )
  )
  mod = mlir.ir_constant(np.array(axis_env.sizes[axis_pos], dtype=np.uint32))
  if isinstance(axis_context, (ShardingContext, SPMDAxisContext)):
    device_id = hlo.partition_id()
  else:
    device_id = hlo.replica_id()
  unsigned_index = hlo.remainder(hlo.divide(device_id, div), mod)
  return hlo.convert(
      ir.RankedTensorType.get([], ir.IntegerType.get_signless(32)),
      unsigned_index)

def _axis_index_lowering(ctx, *, axis_name):
  return [_build_axis_index_lowering_hlo(ctx, axis_name,
                                         ctx.module_context.axis_env)]

def _axis_index_effectful_abstract_eval(*, axis_name):
  effect = {core.NamedAxisEffect(axis_name)}
  axis_name = (axis_name,) if not isinstance(axis_name, tuple) else axis_name
  _check_axis_names(axis_name)
  mesh = get_abstract_mesh()
  sharding = NamedSharding(mesh, P())
  vma = ((frozenset(axis_name) if mesh._any_axis_manual else frozenset())
         if config._check_vma.value else frozenset())
  return ShapedArray((), np.int32, sharding=sharding, vma=vma), effect

def _axis_index_batcher(axis_data, vals_in, dims_in, *, axis_name):
  return lax.iota(np.int32, axis_data.size), 0

axis_index_p = core.Primitive('axis_index')
axis_index_p.def_impl(partial(dispatch.apply_primitive, axis_index_p))
mlir.register_lowering(axis_index_p, _axis_index_lowering)
axis_index_p.def_effectful_abstract_eval(_axis_index_effectful_abstract_eval)
batching.fancy_primitive_batchers[axis_index_p] = _axis_index_batcher
batching.skippable_batchers[axis_index_p] = partial(_names_in_param, 'axis_name')

def _pgather_impl(src, idx, *, axes):
  assert all(isinstance(axis, int) for axis in axes)
  src_axes_front = moveaxis(src, axes, range(len(axes)))
  non_axes_shape = src_axes_front.shape[len(axes):]
  src_one_axis_front = src_axes_front.reshape((-1,) + non_axes_shape)
  slice_sizes = (1,) + non_axes_shape
  idx = lax.expand_dims(idx, (-1,))
  offset_dims = tuple(range(idx.ndim - 1, idx.ndim + src_one_axis_front.ndim - 2))
  dnums = slicing.GatherDimensionNumbers(
      offset_dims=offset_dims,
      collapsed_slice_dims=(0,),
      start_index_map=(0,),
  )
  return slicing.gather(src_one_axis_front, idx, dimension_numbers=dnums,
                        slice_sizes=tuple(slice_sizes))

def _pgather_abstract_eval(src, idx, *, axes):
  # TODO: Avals with names rule: remove all axes from src, insert those from idx
  #       The order is important, because it is ok to re-insert one of the deleted axes!
  _check_axis_names(axes)
  shape = list(src.shape)
  for axis in sorted((a for a in axes if isinstance(a, int)), reverse=True):
    del shape[axis]
  shape = idx.shape + tuple(shape)
  return ShapedArray(shape, src.dtype)

def _pgather_parallel_lowering(ctx, src, idx, *, axes):
  if any(not isinstance(axis, int) for axis in axes):
    raise NotImplementedError("pgather only supported in the SPMD lowering."
                              "Please open a feature request!")
  return mlir.lower_fun(_pgather_impl, multiple_results=False)(
      ctx, src, idx, axes=axes)

def _pgather_collective_batcher(axis_size, frame_name, _, vals_in, dims_in, *, axes):
  src, idx = vals_in
  dsrc, didx = dims_in
  if dsrc is batching.not_mapped:
    raise ValueError("pgather axis {frame.name} is missing from the indexed value")
  if didx is not batching.not_mapped:
    # NOTE: This is allowed and the output would be mapped along this axis!
    raise NotImplementedError("Please open a feature request!")
  # Now source is mapped, idx is not
  new_axes = tuple(dsrc if axis == frame_name else
                   axis + (dsrc <= axis) if isinstance(axis, int) else
                   axis
                   for axis in axes)
  # The result is not mapped, because we eliminate all axes, and those include
  # the batched axis.
  if all(isinstance(axis, int) for axis in axes):
    # We rewrite a purely positional pgather as a gather, because that one
    # is more fully featured (e.g. supports AD).
    return _pgather_impl(src, idx, axes=new_axes), batching.not_mapped
  else:
    return pgather_p.bind(src, idx, axes=new_axes), batching.not_mapped

pgather_p = core.Primitive('pgather')
pgather_p.def_impl(_pgather_impl)
pgather_p.def_abstract_eval(_pgather_abstract_eval)
mlir.register_lowering(pgather_p, _pgather_parallel_lowering)
# TODO: Transpose? That requires adding pscatter...
batching.fancy_primitive_batchers[pgather_p] = _pgather_collective_batcher
batching.skippable_batchers[pgather_p] = partial(_names_in_param, 'axes')

psum_invariant_p = core.Primitive('psum_invariant')
psum_invariant_p.multiple_results = True
psum_invariant_p.def_impl(psum_p.impl)
psum_invariant_p.def_effectful_abstract_eval(
    partial(_psum_invariant_abstract_eval, psum_invariant_p.name))
mlir.register_lowering(psum_invariant_p, mlir._lowerings[psum_p])
batching.fancy_primitive_batchers[psum_invariant_p] = partial(
    _batched_reduction_collective, psum_invariant_p,
    lambda v, axis_size: axis_size * v)
batching.skippable_batchers[psum_invariant_p] = partial(_names_in_param, 'axes')

def _psum_invariant_transpose_rule(cts, *args, axes, axis_index_groups):
  def f(ct, arg):
    assert ad.is_undefined_primal(arg)
    return ad.Zero(arg.aval) if type(ct) is ad.Zero else ct
  cts = map(f, cts, args)
  nonzero_out_cts, treedef = tree_util.tree_flatten(cts)
  nonzero_in_cts = core.pvary_p.bind(*nonzero_out_cts, axes=axes,
                                     axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, nonzero_in_cts)
ad.deflinear2(psum_invariant_p, _psum_invariant_transpose_rule)

########################### pvary ##################################

def _pvary_transpose_rule(cts, *args, axes, axis_index_groups):
  def f(ct, arg):
    assert ad.is_undefined_primal(arg)
    return ad.Zero(arg.aval) if type(ct) is ad.Zero else ct
  cts = map(f, cts, args)
  nonzero_out_cts, treedef = tree_util.tree_flatten(cts)
  nonzero_in_cts = psum_invariant_p.bind(*nonzero_out_cts, axes=axes,
                                         axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, nonzero_in_cts)
ad.deflinear2(core.pvary_p, _pvary_transpose_rule)
