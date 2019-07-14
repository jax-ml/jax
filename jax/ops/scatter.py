# Copyright 2019 Google LLC
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

# Helpers for indexed updates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as onp

from ..abstract_arrays import ShapedArray, ConcreteArray
from .. import core
from .. import lax
from ..numpy import lax_numpy as np


# TODO(mattjj): clean up this logic
def _is_advanced_int_indexer(idx):
  _int = lambda aval: not aval.shape and onp.issubdtype(aval.dtype, onp.integer)
  try:
    abstract_idx = core.get_aval(idx)
  except TypeError:
    abstract_idx = None
  out = not (isinstance(abstract_idx, ConcreteArray) and _int(abstract_idx) or
             isinstance(abstract_idx, ShapedArray) and _int(abstract_idx) or
             isinstance(idx, slice) or
             isinstance(idx, tuple) and all(onp.ndim(elt) == 0 for elt in idx))
  return out and np._is_advanced_int_indexer(idx)

def _triggers_unpack(x):
  return (isinstance(x, np.ndarray) or isinstance(x, collections.Sequence)
          or isinstance(x, slice) or x is Ellipsis or x is None)

def _scatter_update(x, idx, y, scatter_op):
  """Helper for indexed updates.

  Computes the value of x that would result from computing::
    x[idx] op= y
  except in a pure functional way, with no in-place updating.

  Args:
    x: ndarray to be updated.
    idx: None, an integer, a slice, an ellipsis, an ndarray with integer dtype,
      or a tuple of those indicating the locations of `x` into which to scatter-
      update the values in `y`.
    y: values to be scattered.
    scatter_op: callable, one of lax.scatter, lax.scatter_add, lax.scatter_min,
      or lax_scatter_max.

  Returns:
    An ndarray representing an updated `x` after performing the scatter-update.
  """
  # For more clues on the logic of this implementation, see the code for
  # jax.numpy._rewriting_take (which has links to NumPy docs).

  x = np.asarray(x)
  y = np.asarray(y)
  x_shape = np.shape(x)
  y_shape = np.shape(y)
  y = lax.convert_element_type(y, lax.dtype(x))

  # "Basic slicing is initiated if the selection object is a non-array,
  # non-tuple sequence containing slice objects, [Ellipses, or newaxis
  # objects]". Detects this case and canonicalizes to a tuple.
  if not isinstance(idx, tuple):
    if isinstance(idx, collections.Sequence) and not isinstance(idx, np.ndarray):
      if any(_triggers_unpack(i) for i in idx):
        idx = tuple(idx)
      else:
        idx = (idx,)
    else:
      idx = (idx,)

  # Remove ellipses and add trailing slice(None)s.
  idx = np._canonicalize_tuple_index(x, idx)

  # Check for advanced indexing.

  # Do the advanced indexing axes appear contiguously? If not, NumPy semantics
  # move the advanced axes to the front.
  advanced_axes_are_contiguous = False

  advanced_indexes = None

  # The positions of the advanced indexing axes in `idx`.
  idx_advanced_axes = []

  # The positions of the advanced indexes in x's shape.
  # collapsed, after None axes have been removed. See below.
  x_advanced_axes = None

  if _is_advanced_int_indexer(idx):
    idx_no_nones = [(i, d) for i, d in enumerate(idx) if d is not None]
    advanced_pairs = (
      (np.asarray(e), i, j) for j, (i, e) in enumerate(idx_no_nones)
      if (isinstance(e, collections.Sequence) or isinstance(e, np.ndarray)))
    advanced_pairs = ((np.mod(e, np._constant_like(e, x_shape[j])), i, j)
                      for e, i, j in advanced_pairs)
    advanced_indexes, idx_advanced_axes, x_advanced_axes = zip(*advanced_pairs)
    advanced_axes_are_contiguous = onp.all(onp.diff(idx_advanced_axes) == 1)

  _int = lambda aval: not aval.shape and onp.issubdtype(aval.dtype, onp.integer)

  x_axis = 0  # Current axis in x.
  y_axis = 0  # Current axis in y, before collapsing. See below.
  collapsed_y_axis = 0  # Current axis in y, after collapsing.

  # Scatter dimension numbers.
  update_window_dims = []
  inserted_window_dims = []
  scatter_dims_to_operand_dims = []

  scatter_indices = np.zeros((0,), dtype=np.int32)

  # We perform three transformations to y before the scatter op, in order:
  # First, y is broadcast to slice_shape. In general `y` only need broadcast to
  # the right shape.
  slice_shape = []
  # Next, y is reshaped to collapsed_slice_shape. This is to handle `None`
  # indices, which the scatter cannot remove itself.
  collapsed_slice_shape = []
  # Finally, we reverse reversed_y_dims to handle slices with negative strides.
  reversed_y_dims = []


  for idx_pos, i in enumerate(idx):
    # If the advanced indices are not contiguous they are moved to the front
    # of the slice. Otherwise, they replace the chunk of advanced indices.
    if (advanced_indexes is not None and
        (advanced_axes_are_contiguous and idx_pos == idx_advanced_axes[0] or
         not advanced_axes_are_contiguous and idx_pos == 0)):
      advanced_indexes = np.broadcast_arrays(*advanced_indexes)
      shape = advanced_indexes[0].shape
      ndim = len(shape)
      advanced_indexes = [
        lax.convert_element_type(lax.reshape(a, shape + (1,)), np.int32)
        for a in advanced_indexes]

      scatter_indices = lax.broadcast_in_dim(
        scatter_indices, onp.insert(scatter_indices.shape, -1, shape),
        tuple(range(scatter_indices.ndim - 1)) + (scatter_indices.ndim + ndim - 1,))
      scatter_indices = np.concatenate([scatter_indices] + advanced_indexes, -1)
      scatter_dims_to_operand_dims.extend(x_advanced_axes)
      inserted_window_dims.extend(x_advanced_axes)
      slice_shape.extend(shape)
      collapsed_slice_shape.extend(shape)
      y_axis += ndim
      collapsed_y_axis += ndim

    if idx_pos in idx_advanced_axes:
      x_axis += 1
      continue

    try:
      abstract_i = core.get_aval(i)
    except TypeError:
      abstract_i = None
    if (isinstance(abstract_i, ConcreteArray) or
        isinstance(abstract_i, ShapedArray)) and _int(abstract_i):
      i = np.mod(i, np._constant_like(i, x.shape[x_axis]))
      i = lax.convert_element_type(i, np.int32)
      i = np.broadcast_to(i, tuple(scatter_indices.shape[:-1]) + (1,))
      scatter_indices = np.concatenate((scatter_indices, i), -1)
      inserted_window_dims.append(x_axis)
      scatter_dims_to_operand_dims.append(x_axis)
      x_axis += 1
    elif i is None:
      slice_shape.append(1)
      y_axis += 1
    elif np._is_slice_none(i):
      slice_shape.append(x_shape[x_axis])
      collapsed_slice_shape.append(x_shape[x_axis])
      update_window_dims.append(collapsed_y_axis)
      collapsed_y_axis += 1
      y_axis += 1
      x_axis += 1
    elif isinstance(i, slice):
      start, limit, stride, needs_rev = np._static_idx(i, x.shape[x_axis])
      if needs_rev:
        reversed_y_dims.append(collapsed_y_axis)
      if stride == 1:
        i = lax.convert_element_type(start, np.int32)
        i = np.broadcast_to(i, tuple(scatter_indices.shape[:-1]) + (1,))
        scatter_indices = np.concatenate((scatter_indices, i), -1)
        slice_shape.append(limit - start)
        collapsed_slice_shape.append(limit - start)
        update_window_dims.append(collapsed_y_axis)
        scatter_dims_to_operand_dims.append(x_axis)
      else:
        i = np.arange(start, limit, stride, dtype=np.int32)
        size = i.shape[0]
        slice_shape.append(size)
        collapsed_slice_shape.append(size)
        scatter_indices_shape = tuple(scatter_indices.shape[:-1]) + (size,)
        i = lax.broadcast_in_dim(
            i, shape=scatter_indices_shape + (1,),
            broadcast_dimensions=(len(scatter_indices_shape) - 1,))
        scatter_indices = lax.broadcast_in_dim(
            scatter_indices,
            shape=scatter_indices_shape + (len(scatter_dims_to_operand_dims),),
            broadcast_dimensions=(
              tuple(range(len(scatter_indices_shape) - 1)) +
              (len(scatter_indices_shape),)))
        scatter_indices = np.concatenate(
          (scatter_indices, i), len(scatter_indices_shape))
        scatter_dims_to_operand_dims.append(x_axis)
        inserted_window_dims.append(x_axis)

      collapsed_y_axis += 1
      y_axis += 1
      x_axis += 1
    else:
      raise IndexError("Unknown index type ", i)

  y = np.broadcast_to(y, tuple(slice_shape))
  y = lax.reshape(y, collapsed_slice_shape)
  if reversed_y_dims:
    y = lax.rev(y, reversed_y_dims)

  dnums = lax.ScatterDimensionNumbers(
    update_window_dims = tuple(update_window_dims),
    inserted_window_dims = tuple(sorted(inserted_window_dims)),
    scatter_dims_to_operand_dims = tuple(scatter_dims_to_operand_dims)
  )
  return scatter_op(x, scatter_indices, y, dnums)


class _Indexable(object):
  """Helper object for building indexes for indexed update functions.

  This is a singleton object that overrides the :code:`__getitem__` method
  to return the index it is passed.

  >>> jax.ops.index[1:2, 3, None, ..., ::2]
  (slice(1, 2, None), 3, None, Ellipsis, slice(None, None, 2))
  """
  __slots__ = ()

  def __getitem__(self, index):
    return index

#: Index object singleton
index = _Indexable()


def index_add(x, idx, y):
  """Pure equivalent of :code:`x[idx] += y`.

  Returns the value of `x` that would result from the
  NumPy-style :mod:`indexed assignment <numpy.doc.indexing>`::
    x[idx] += y

  Note the `index_add` operator is pure; `x` itself is
  not modified, instead the new value that `x` would have taken is returned.

  Unlike the NumPy code :code:`x[idx] += y`, if multiple indices refer to the
  same location the updates will be summed. (NumPy would only apply the last
  update, rather than summing the updates.) The order in which conflicting
  updates are applied is implementation-defined and may be nondeterministic
  (e.g., due to concurrency on some hardware platforms).

  Args:
    x: an array with the values to be updated.
    idx: a Numpy-style index, consisting of `None`, integers, `slice` objects,
      ellipses, ndarrays with integer dtypes, or a tuple of the above. A
      convenient syntactic sugar for forming indices is via the
      :data:`jax.ops.index` object.
    y: the array of updates. `y` must be broadcastable to the shape of the
      array that would be returned by `x[idx]`.

  Returns:
    An array.

  >>> x = jax.numpy.ones((5, 6))
  >>> jax.ops.index_add(x, jax.ops.index[2:4, 3:], 6.)
  array([[1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 7., 7., 7.],
         [1., 1., 1., 7., 7., 7.],
         [1., 1., 1., 1., 1., 1.]], dtype=float32)
  """
  return _scatter_update(x, idx, y, lax.scatter_add)

def index_min(x, idx, y):
  """Pure equivalent of :code:`x[idx] = minimum(x[idx], y)`.

  Returns the value of `x` that would result from the
  NumPy-style :mod:`indexed assignment <numpy.doc.indexing>`::
    x[idx] = minimum(x[idx], y)

  Note the `index_min` operator is pure; `x` itself is
  not modified, instead the new value that `x` would have taken is returned.

  Unlike the NumPy code :code:`x[idx] = minimum(x[idx], y)`, if multiple indices
  refer to the same location the final value will be the overall min. (NumPy
  would only look at the last update, rather than all of the updates.)

  Args:
    x: an array with the values to be updated.
    idx: a Numpy-style index, consisting of `None`, integers, `slice` objects,
      ellipses, ndarrays with integer dtypes, or a tuple of the above. A
      convenient syntactic sugar for forming indices is via the
      :data:`jax.ops.index` object.
    y: the array of updates. `y` must be broadcastable to the shape of the
      array that would be returned by `x[idx]`.

  Returns:
    An array.

  >>> x = jax.numpy.ones((5, 6))
  >>> jax.ops.index_minimum(x, jax.ops.index[2:4, 3:], 0.)
  array([[1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 0., 0., 0.],
         [1., 1., 1., 0., 0., 0.],
         [1., 1., 1., 1., 1., 1.]], dtype=float32)
  """
  return _scatter_update(x, idx, y, lax.scatter_min)

def index_max(x, idx, y):
  """Pure equivalent of :code:`x[idx] = maximum(x[idx], y)`.

  Returns the value of `x` that would result from the
  NumPy-style :mod:`indexed assignment <numpy.doc.indexing>`::
    x[idx] = maximum(x[idx], y)

  Note the `index_max` operator is pure; `x` itself is
  not modified, instead the new value that `x` would have taken is returned.

  Unlike the NumPy code :code:`x[idx] = maximum(x[idx], y)`, if multiple indices
  refer to the same location the final value will be the overall max. (NumPy
  would only look at the last update, rather than all of the updates.)

  Args:
    x: an array with the values to be updated.
    idx: a Numpy-style index, consisting of `None`, integers, `slice` objects,
      ellipses, ndarrays with integer dtypes, or a tuple of the above. A
      convenient syntactic sugar for forming indices is via the
      :data:`jax.ops.index` object.
    y: the array of updates. `y` must be broadcastable to the shape of the
      array that would be returned by `x[idx]`.

  Returns:
    An array.

  >>> x = jax.numpy.ones((5, 6))
  >>> jax.ops.index_max(x, jax.ops.index[2:4, 3:], 6.)
  array([[1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 6., 6., 6.],
         [1., 1., 1., 6., 6., 6.],
         [1., 1., 1., 1., 1., 1.]], dtype=float32)
  """
  return _scatter_update(x, idx, y, lax.scatter_max)

def index_update(x, idx, y):
  """Pure equivalent of :code:`x[idx] = y`.

  Returns the value of `x` that would result from the
  NumPy-style :mod:`indexed assignment <numpy.doc.indexing>`::
    x[idx] = y

  Note the `index_update` operator is pure; `x` itself is
  not modified, instead the new value that `x` would have taken is returned.

  Unlike NumPy's :code:`x[idx] = y`, if multiple indices refer to the same
  location it is undefined which update is chosen; JAX may choose the order of
  updates arbitrarily and nondeterministically (e.g., due to concurrent
  updates on some hardware platforms).

  Args:
    x: an array with the values to be updated.
    idx: a Numpy-style index, consisting of `None`, integers, `slice` objects,
      ellipses, ndarrays with integer dtypes, or a tuple of the above. A
      convenient syntactic sugar for forming indices is via the
      :data:`jax.ops.index` object.
    y: the array of updates. `y` must be broadcastable to the shape of the
      array that would be returned by `x[idx]`.

  Returns:
    An array.

  >>> x = jax.numpy.ones((5, 6))
  >>> jax.ops.index_update(x, jax.ops.index[::2, 3:], 6.)
  array([[1., 1., 1., 6., 6., 6.],
         [1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 6., 6., 6.],
         [1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 6., 6., 6.]], dtype=float32)
  """
  return _scatter_update(x, idx, y, lax.scatter)

def segment_sum(data, segment_ids, num_segments=None):
  """Computes the sum within segments of an array.

  Similar to TensorFlow's segment_sum:
  https://www.tensorflow.org/api_docs/python/tf/math/segment_sum

  Args:
    data: an array with the values to be summed.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be summed. Values can be repeated and
      need not be sorted. Values outside of the range [0, num_segments) are
      wrapped into that range by applying np.mod.
    num_segments: optional, an int with positive value indicating the number of
      segments. The default is ``max(segment_ids % data.shape[0]) + 1`` but
      since `num_segments` determines the size of the output, a static value
      must be provided to use `segment_sum` in a `jit`-compiled function.

  Returns:
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
    segment sums.
  """
  if num_segments is None:
    num_segments = np.max(np.mod(segment_ids, data.shape[0])) + 1
  num_segments = int(num_segments)

  out = np.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
  segment_ids = np.mod(segment_ids, num_segments)
  return index_add(out, segment_ids, data)
