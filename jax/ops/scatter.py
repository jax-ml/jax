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
    scatter_op: callable, either lax.scatter or lax.scatter_add.

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

  # Check if there's advanced indexing going on, and handle differently based on
  # whether it is or isn't mixed with basic indexing.
  if _is_advanced_int_indexer(idx):
    if np._is_advanced_int_indexer_without_slices(idx):
      if isinstance(idx, (tuple, list)):
        if any(onp.shape(e) for e in idx):
          # At least one sequence element in the index list means broadcasting.
          idx = np.broadcast_arrays(*idx)
        else:
          # The index list is a flat list of integers.
          idx = [lax.concatenate([lax.reshape(e, (1,)) for e in idx], 0)]
      else:
        # The indexer is just a single integer array.
        idx = [idx]

      stacked_idx = np.concatenate(
          [np.mod(np.reshape(a, (-1, 1)), np._constant_like(a, x.shape[i]))
          for i, a in enumerate(idx)], axis=1)

      y = np.broadcast_to(y, idx[0].shape + onp.shape(x)[len(idx):])
      y = lax.reshape(y, (stacked_idx.shape[0],) + onp.shape(x)[len(idx):])

      dnums = lax.ScatterDimensionNumbers(
          update_window_dims=tuple(range(1, y.ndim)),
          inserted_window_dims=tuple(range(len(idx))),
          scatter_dims_to_operand_dims=tuple(range(len(idx))))
      return scatter_op(x, stacked_idx, y, dnums)
    elif np._is_advanced_int_indexer(idx):
      # TODO(mattjj, phawkins): one of us is going to implement this case someday
      msg = "Unimplemented case for indexed update. Open a feature request!"
      raise NotImplementedError(msg)
    else:
      assert False  # unreachable

  # At this point there's no advanced indexing going on, so we process each
  # element of the index one at a time to build up a scatter.
  if not isinstance(idx, tuple):
    idx = (idx,)

  # Remove ellipses and add trailing slice(None)s.
  idx = np._canonicalize_tuple_index(x, idx)

  _int = lambda aval: not aval.shape and onp.issubdtype(aval.dtype, onp.integer)

  x_axis = 0
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

  for i in idx:
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
    inserted_window_dims = tuple(inserted_window_dims),
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
