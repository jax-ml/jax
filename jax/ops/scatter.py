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


from .. import lax
from ..numpy import lax_numpy as jnp


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

  x = jnp.asarray(x)
  y = jnp.asarray(y)
  # XLA gathers and scatters are very similar in structure; the scatter logic
  # is more or less a transpose of the gather equivalent.
  treedef, static_idx, dynamic_idx = jnp._split_index_for_jit(idx)
  return _scatter_impl(x, y, scatter_op, treedef, static_idx, dynamic_idx)


# TODO(phawkins): re-enable jit after fixing excessive recompilation for
# slice indexes (e.g., slice(0, 5, None), slice(10, 15, None), etc.).
# @partial(jit, static_argnums=(2, 3, 4))
def _scatter_impl(x, y, scatter_op, treedef, static_idx, dynamic_idx):
  dtype = lax.dtype(x)
  x, y = jnp._promote_dtypes(x, y)

  idx = jnp._merge_static_and_dynamic_indices(treedef, static_idx, dynamic_idx)
  indexer = jnp._index_to_gather(jnp.shape(x), idx)

  # Broadcast `y` to the slice output shape.
  y = jnp.broadcast_to(y, tuple(indexer.slice_shape))
  # Collapse any `None`/`jnp.newaxis` dimensions.
  y = jnp.squeeze(y, axis=indexer.newaxis_dims)
  if indexer.reversed_y_dims:
    y = lax.rev(y, indexer.reversed_y_dims)

  # Transpose the gather dimensions into scatter dimensions (cf.
  # lax._gather_transpose_rule)
  dnums = lax.ScatterDimensionNumbers(
    update_window_dims=indexer.dnums.offset_dims,
    inserted_window_dims=indexer.dnums.collapsed_slice_dims,
    scatter_dims_to_operand_dims=indexer.dnums.start_index_map
  )
  out = scatter_op(x, indexer.gather_indices, y, dnums)
  return lax.convert_element_type(out, dtype)


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


def index_mul(x, idx, y):
  """Pure equivalent of :code:`x[idx] *= y`.

  Returns the value of `x` that would result from the
  NumPy-style :mod:`indexed assignment <numpy.doc.indexing>`::
    x[idx] *= y

  Note the `index_mul` operator is pure; `x` itself is
  not modified, instead the new value that `x` would have taken is returned.

  Unlike the NumPy code :code:`x[idx] *= y`, if multiple indices refer to the
  same location the updates will be multiplied. (NumPy would only apply the last
  update, rather than multiplying the updates.) The order in which conflicting
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
  >>> jax.ops.index_mul(x, jax.ops.index[2:4, 3:], 6.)
  array([[1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 6., 6., 6.],
         [1., 1., 1., 6., 6., 6.],
         [1., 1., 1., 1., 1., 1.]], dtype=float32)
  """
  return _scatter_update(x, idx, y, lax.scatter_mul)


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
      wrapped into that range by applying jnp.mod.
    num_segments: optional, an int with positive value indicating the number of
      segments. The default is ``max(segment_ids % data.shape[0]) + 1`` but
      since `num_segments` determines the size of the output, a static value
      must be provided to use `segment_sum` in a `jit`-compiled function.

  Returns:
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
    segment sums.
  """
  if num_segments is None:
    num_segments = jnp.max(jnp.mod(segment_ids, data.shape[0])) + 1
  num_segments = int(num_segments)

  out = jnp.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
  segment_ids = jnp.mod(segment_ids, num_segments)
  return index_add(out, segment_ids, data)


def segment_max(data, segment_ids, num_segments=None):
  """Computes the max within segments of an array.

  Similar to TensorFlow's segment_max:
  https://www.tensorflow.org/api_docs/python/tf/math/segment_max
  Args:
    data: an array with the values to be aggregated by max.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis). Values can be repeated and
      need not be sorted.
    num_segments: optional, an int with positive value indicating the number of
      segments. The default is ``max(segment_ids % data.shape[0]) + 1`` but
      since `num_segments` determines the size of the output, a static value
      must be provided to use `segment_max` in a `jit`-compiled function.
  Returns:
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing
    the max of the associated segments.
  """
  if num_segments is None:
    num_segments = jnp.max(jnp.mod(segment_ids, data.shape[0])) + 1
  num_segments = int(num_segments)

  full_zeros = jnp.zeros((num_segments,) + data.shape[1:], dtype=jnp.bool_)
  full_ones = jnp.ones((num_segments,) + data.shape[1:], dtype=data.dtype)

  valid_updates = index_update(full_zeros, segment_ids,
                               jnp.ones_like(data, dtype=jnp.bool_))
  updated = index_max(full_ones * -jnp.inf, segment_ids, data)

  return jnp.where(valid_updates, updated, full_zeros)
