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
"""
Parallelization primitives.
"""

from jax.lax import lax
from jax.abstract_arrays import ShapedArray
from jax.core import Primitive
from jax.interpreters import ad
from jax.interpreters import parallel
from jax.interpreters import xla
from jax.interpreters import pxla
from jax.util import partial, unzip2
from jax.lib import xla_bridge


### parallel traceables

def psum(x, axis_name):
  """Compute an all-reduce sum on ``x`` over the pmapped axis ``axis_name``.

  Args:
    x: array with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).

  Returns:
    An array with the same shape as ``x`` representing the result of an
    all-reduce sum along the axis ``axis_name``.

  For example, with 4 XLA devices available:

  >>> x = np.arange(4)
  >>> y = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [6 6 6 6]
  >>> y = jax.pmap(lambda x: x / jax.lax.psum(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [ 0.          0.16666667  0.33333334  0.5       ]
  """
  return psum_p.bind(x, axis_name=axis_name)

def pmax(x, axis_name):
  """Compute an all-reduce max on ``x`` over the pmapped axis ``axis_name``.

  Args:
    x: array with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).

  Returns:
    An array with the same shape as ``x`` representing the result of an
    all-reduce max along the axis ``axis_name``.
  """
  return pmax_p.bind(x, axis_name=axis_name)

def pmin(x, axis_name):
  """Compute an all-reduce min on ``x`` over the pmapped axis ``axis_name``.

  Args:
    x: array with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).

  Returns:
    An array with the same shape as ``x`` representing the result of an
    all-reduce min along the axis ``axis_name``.
  """
  return pmin_p.bind(x, axis_name=axis_name)

def ppermute(x, axis_name, perm):
  """Perform a collective permutation according to the permutation ``perm``.

  This function is an analog of the CollectivePermute XLA HLO.

  Args:
    x: array with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).
    perm: list of pairs of ints, representing (source_index, destination_index)
      pairs that encode how the mapped axis named ``axis_name`` should be
      shuffled. The integer values are treated as indices into the mapped axis
      ``axis_name``. Any two pairs should not have the same source index or the
      same destination index. For each index of the axis ``axis_name`` that does
      not correspond to a destination index in ``perm``, the corresponding
      values in ``x`` are filled with zeros of the appropriate type.

  Returns:
    An array with the same shape as ``x`` representing the result of an
    all-reduce min along the axis ``axis_name``.
  """
  return ppermute_p.bind(x, axis_name=axis_name, perm=perm)

def pswapaxes(x, axis_name, axis):
  """Swap the pmapped axis ``axis_name`` with the unmapped axis ``axis``.

  This function is similar to ``psplit`` except the pmapped axis of the input is
  placed at the position ``axis`` in the output.

  Args:
    x: array with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).
    axis: int indicating the unmapped axis of ``x`` to map with the name
      ``axis_name``.

  Returns:
    An array with shape ``np.insert(np.delete(x.shape, axis), axis, axis_size)``
    where ``axis_size`` is the size of the mapped axis named ``axis_name`` in
    the input ``x``.
  """
  return pswapaxes_p.bind(x, axis_name=axis_name, axis=axis)

def psplit(x, axis_name, axis):
  """Unmap the pmapped axis ``axis_name`` and map ``axis`` with the same name.

  This function is similar to ``pswapaxes`` except the pmapped axis of the input
  is placed as the leading logical axis of the output.

  Args:
    x: array with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).
    axis: int indicating the unmapped axis of ``x`` to map with the name
      ``axis_name``.

  Returns:
    An array with shape ``(axis_size,) + tuple(np.delete(x.shape, axis))`` where
    ``axis_size`` is the size of the mapped axis named ``axis_name`` in the
    input ``x``.
  """
  return psplit_p.bind(x, axis_name=axis_name, axis=axis)

def psplit_like(x, y, axis_name):
  """Ensure the named mapped axis of ``x`` aligns with that of ``y``."""
  return psplit_like_p.bind(x, y, axis_name=axis_name)

def pcollect(x, axis_name):
  return pcollect_p.bind(x, axis_name=axis_name)


### parallel primitives

def _unbound_name_error(primitive_name, *args, **kwargs):
  axis_name = kwargs['axis_name']
  msg = "axis name '{}' is unbound for primitive {}."
  raise NameError(msg.format(axis_name, primitive_name))

def PmapPrimitive(name):
  prim = Primitive(name)
  prim.def_impl(partial(_unbound_name_error, name))
  prim.def_abstract_eval(lambda x, *args, **params: x)
  return prim


def _allreduce_serial_pmap_rule(reducer, vals, axes):
  val, = vals
  axis, = axes
  return reducer(val, [axis]), None

def _allreduce_translation_rule(prim, c, val, device_groups):
  dtype = c.GetShape(val).numpy_dtype()
  scalar = xla_bridge.Shape.array_shape(dtype, ())
  computation = xla.primitive_computation(prim, scalar, scalar)
  return c.AllReduce(val, computation, replica_groups=device_groups)


psum_p = PmapPrimitive('psum')
parallel.defreducer(lax.reduce_sum_p, psum_p)
parallel.serial_pmap_primitive_rules[psum_p] = \
    partial(_allreduce_serial_pmap_rule, lax._reduce_sum)
# TODO(mattjj): replace translation rule when we update jaxlib
# pxla.parallel_translation_rules[psum_p] = \
#     partial(_allreduce_translation_rule, lax.add_p)
pxla.parallel_translation_rules[psum_p] = \
    lambda c, val, device_groups: c.CrossReplicaSum(val, device_groups)
ad.deflinear(psum_p, lambda t, axis_name: [t])


pmax_p = PmapPrimitive('pmax')
parallel.defreducer(lax.reduce_max_p, pmax_p)
parallel.serial_pmap_primitive_rules[pmax_p] = \
    partial(_allreduce_serial_pmap_rule, lax._reduce_max)
pxla.parallel_translation_rules[pmax_p] = \
    partial(_allreduce_translation_rule, lax.max_p)


pmin_p = PmapPrimitive('pmin')
parallel.defreducer(lax.reduce_min_p, pmin_p)
parallel.serial_pmap_primitive_rules[pmin_p] = \
    partial(_allreduce_serial_pmap_rule, lax._reduce_min)
pxla.parallel_translation_rules[pmin_p] = \
    partial(_allreduce_translation_rule, lax.min_p)


def _ppermute_translation_rule(c, x, device_groups, perm):
  group_size = len(device_groups[0])
  srcs, dsts = unzip2((src % group_size, dst % group_size) for src, dst in perm)
  if not (len(srcs) == len(set(srcs)) and len(dsts) == len(set(dsts))):
    msg = "ppermute sources and destinations must be unique, got {}."
    raise ValueError(msg.format(perm))

  full_perm = []
  for grp in device_groups:
    grp = list(sorted(grp))
    full_perm.extend((grp[src], grp[dst]) for src, dst in perm)
  return c.CollectivePermute(x, full_perm)

def _ppermute_transpose_rule(t, perm, axis_name):
  sources, dests = unzip2(perm)
  inverse_perm = zip(dests, srcs)
  return ppermute(t, axis_name=axis_name, perm=inverse_perm)

ppermute_p = PmapPrimitive('ppermute')
# ad.deflinear(ppermute_p, _ppermute_transpose_rule)  # TODO(mattjj): test this
pxla.parallel_translation_rules[ppermute_p] = _ppermute_translation_rule


def _pswapaxes_serial_pmap_rule(vals, axes, axis):
  x, = vals
  axis_in, = axes
  if x.shape[axis_in] != x.shape[axis]:
    raise ValueError("pswapaxes between non-square dimensions")
  perm = list(range(x.ndim))
  perm[axis_in] = axis
  perm[axis] = axis_in
  return lax.transpose(x, perm), axis_in

pswapaxes_p = PmapPrimitive('pswapaxes')
parallel.serial_pmap_primitive_rules[pswapaxes_p] = _pswapaxes_serial_pmap_rule


def _psplit_serial_pmap_rule(vals, axes, axis):
  x, = vals
  axis_in, = axes
  if x.shape[axis_in] != x.shape[axis]:
    raise ValueError(
        "psplit between non-square dimensions {} and {} of {}".format(
            axis_in, axis, x.shape))
  return x, axis

psplit_p = PmapPrimitive('psplit')
parallel.serial_pmap_primitive_rules[psplit_p] = _psplit_serial_pmap_rule


def _psplit_like_serial_pmap_rule(vals, axes):
  x, y = vals
  xaxis, yaxis = axes
  if xaxis is not None and x.shape[xaxis] != x.shape[yaxis]:
    raise ValueError(
        "psplit_like is a non-square re-split along {} and {} of {}".format(
            xaxis, yaxis, x.shape))
  return x, yaxis

psplit_like_p = PmapPrimitive('psplit_like')
psplit_like_p.def_abstract_eval(
    lambda x, y, *args, **kwargs: ShapedArray(y.shape, x.dtype))
parallel.serial_pmap_primitive_rules[psplit_like_p] = _psplit_like_serial_pmap_rule


def _pcollect_serial_pmap_rule(vals, axes):
  x, = vals
  return x, None

pcollect_p = PmapPrimitive('pcollect')
parallel.serial_pmap_primitive_rules[pcollect_p] = _pcollect_serial_pmap_rule


### papply rules
# TODO(skye): it would be nice if we could put these with their corresponding
# primitives, but that currently causes circular dependencies. More refactoring
# might fix this.

def _dot_papply_rule(name, vals, dims):
  x, y = vals
  xdim, ydim = dims
  if xdim is None:
    return lax.dot(x, y), ydim
  elif ydim is None:
    return lax.dot(x, y), xdim
  elif ydim == 0:
    if xdim != x.ndim:
      x = psplit(x, name, x.ndim)
    x = x[..., None]
    y = y[..., None, :]
    return psum(x * y, name), None
  else:
    y = pcollect(y, name)
    return lax.dot(x, y), xdim


def _dot_general_papply_rule(name, vals, dims, dimension_numbers):
  x, y = vals
  xdim, ydim = dims

  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

  if len(lhs_batch) > 0 or len(rhs_batch) > 0:
    raise NotImplementedError

  def adjust_dims(dims, thresh):
    return tuple(i - 1 if i >= thresh else i for i in dims if i != thresh)

  sub_lhs_contract, sub_rhs_contract = lhs_contract, rhs_contract
  if xdim is not None:
    sub_lhs_contract = adjust_dims(lhs_contract, xdim)
  if ydim is not None:
    sub_rhs_contract = adjust_dims(rhs_contract, ydim)

  sub_dimension_numbers = (
      (sub_lhs_contract, sub_rhs_contract), (lhs_batch, rhs_batch))

  if xdim in lhs_contract and ydim in rhs_contract:
    z = lax.dot_general(x, y, sub_dimension_numbers)
    return psum(z, name), None
  elif xdim in lhs_contract:
    if ydim is not None:        # Cannot hide two dimensions, so collect one
      y = pcollect(y, name)
    return lax.dot_general(x, y, sub_dimension_numbers), xdim
  elif ydim in rhs_contract:
    if xdim is not None:        # Cannot hide two dimensions, so collect one
      x = pcollect(x, name)
    return lax.dot_general(x, y, sub_dimension_numbers), ydim
  elif xdim is not None:
    if ydim is not None:        # Cannot hide two dimensions, so collect one
      y = pcollect(y, name)
    return lax.dot_general(x, y, sub_dimension_numbers), xdim
  elif ydim is not None:
    return lax.dot_general(x, y, sub_dimension_numbers), ydim
  else:
    return lax.dot_general(x, y, sub_dimension_numbers), None


def _reshape_papply_rule(name, vals, axes, new_sizes, dimensions, old_sizes):
  operand, = vals
  axis, = axes

  def filter_ones(xs):
    return filter(lambda x: x != 1, xs)

  def find_new_axis(old_axis, old_sizes, new_sizes):
    if len(filter_ones(new_sizes)) != len(filter_ones(old_sizes)):
      return None
    num_before = len(filter_ones(old_sizes[:old_axis]))
    sz = old_sizes[old_axis]
    for i, new_sz in enumerate(new_sizes):
      if num_before == 0:
        if new_sz == sz:
          return i
        elif new_sz != 1:
          return None
      elif new_sz != 1:
        num_before -= 1
    return None

  err = NotImplementedError(
      'papply of reshape that would change hidden dimension size')

  if dimensions is None:
    new_axis = find_new_axis(axis, old_sizes, new_sizes)
    if new_axis is not None:
      if (lax.prod(old_sizes[:axis]) != lax.prod(new_sizes[:new_axis]) or
          lax.prod(old_sizes[axis + 1:]) != lax.prod(new_sizes[new_axis + 1:])):
        raise err
      new_sizes_ = new_sizes[:new_axis] + new_sizes[new_axis + 1:]
      return lax.reshape(operand, new_sizes_, dimensions=dimensions), new_axis
    else:
      raise err
  else:
    raise NotImplementedError('papply of reshape with `dimensions`')


def _transpose_papply_rule(name, vals, dims, permutation):
  x, = vals
  xdim, = dims
  perm = list(permutation)
  if perm[xdim] == xdim:
    x = lax.transpose(x, perm)
    out_dim = xdim
  else:
    in_dim, = [i for i in range(len(perm)) if perm[i] == xdim]
    out_dim = perm[xdim]
    perm[in_dim] = out_dim
    perm[out_dim] = in_dim
    perm = perm[:xdim] + perm[xdim + 1:]
    perm = [i - 1 if i > xdim else i for i in perm]
    x = lax.transpose(x, perm)
    x = pswapaxes(x, name, in_dim)
  return x, xdim


def _select_papply_rule(name, vals, dims):
  dimset = set([d for d in dims if d is not None])
  if len(dimset) != 1:
    raise NotImplementedError(
        'papply of select with operands split along different dimensions')
  like_val, like_dim = [(v, d) for v, d in zip(vals, dims) if d is not None][0]

  def normalize_split(val, dim):
    return psplit_like(val, like_val, name) if dim is None else val

  vals = [normalize_split(v, d) for v, d in zip(vals, dims)]
  return lax.select_p.bind(*vals), like_dim


parallel.papply_primitive_rules[lax.dot_p] = _dot_papply_rule
parallel.papply_primitive_rules[lax.dot_general_p] = _dot_general_papply_rule
parallel.papply_primitive_rules[lax.reshape_p] = _reshape_papply_rule
parallel.papply_primitive_rules[lax.transpose_p] = _transpose_papply_rule
parallel.papply_primitive_rules[lax.select_p] = _select_papply_rule
