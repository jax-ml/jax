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

import numpy as onp

from jax import ad_util
from jax.lax import lax
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad
from jax.interpreters import parallel
from jax.interpreters import xla
from jax.interpreters import pxla
from jax.util import partial, unzip2, prod
from jax.lib import xla_bridge

from jax.interpreters.pxla import axis_index


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

  The mapped axis size must be equal to the size of the unmapped axis; that is,
  we must have ``lax.psum(1, axis_name) == x.shape[axis]``.

  This function is a special case of ``all_to_all`` where the pmapped axis of
  the input is placed at the position ``axis`` in the output. That is, it is
  equivalent to ``all_to_all(x, axis_name, axis, axis)``.

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
  return all_to_all(x, axis_name, axis, axis)

def all_to_all(x, axis_name, split_axis, concat_axis):
  """Materialize the mapped axis and map a different axis.

  In the output, the input mapped axis ``axis_name`` is materialized at the
  logical axis position ``concat_axis``, and the input unmapped axis at position
  ``split_axis`` is mapped with the name ``axis_name``.

  The input mapped axis size must be equal to the size of the axis to be mapped;
  that is, we must have ``lax.psum(1, axis_name) == x.shape[split_axis]``.

  Args:
    x: array with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).
    split_axis: int indicating the unmapped axis of ``x`` to map with the name
      ``axis_name``.
    concat_axis: int indicating the position in the output to materialize the
      mapped axis of the input with the name ``axis_name``.

  Returns:
    An array with shape given by the expression::
      np.insert(np.delete(x.shape, split_axis), concat_axis, axis_size)

    where ``axis_size`` is the size of the mapped axis named ``axis_name`` in
    the input ``x``, i.e. ``axis_size = lax.psum(1, axis_name)``.
  """
  if psum(1, axis_name) != x.shape[split_axis]:
    msg = ("all_to_all requires the size of the mapped axis axis_name to equal "
          "x.shape[split_axis], but they are {} and {} respectively.")
    raise ValueError(msg.format(psum(1, axis_name), x.shape[split_axis]))
  return all_to_all_p.bind(x, split_axis=split_axis, concat_axis=concat_axis,
                           axis_name=axis_name)


def pcollect(x, axis_name):
  return pcollect_p.bind(x, axis_name=axis_name)


### parallel primitives

def standard_pmap_primitive(name):
  prim = pxla.PmapPrimitive(name)
  prim.def_impl(partial(pxla.apply_parallel_primitive, prim))
  prim.def_abstract_eval(lambda x, *args, **params: x)
  return prim


def _allreduce_split_axis_rule(prim, reducer, vals, which_mapped, axis_name):
  assert tuple(which_mapped) == (True,)
  x, = vals
  return prim.bind(reducer(x, [0]), axis_name=axis_name), False

def _allreduce_translation_rule(prim, c, val, replica_groups):
  dtype = c.GetShape(val).numpy_dtype()
  scalar = xla_bridge.Shape.array_shape(dtype, ())
  computation = xla.primitive_computation(prim, scalar, scalar)
  return c.AllReduce(val, computation, replica_groups=replica_groups)

psum_p = standard_pmap_primitive('psum')
parallel.defreducer(lax.reduce_sum_p, psum_p)
pxla.split_axis_rules[psum_p] = \
    partial(_allreduce_split_axis_rule, psum_p, lax._reduce_sum)
pxla.parallel_translation_rules[psum_p] = \
    partial(_allreduce_translation_rule, lax.add_p)
pxla.parallel_pure_rules[psum_p] = lambda x, shape: x * prod(shape)
ad.deflinear(psum_p, lambda t, axis_name: [t])


pmax_p = standard_pmap_primitive('pmax')
parallel.defreducer(lax.reduce_max_p, pmax_p)
pxla.parallel_translation_rules[pmax_p] = \
    partial(_allreduce_translation_rule, lax.max_p)
pxla.split_axis_rules[pmax_p] = \
    partial(_allreduce_split_axis_rule, pmax_p, lax._reduce_max)


pmin_p = standard_pmap_primitive('pmin')
parallel.defreducer(lax.reduce_min_p, pmin_p)
pxla.parallel_translation_rules[pmin_p] = \
    partial(_allreduce_translation_rule, lax.min_p)
pxla.split_axis_rules[pmin_p] = \
    partial(_allreduce_split_axis_rule, pmin_p, lax._reduce_min)


def _ppermute_translation_rule(c, x, replica_groups, perm):
  group_size = len(replica_groups[0])
  srcs, dsts = unzip2((src % group_size, dst % group_size) for src, dst in perm)
  if not (len(srcs) == len(set(srcs)) and len(dsts) == len(set(dsts))):
    msg = "ppermute sources and destinations must be unique, got {}."
    raise ValueError(msg.format(perm))

  full_perm = []
  for grp in replica_groups:
    grp = list(sorted(grp))
    full_perm.extend((grp[src], grp[dst]) for src, dst in perm)
  return c.CollectivePermute(x, full_perm)

def _ppermute_transpose_rule(t, perm, axis_name):
  srcs, dsts = unzip2(perm)
  inverse_perm = list(zip(dsts, srcs))
  return [ppermute(t, axis_name=axis_name, perm=inverse_perm)]

ppermute_p = standard_pmap_primitive('ppermute')
ad.deflinear(ppermute_p, _ppermute_transpose_rule)
pxla.parallel_translation_rules[ppermute_p] = _ppermute_translation_rule


def _all_to_all_translation_rule(c, x, split_axis, concat_axis, replica_groups):
  return c.AllToAll(x, split_axis, concat_axis, replica_groups)

def _all_to_all_split_axis_rule(vals, which_mapped, split_axis, concat_axis,
                                axis_name):
  assert tuple(which_mapped) == (True,)
  x, = vals
  # perform the communication to swap the hardware-mapped axes
  stacked = all_to_all_p.bind(x, split_axis=split_axis + 1, concat_axis=0,
                              axis_name=axis_name)
  # transpose the newly mapped axis to the front, newly unmapped to concat_axis
  out = _moveaxis(split_axis + 1, 0, stacked)
  out = _moveaxis(1, concat_axis + 1, out)
  return out, True

def _moveaxis(src, dst, x):
  perm = [i for i in range(x.ndim) if i != src]
  perm.insert(dst, src)
  return lax.transpose(x, perm)

all_to_all_p = standard_pmap_primitive('all_to_all')
pxla.parallel_translation_rules[all_to_all_p] = _all_to_all_translation_rule
pxla.split_axis_rules[all_to_all_p] = _all_to_all_split_axis_rule


### papply rules
# TODO(skye): it would be nice if we could put these with their corresponding
# primitives, but that currently causes circular dependencies. More refactoring
# might fix this.

def _dot_papply_rule(name, size, vals, dims):
  x, y = vals
  xdim, ydim = dims
  if xdim is None:
    return lax.dot(x, y), ydim
  elif ydim is None:
    return lax.dot(x, y), xdim
  elif ydim == 0:
    if xdim != x.ndim:
      x = psplit(x, name, x.ndim, xdim)
    x = x[..., None]
    y = y[..., None, :]
    return psum(x * y, name), None
  else:
    y = pcollect(y, name)
    return lax.dot(x, y), xdim


def _dot_general_papply_rule(name, size, vals, dims, dimension_numbers):
  x, y = vals
  xdim, ydim = dims

  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

  def adjust_dims(dims, thresh):
    return tuple(i - 1 if i > thresh else i for i in dims if i != thresh)

  sub_lhs_contract, sub_rhs_contract = lhs_contract, rhs_contract
  sub_lhs_batch, sub_rhs_batch = lhs_batch, rhs_batch

  def sub_dims(xdim, ydim):
    sub_lhs_contract, sub_rhs_contract = lhs_contract, rhs_contract
    sub_lhs_batch, sub_rhs_batch = lhs_batch, rhs_batch
    if xdim is not None:
      sub_lhs_batch = adjust_dims(lhs_batch, xdim)
      sub_lhs_contract = adjust_dims(lhs_contract, xdim)
    if ydim is not None:
      sub_rhs_batch = adjust_dims(rhs_batch, ydim)
      sub_rhs_contract = adjust_dims(rhs_contract, ydim)
    return (
      (sub_lhs_contract, sub_rhs_contract), (sub_lhs_batch, sub_rhs_batch))

  def cases(x, y, xdim, ydim, xcontract, ycontract):
    if xdim in xcontract:
      if ydim in ycontract:
        # case: both operands are split and contracting
        z = lax.dot_general(x, y, sub_dims(xdim, ydim))
        return True, (psum(z, name), None)
      elif ydim is not None:
        # case: x split and contracting, y split but not contracting
        new_ydim = ycontract[xcontract.index(xdim)]
        y = psplit(y, name, new_ydim, ydim)
        z = lax.dot_general(x, y, sub_dims(xdim, new_ydim))
        return True, (psum(z, name), None)
      else:
        # case: x split and contracting, y not split
        return False, 'one operand split and contracting, other is not split'
    else:
      return False, 'unhandled case'

  ok, out = cases(x, y, xdim, ydim, lhs_contract, rhs_contract)
  if not ok:
    ok, out = cases(y, x, ydim, xdim, rhs_contract, lhs_contract)
  if not ok:
    raise NotImplementedError(
        ('papply of dot_general, {}: '
         'xdim={}, ydim={}, dimension_numbers={}').format(
             out, xdim, ydim, dimension_numbers))
  else:
    return out


def _reshape_papply_rule(name, size, vals, axes, new_sizes, dimensions,
                         old_sizes):
  operand, = vals
  axis, = axes

  def filter_ones(xs):
    return filter(lambda x: x != 1, xs)

  def find_new_axis(old_axis, old_sizes, new_sizes):
    left = onp.prod(old_sizes[:old_axis])
    size = old_sizes[old_axis]
    prod = 1
    for i, cur_sz in enumerate(new_sizes):
      if prod == left and cur_sz == size:
        return i
      prod = prod * sz
    return None

  if dimensions is None:
    new_axis = find_new_axis(axis, old_sizes, new_sizes)
    if new_axis is not None:
      new_sizes_ = new_sizes[:new_axis] + new_sizes[new_axis + 1:]
      return lax.reshape(operand, new_sizes_, dimensions=dimensions), new_axis
    else:
      raise NotImplementedError(
          'papply of reshape that would change hidden dimension size')
  else:
    raise NotImplementedError('papply of reshape with `dimensions`')


def _transpose_papply_rule(name, size, vals, dims, permutation):
  x, = vals
  xdim, = dims
  local_perm = [i if i < xdim else i - 1 for i in permutation if i != xdim]
  return lax.transpose(x, local_perm), permutation.index(xdim)


def _select_papply_rule(name, size, vals, dims):
  dimset = {d for d in dims if d is not None}
  if len(dimset) != 1:
    raise NotImplementedError(
        'papply of select with operands split along different dimensions')
  dim, = dimset

  def drop(x, d):
    if d is None:
      return lax.dynamic_index_in_dim(x, axis_index(name), dim, False)
    else:
      return x

  return lax.select_p.bind(*map(drop, vals, dims)), dim


def _add_jaxvals_papply_rule(name, size, vals, dims):
  x, y = vals
  xdim, ydim = dims
  if xdim == ydim:
    out_dim = xdim
  elif ydim is None:
    y = lax.psplit_like(y, x, name)
    out_dim = xdim
  else:
    x = lax.psplit_like(x, y, name)
    out_dim = ydim
  return ad_util.add_jaxvals_p.bind(x, y), out_dim


parallel.papply_primitive_rules[lax.dot_p] = _dot_papply_rule
parallel.papply_primitive_rules[lax.dot_general_p] = _dot_general_papply_rule
parallel.papply_primitive_rules[lax.reshape_p] = _reshape_papply_rule
parallel.papply_primitive_rules[lax.transpose_p] = _transpose_papply_rule
parallel.papply_primitive_rules[lax.select_p] = _select_papply_rule
parallel.papply_primitive_rules[ad_util.add_jaxvals_p] = (
    _add_jaxvals_papply_rule)
