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

import collections

import numpy as onp

from jax import core
from jax import ad_util
from jax import dtypes
from jax import tree_util
from jax.lax import lax
from jax.abstract_arrays import ShapedArray, raise_to_shaped
from jax.interpreters import ad
from jax.interpreters import parallel
from jax.interpreters import xla
from jax.interpreters import pxla
from jax.util import partial, unzip2, prod
from jax.lib import xla_client

from jax.interpreters.pxla import axis_index


### parallel traceables

def psum(x, axis_name):
  """Compute an all-reduce sum on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
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
  leaves, treedef = tree_util.tree_flatten(x)
  return treedef.unflatten(psum_p.bind(*leaves, axis_name=axis_name))

def pmean(x, axis_name):
  """Compute an all-reduce mean on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce mean along the axis ``axis_name``.

  For example, with 4 XLA devices available:

  >>> x = np.arange(4)
  >>> y = jax.pmap(lambda x: jax.lax.pmean(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [ 1.5         1.5         1.5         1.5       ]
  >>> y = jax.pmap(lambda x: x / jax.lax.pmean(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [ 0.          0.66666667  1.33333334  2.0       ]
  """
  x, n = psum((x, 1), axis_name=axis_name)
  return tree_util.tree_map(lambda v: v / n, x)

def pmax(x, axis_name):
  """Compute an all-reduce max on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce max along the axis ``axis_name``.
  """
  return tree_util.tree_map(partial(pmax_p.bind, axis_name=axis_name), x)

def pmin(x, axis_name):
  """Compute an all-reduce min on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce min along the axis ``axis_name``.
  """
  return tree_util.tree_map(partial(pmin_p.bind, axis_name=axis_name), x)

def ppermute(x, axis_name, perm):
  """Perform a collective permutation according to the permutation ``perm``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  This function is an analog of the CollectivePermute XLA HLO.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).
    perm: list of pairs of ints, representing (source_index, destination_index)
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
  return tree_util.tree_map(
      partial(ppermute_p.bind, axis_name=axis_name, perm=tuple(perm)), x)

def pshuffle(x, axis_name, perm):
  """Perform a collective shuffle according to the permutation ``perm``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  This function is a simple wrapper around jax.lax.ppermute.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).
    perm: list of of ints, representing the new order of the source indices
      that encode how the mapped axis named ``axis_name`` should be
      shuffled. The integer values are treated as indices into the mapped axis
      ``axis_name``. Every int between 0 and ``len(perm)-1`` should be included.

  Returns:
    Array(s) with the same shape as ``x`` with slices along the axis
    ``axis_name`` gathered from ``x`` according to the permutation ``perm``.
  """
  if set(perm) != set(range(len(perm))):
    raise AssertionError(
      "Given `perm` does not represent a real permutation: {}".format(perm))
  return ppermute(x, axis_name, list(zip(perm, range(len(perm)))))


def pswapaxes(x, axis_name, axis):
  """Swap the pmapped axis ``axis_name`` with the unmapped axis ``axis``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  The mapped axis size must be equal to the size of the unmapped axis; that is,
  we must have ``lax.psum(1, axis_name) == x.shape[axis]``.

  This function is a special case of ``all_to_all`` where the pmapped axis of
  the input is placed at the position ``axis`` in the output. That is, it is
  equivalent to ``all_to_all(x, axis_name, axis, axis)``.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).
    axis: int indicating the unmapped axis of ``x`` to map with the name
      ``axis_name``.

  Returns:
    Array(s) with shape ``np.insert(np.delete(x.shape, axis), axis, axis_size)``
    where ``axis_size`` is the size of the mapped axis named ``axis_name`` in
    the input ``x``.
  """
  return all_to_all(x, axis_name, axis, axis)

def all_to_all(x, axis_name, split_axis, concat_axis):
  """Materialize the mapped axis and map a different axis.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  In the output, the input mapped axis ``axis_name`` is materialized at the
  logical axis position ``concat_axis``, and the input unmapped axis at position
  ``split_axis`` is mapped with the name ``axis_name``.

  The input mapped axis size must be equal to the size of the axis to be mapped;
  that is, we must have ``lax.psum(1, axis_name) == x.shape[split_axis]``.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      ``pmap`` docstring for more details).
    split_axis: int indicating the unmapped axis of ``x`` to map with the name
      ``axis_name``.
    concat_axis: int indicating the position in the output to materialize the
      mapped axis of the input with the name ``axis_name``.

  Returns:
    Array(s) with shape given by the expression::
      np.insert(np.delete(x.shape, split_axis), concat_axis, axis_size)

    where ``axis_size`` is the size of the mapped axis named ``axis_name`` in
    the input ``x``, i.e. ``axis_size = lax.psum(1, axis_name)``.
  """
  def bind(x):
    if psum(1, axis_name) != x.shape[split_axis]:
      msg = ("all_to_all requires the size of the mapped axis axis_name to "
             "equal x.shape[split_axis], but they are {} and {} respectively.")
      raise ValueError(msg.format(psum(1, axis_name), x.shape[split_axis]))
    return all_to_all_p.bind(x, split_axis=split_axis, concat_axis=concat_axis,
                             axis_name=axis_name)
  return tree_util.tree_map(bind, x)


### parallel primitives

def standard_pmap_primitive(name, multiple_results=False):
  prim = core.Primitive(name)
  prim.multiple_results = multiple_results
  prim.def_impl(partial(pxla.apply_parallel_primitive, prim))
  prim.def_abstract_eval(lambda x, *args, **params: x)
  return prim


def _allreduce_split_axis_rule(prim, reducer, vals, which_mapped, axis_name):
  assert tuple(which_mapped) == (True,)
  vals = (reducer(x, [0]) for x in vals)
  return prim.bind(*vals, axis_name=axis_name), False

def _allreduce_translation_rule(prim, c, val, replica_groups, platform=None):
  dtype = c.GetShape(val).numpy_dtype()
  scalar = ShapedArray((), dtype)
  computation = xla.primitive_subcomputation(prim, scalar, scalar)
  return c.AllReduce(val, computation, replica_groups=replica_groups)

# psum translation rule has special handling for complex dtypes
def _psum_translation_rule(c, *args, replica_groups=None, platform=None):
  if platform == "cpu":
    return _cpu_psum_translation_rule(c, *args, replica_groups=replica_groups)

  # XLA's tuple all-reduce doesn't support different dtypes in the same
  # allreduce. Instead, we perform once all-reduce for each argument input type.
  args_by_type = collections.defaultdict(lambda: ([], []))
  for i, arg in enumerate(args):
    indices, dtype_args = args_by_type[c.GetShape(arg).numpy_dtype()]
    indices.append(i)
    dtype_args.append(arg)

  # The outputs, in the original argument order.
  out = [None] * len(args)
  for dtype, (indices, dtype_args) in sorted(args_by_type.items()):
    is_complex = dtypes.issubdtype(dtype, onp.complexfloating)
    n = len(dtype_args)
    if is_complex:
      dtype_args = ([c.Real(x) for x in dtype_args] +
                    [c.Imag(x) for x in dtype_args])
    scalar = ShapedArray((), c.GetShape(dtype_args[0]).numpy_dtype())
    computation = xla.primitive_subcomputation(lax.add_p, scalar, scalar)
    all_reduce = c.AllReduce(c.Tuple(*dtype_args), computation,
                             replica_groups=replica_groups)
    if is_complex:
      xs = [c.Complex(c.GetTupleElement(all_reduce, i),
                      c.GetTupleElement(all_reduce, n + i)) for i in range(n)]
    else:
      xs = [c.GetTupleElement(all_reduce, i) for i in range(n)]
    for i, x in zip(indices, xs):
      out[i] = x
  return c.Tuple(*out)

# TODO(b/150476027): CPU doesn't support tuple all-reduce correctly. But
# fortunately we don't really need it in that case because CPU doesn't support
# cross-task communication either.
def _cpu_psum_translation_rule(c, *args, replica_groups):
  def _translate(val):
    psum = partial(_allreduce_translation_rule, lax.add_p, c,
                   replica_groups=replica_groups)
    dtype = c.GetShape(val).numpy_dtype()
    if dtypes.issubdtype(dtype, onp.complexfloating):
      return c.Complex(psum(c.Real(val)), psum(c.Imag(val)))
    else:
      return psum(val)
  return c.Tuple(*map(_translate, args))

psum_p = standard_pmap_primitive('psum', multiple_results=True)
psum_p.def_abstract_eval(lambda *args, **params: map(raise_to_shaped, args))
pxla.split_axis_rules[psum_p] = \
    partial(_allreduce_split_axis_rule, psum_p, lax._reduce_sum)
xla.parallel_translations[psum_p] = _psum_translation_rule
pxla.parallel_pure_rules[psum_p] = lambda *args, shape: (x * prod(shape) for x in args)
ad.deflinear(psum_p, lambda *ts, axis_name: psum(*ts, axis_name))
pxla.multi_host_supported_collectives.add(psum_p)


pmax_p = standard_pmap_primitive('pmax')
xla.parallel_translations[pmax_p] = \
    partial(_allreduce_translation_rule, lax.max_p)
pxla.split_axis_rules[pmax_p] = \
    partial(_allreduce_split_axis_rule, pmax_p, lax._reduce_max)


pmin_p = standard_pmap_primitive('pmin')
xla.parallel_translations[pmin_p] = \
    partial(_allreduce_translation_rule, lax.min_p)
pxla.split_axis_rules[pmin_p] = \
    partial(_allreduce_split_axis_rule, pmin_p, lax._reduce_min)


def _ppermute_translation_rule(c, x, replica_groups, perm, platform=None):
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
xla.parallel_translations[ppermute_p] = _ppermute_translation_rule
pxla.multi_host_supported_collectives.add(ppermute_p)


def _all_to_all_translation_rule(c, x, split_axis, concat_axis, replica_groups,
                                 platform=None):
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
xla.parallel_translations[all_to_all_p] = _all_to_all_translation_rule
pxla.split_axis_rules[all_to_all_p] = _all_to_all_split_axis_rule


### papply rules
# TODO(skye): it would be nice if we could put these with their corresponding
# primitives, but that currently causes circular dependencies. More refactoring
# might fix this.


def _drop(x, dim, axis_name):
  return lax.dynamic_index_in_dim(x, axis_index(axis_name), dim, False)

def _allgather(x, dim, size, axis_name):
  shape = list(x.shape)
  shape.insert(dim, size)
  out = lax.full(shape, lax._const(x, 0))
  out = lax.dynamic_update_index_in_dim(out, x, axis_index(axis_name), dim)
  return psum(out, axis_name)


def _broadcasting_papply(prim, name, size, vals, axes, **params):
  x, y = vals
  xdim, ydim = axes

  if xdim is None:
    if x.shape:
      if x.shape[ydim] == 1:
        x = x.reshape(onp.delete(x.shape, ydim))
      else:
        x = _drop(x, ydim, name)
    return prim.bind(x, y, **params), ydim
  elif ydim is None:
    if y.shape:
      if y.shape[xdim] == 1:
        y = y.reshape(onp.delete(y.shape, xdim))
      else:
        y = _drop(y, xdim, name)
    return prim.bind(x, y, **params), xdim
  elif xdim == ydim:
    return prim.bind(x, y, **params), xdim
  else:
    x_tosplit = ydim - int(xdim <= ydim)
    y_tosplit = xdim - int(ydim <= xdim)
    if y.shape[y_tosplit] == 1:
      y = _allgather(y, ydim, size, name)
      y = y.reshape(onp.delete(y.shape, xdim))
      return prim.bind(x, y, **params), ydim
    elif x.shape[x_tosplit] == 1:
      x = _allgather(x, xdim, size, name)
      x = x.reshape(onp.delete(x.shape, ydim))
      return prim.bind(x, y, **params), ydim
    else:
      x = all_to_all(x, name, x_tosplit, xdim)
      return prim.bind(x, y, **params), ydim

def _defbroadcasting(prim):
  parallel.papply_primitive_rules[prim] = partial(_broadcasting_papply, prim)


def _vectorized_papply(prim, name, size, vals, axes, **params):
  assert all(axes[0] == a for a in axes[1:])
  return prim.bind(*vals, **params), axes[0]

def _defvectorized(prim):
  parallel.papply_primitive_rules[prim] = partial(_vectorized_papply, prim)


def _reducer_papply(prim, collective, name, size, vals, papply_axes, axes, **kwargs):
  operand, = vals
  papply_axis, = papply_axes

  other_axes = [i for i in axes if i != papply_axis]
  other_axes = [i - 1 if i > papply_axis else i for i in other_axes]

  if other_axes:
    if 'input_shape' in kwargs:  # special to the reduce-sum family
      s = kwargs['input_shape']
      kwargs['input_shape'] = s[:papply_axis] + s[papply_axis + 1:]
    result = prim.bind(operand, axes=tuple(other_axes), **kwargs)
  else:
    result = operand

  if not axes or papply_axis in axes:
    return collective(result, axis_name=name), None
  else:
    new_papply_axis = papply_axis - onp.sum(onp.less(other_axes, papply_axis))
    return result, new_papply_axis

def _defreducer(prim, collective_prim):
  parallel.papply_primitive_rules[prim] = partial(_reducer_papply, prim, collective_prim)


def _identity_papply(prim, argnum, name, size, vals, axes, **params):
  return prim.bind(*vals, **params), axes[argnum]

def _defidentity(prim, argnum=0):
  parallel.papply_primitive_rules[prim] = partial(_identity_papply, prim, argnum)


_defvectorized(lax.neg_p)
_defvectorized(lax.sign_p)
_defvectorized(lax.floor_p)
_defvectorized(lax.ceil_p)
_defvectorized(lax.round_p)
_defvectorized(lax.is_finite_p)
_defvectorized(lax.exp_p)
_defvectorized(lax.log_p)
_defvectorized(lax.expm1_p)
_defvectorized(lax.log1p_p)
_defvectorized(lax.tanh_p)
_defvectorized(lax.sin_p)
_defvectorized(lax.cos_p)
_defvectorized(lax.lgamma_p)
_defvectorized(lax.digamma_p)
_defvectorized(lax.erf_p)
_defvectorized(lax.erfc_p)
_defvectorized(lax.erf_inv_p)
_defvectorized(lax.real_p)
_defvectorized(lax.imag_p)
_defvectorized(lax.conj_p)
_defvectorized(lax.abs_p)
_defvectorized(lax.sqrt_p)

_defbroadcasting(lax.atan2_p)
_defbroadcasting(lax.complex_p)
_defbroadcasting(lax.pow_p)
_defbroadcasting(lax.and_p)
_defbroadcasting(lax.or_p)
_defbroadcasting(lax.xor_p)
_defbroadcasting(lax.add_p)
_defbroadcasting(lax.sub_p)
_defbroadcasting(lax.mul_p)
_defbroadcasting(lax.safe_mul_p)
_defbroadcasting(lax.div_p)
_defbroadcasting(lax.rem_p)
_defbroadcasting(lax.max_p)
_defbroadcasting(lax.min_p)
_defbroadcasting(lax.shift_left_p)
_defbroadcasting(lax.shift_right_arithmetic_p)
_defbroadcasting(lax.shift_right_logical_p)

_defidentity(lax.tie_in_p)

_defreducer(lax.reduce_sum_p, psum)
_defreducer(lax.reduce_max_p, pmax)
_defreducer(lax.reduce_min_p, pmin)


def _dot_general_papply_rule(name, size, vals, dims, dimension_numbers,
                             precision):
  x, y = vals
  xdim, ydim = dims

  (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

  if lhs_batch or rhs_batch:
    raise NotImplementedError(
        ('papply of dot_general with batch dimensions: '
         'xdim={}, ydim={}, dimension_numbers={}').format(
             xdim, ydim, dimension_numbers))

  def adjust_dims(dims, thresh):
    return tuple(i - 1 if i > thresh else i for i in dims if i != thresh)

  def sub_dims(xdim, ydim, xcontract, ycontract, xbatch, ybatch):
    if xdim is not None:
      xbatch = adjust_dims(xbatch, xdim)
      xcontract = adjust_dims(xcontract, xdim)
    if ydim is not None:
      ybatch = adjust_dims(ybatch, ydim)
      ycontract = adjust_dims(ycontract, ydim)
    return ((xcontract, ycontract), (xbatch, ybatch))

  def cases(x, y, xdim, ydim, xc, yc, xb, yb):
    # Consider three states in which an operand may be
    #   1: split, contracting
    #   2: split, not contracting
    #   3: not split
    #
    # We will handle the following cases, marked by corresponding letter
    # symbols:
    #
    #  |1 2 3|y
    # -+-----+-
    # 1|a b c
    # 2|d e f
    # 3|g h i
    # -+
    # x|
    #
    # Case i is already covered and we can assume that it is excluded at the
    # outset, since a papply rule is not invoked when no operands are split.

    if xdim in xc:
      # cases a, b, c
      if ydim in yc:
        # case a: both operands are split and contracting
        # TODO(frostig): Might the following work?
        # z = lax.dot_general(
        #     x, y, sub_dims(xdim, ydim, xc, yc, xb, yb), precision)
        # return True, (psum(z, name), None)
        return False, 'both operands split and contracting'
      elif ydim is not None:
        # case b: x split and contracting, y split but not contracting
        # TODO(frostig): Might the following work?
        # new_ydim = yc[xc.index(xdim)]
        # y = all_to_all(y, name, new_ydim, ydim)
        # z = lax.dot_general(
        #     x, y, sub_dims(xdim, new_ydim, xc, yc, xb, yb), precision)
        # return True, (psum(z, name), None)
        return False, 'rhs split but not contracting, lhs split and contracting'
      else:
        # case c: x split and contracting, y not split
        assert ydim is None
        return False, 'one operand split and contracting, other is not split'
    elif xdim is not None:
      # cases d, e, f
      if ydim in yc:
        # case d: x split but not contracting, y split and contracting
        # TODO(frostig): Might the following work?
        # new_xdim = xc[yc.index(ydim)]
        # x = all_to_all(x, name, new_xdim, xdim)
        # z = lax.dot_general(
        #     x, y, sub_dims(new_xdim, ydim, xc, yc, xb, yb), precision)
        # return True, (psum(z, name), None)
        return False, 'lhs split but not contracting, rhs split and contracting'
      elif ydim is not None:
        # case e: both operands are split but not contracting
        y = _allgather(y, ydim, size, name)
        z = lax.dot_general(
            x, y, sub_dims(xdim, None, xc, yc, xb, yb), precision)
        zdim = xdim + len(xb) - len([d for d in range(xdim) if d in xc])
        return True, (z, zdim)
      else:
        # case f: x split but not contracting, y not split
        assert ydim is None
        z = lax.dot_general(
            x, y, sub_dims(xdim, None, xc, yc, xb, yb), precision)
        zdim = xdim + len(xb) - len([d for d in range(xdim) if d in xc])
        return True, (z, zdim)
    else:
      # cases g, h
      assert xdim is None
      if ydim in yc:
        # case g: x not split, y split and contracting
        return False, 'one operand split and contracting, other is not split'
      else:
        # case h: x not split, y split but not contracting
        assert ydim is not None
        # TODO(frostig): Might the following work?
        # z = lax.dot_general(
        #     x, y, sub_dims(None, ydim, xc, yc, xb, yb), precision)
        # zdim = (
        #     ydim + len(xb) +                # batch dimensions
        #     x.ndim - len(xc) -              # non-contracting x dimensions
        #     len([d for d in range(ydim) if d in yc]))
        # return True, (z, zdim)
        return False, 'lhs not split, rhs split but not contracting'

    assert False, 'unreachable'

  ok, out = cases(
      x, y, xdim, ydim, lhs_contract, rhs_contract, lhs_batch, rhs_batch)
  if ok:
    return out
  else:
    raise NotImplementedError(
        ('papply of dot_general, {}: '
         'xdim={}, ydim={}, dimension_numbers={}').format(
             out, xdim, ydim, dimension_numbers))


def _reshape_papply_rule(name, size, vals, axes, new_sizes, dimensions):
  operand, = vals
  axis, = axes
  old_sizes = tuple(onp.insert(operand.shape, axis, size))

  def filter_ones(xs):
    return filter(lambda x: x != 1, xs)

  def find_new_axis(old_axis, old_sizes, new_sizes):
    left = onp.prod(old_sizes[:old_axis])
    size = old_sizes[old_axis]
    prod = 1
    for i, cur_sz in enumerate(new_sizes):
      if prod == left and cur_sz == size:
        return i
      prod = prod * cur_sz
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
    return _drop(x, dim, name) if d is None else x
  return lax.select_p.bind(*map(drop, vals, dims)), dim


def _add_jaxvals_papply_rule(name, size, vals, dims):
  x, y = vals
  xdim, ydim = dims
  if xdim == ydim:
    out_dim = xdim
  else:
    raise NotImplementedError
  # elif ydim is None:
  #   y = lax.psplit_like(y, x, name)
  #   out_dim = xdim
  # else:
  #   x = lax.psplit_like(x, y, name)
  #   out_dim = ydim
  return ad_util.add_jaxvals_p.bind(x, y), out_dim


def _convert_element_type_papply_rule(
    name, size, vals, dims, new_dtype, **params):
  operand, = vals
  dim, = dims
  return lax.convert_element_type(operand, new_dtype), dim


def _conv_general_dilated_papply_rule(
    name, size, vals, dims, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers, feature_group_count, precision, **unused_kwargs):
  lhs, rhs = vals
  lhs_dim, rhs_dim = dims
  lhs_spec_batch_dim = dimension_numbers.lhs_spec[0]
  if rhs_dim is None and lhs_dim == lhs_spec_batch_dim:
    lhs = lax.reshape(lhs, tuple(onp.insert(lhs.shape, lhs_dim, 1)))
    out = lax.conv_general_dilated(
        lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
        dimension_numbers, feature_group_count, precision)
    return out, lhs_dim
  else:
    raise NotImplementedError(
        "splitting a convolution along anything but input batch dimension")


def _broadcast_in_dim_papply_rule(name, size, vals, dims, shape,
                                  broadcast_dimensions):
  operand, = vals
  dim, = dims
  out_dim = broadcast_dimensions[dim]
  if shape[out_dim] != shape[dim]:
    raise ValueError(
        "broadcast_in_dim changes hidden dimension size: {} to {}".format(
            shape[dim], shape[out_dim]))
  sub_bdims = tuple(onp.delete(broadcast_dimensions, dim))
  sub_shape = tuple(onp.delete(shape, out_dim))
  return lax.broadcast_in_dim(operand, sub_shape, sub_bdims), out_dim


def _pad_papply_rule(name, size, vals, dims, padding_config):
  operand, padding_value = vals
  operand_dim, padding_value_dim = dims
  assert padding_value_dim is None
  padding_config = list(padding_config)
  if padding_config[operand_dim] == (0, 0, 0):
    padded = lax.pad(
        operand,
        padding_value,
        padding_config[:operand_dim] + padding_config[operand_dim + 1:])
    return padded, operand_dim
  else:
    raise NotImplementedError(
        'pad changes size of hidden dimension {} with config {}'.format(
            operand_dim, padding_config))


def _slice_papply_rule(name, size, vals, dims, start_indices, limit_indices,
                       strides, **kwargs):
  operand, = vals
  dim, = dims
  start_indices = list(start_indices)
  limit_indices = list(limit_indices)

  if (start_indices[dim] != 0 or
      limit_indices[dim] != size or
      strides is not None and strides[dim] != 1):
    raise NotImplementedError('slice changes side of hidden dimension')

  out = lax.slice(
      operand,
      start_indices[:dim] + start_indices[dim + 1:],
      limit_indices[:dim] + limit_indices[dim + 1:],
      strides[:dim] + strides[dim + 1:] if strides is not None else None)
  return out, dim


def _gather_papply_rule(
    name, size, vals, dims, dimension_numbers, slice_sizes, operand_shape):
  operand, start_indices = vals
  operand_dim, start_indices_dim = dims
  if (operand_dim is None and
      start_indices_dim is not None and
      start_indices_dim not in dimension_numbers.offset_dims and
      dimension_numbers.collapsed_slice_dims == (0,)):
    offset_dims = tuple(i - 1 if i > start_indices_dim else i
                        for i in dimension_numbers.offset_dims)
    dnums = lax.GatherDimensionNumbers(
        offset_dims=offset_dims,
        collapsed_slice_dims=dimension_numbers.collapsed_slice_dims,
        start_index_map=dimension_numbers.start_index_map)
    out = lax.gather(operand, start_indices, dimension_numbers=dnums,
                     slice_sizes=slice_sizes)
    out_dim = start_indices_dim + onp.sum(
        onp.less_equal(offset_dims, start_indices_dim))
    return out, out_dim
  else:
    raise NotImplementedError


parallel.papply_primitive_rules[lax.dot_general_p] = _dot_general_papply_rule
parallel.papply_primitive_rules[lax.reshape_p] = _reshape_papply_rule
parallel.papply_primitive_rules[lax.transpose_p] = _transpose_papply_rule
parallel.papply_primitive_rules[lax.select_p] = _select_papply_rule
parallel.papply_primitive_rules[ad_util.add_jaxvals_p] = \
    _add_jaxvals_papply_rule
parallel.papply_primitive_rules[lax.convert_element_type_p] = \
    _convert_element_type_papply_rule
parallel.papply_primitive_rules[lax.conv_general_dilated_p] = \
    _conv_general_dilated_papply_rule
parallel.papply_primitive_rules[lax.broadcast_in_dim_p] = \
    _broadcast_in_dim_papply_rule
parallel.papply_primitive_rules[lax.pad_p] = _pad_papply_rule
parallel.papply_primitive_rules[lax.slice_p] = _slice_papply_rule
parallel.papply_primitive_rules[lax.gather_p] = _gather_papply_rule
