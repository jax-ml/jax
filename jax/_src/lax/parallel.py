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
import warnings

import numpy as np

from jax import core
from jax import dtypes
from jax import tree_util
from jax._src import source_info_util
from . import lax
from jax.core import ShapedArray, raise_to_shaped
from jax.interpreters import ad
from jax.interpreters import xla
from jax.interpreters import pxla
from jax.interpreters import batching
from jax.interpreters import partial_eval as pe
from jax._src.util import partial, unzip2, prod
from jax.lib import xla_client as xc
from jax.lib import xla_bridge as xb
from jax.config import config
from jax._src.numpy import lax_numpy

xops = xc.ops


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
      once, and all groups must be the same size.


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
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  _validate_axis_index_groups(axis_index_groups)
  leaves, treedef = tree_util.tree_flatten(x)
  leaves = [lax.convert_element_type(l, np.int32)
            if dtypes.dtype(l) == np.bool_ else l for l in leaves]
  out_flat = psum_p.bind(*leaves, axis_name=axis_name,
                         axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, out_flat)

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
      once, and all groups must be the same size.

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
  x = psum(x, axis_name=axis_name, axis_index_groups=axis_index_groups)
  n = psum(1, axis_name=axis_name, axis_index_groups=axis_index_groups)
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
      once, and all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce max along the axis ``axis_name``.
  """
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  _validate_axis_index_groups(axis_index_groups)
  leaves, treedef = tree_util.tree_flatten(x)
  out_flat = pmax_p.bind(*leaves, axis_name=axis_name,
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
      once, and all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce min along the axis ``axis_name``.
  """
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  _validate_axis_index_groups(axis_index_groups)
  leaves, treedef = tree_util.tree_flatten(x)
  out_flat = pmin_p.bind(*leaves, axis_name=axis_name,
                         axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, out_flat)

def _validate_axis_index_groups(axis_index_groups):
  if axis_index_groups is None:
    return
  len_0 = len(axis_index_groups[0])
  if any(len(g) != len_0 for g in axis_index_groups):
    raise ValueError("axis_index_groups must all be the same size")
  axis_space = range(len_0 * len(axis_index_groups))
  if {i for g in axis_index_groups for i in g} != set(axis_space):
    raise ValueError("axis_index_groups must cover all indices exactly once")

def ppermute(x, axis_name, perm):
  """Perform a collective permutation according to the permutation ``perm``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  This function is an analog of the CollectivePermute XLA HLO.

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
  return tree_util.tree_map(
      partial(ppermute_p.bind, axis_name=axis_name, perm=tuple(perm)), x)

def pshuffle(x, axis_name, perm):
  """Convenience wrapper of jax.lax.ppermute with alternate permutation encoding

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    perm: list of of ints encoding sources for the permutation to be applied to
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

def all_to_all(x, axis_name, split_axis, concat_axis, *, axis_index_groups=None):
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

  Returns:
    Array(s) with shape given by the expression::

      np.insert(np.delete(x.shape, split_axis), concat_axis, axis_size)

    where ``axis_size`` is the size of the mapped axis named ``axis_name`` in
    the input ``x``, i.e. ``axis_size = lax.psum(1, axis_name)``.
  """
  def bind(x):
    group_size = psum(1, axis_name, axis_index_groups=axis_index_groups)
    if group_size != x.shape[split_axis]:
      msg = ("all_to_all requires the size of the mapped axis axis_name to "
             "equal x.shape[split_axis], but they are {} and {} respectively.")
      raise ValueError(msg.format(group_size, x.shape[split_axis]))
    return all_to_all_p.bind(x, split_axis=split_axis, concat_axis=concat_axis,
                             axis_name=axis_name,
                             axis_index_groups=axis_index_groups)

  return tree_util.tree_map(bind, x)

def axis_index(axis_name):
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
  >>> f(np.zeros(4))
  ShardedDeviceArray([0, 1, 2, 3], dtype=int32)
  >>> f(np.zeros(8))
  ShardedDeviceArray([0, 1, 2, 3, 4, 5, 6, 7], dtype=int32)
  >>> @partial(jax.pmap, axis_name='i')
  ... @partial(jax.pmap, axis_name='j')
  ... def f(_):
  ...   return lax.axis_index('i'), lax.axis_index('j')
  ...
  >>> x, y = f(np.zeros((4, 2)))
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
  return axis_index_p.bind(axis_name=axis_name)


def pdot(x, y, axis_name, pos_contract=((), ())):
  if not isinstance(axis_name, (list, tuple)):
    axis_name = (axis_name,)
  return pdot_p.bind(x, y, axis_name=axis_name,
                     pos_contract=pos_contract, pos_batch=[(), ()])


### parallel primitives

def _allreduce_soft_pmap_rule(prim, reducer, vals, mapped, chunk_size,
                              *, axis_name, axis_index_groups):
  if axis_index_groups is not None:
    raise NotImplementedError("soft_pmap does not yet support axis_index_groups")
  reduced_vals = [reducer(x, [0]) if m else x for x, m in zip(vals, mapped)]
  outs = prim.bind(*reduced_vals, axis_name=axis_name,
                   axis_index_groups=axis_index_groups)
  return outs, (False,) * len(vals)

# This is only used for collectives that do not include the vmapped axis name,
# which is why the rule is so simple.
def _collective_batcher(prim, args, dims, **params):
  return prim.bind(*args, **params), dims if prim.multiple_results else dims[0]

def _batched_reduction_collective(
    prim, if_mapped, if_unmapped, frame, vals_in, dims_in, axis_name,
    axis_index_groups):
  assert prim.multiple_results
  assert frame.name in axis_name
  if axis_index_groups is not None:
    raise NotImplementedError("axis_index_groups not supported in vmap collectives. "
                              "Please open a feature request!")
  vals_out = [if_mapped(v, d) if d is not batching.not_mapped
              else if_unmapped(v, frame.size) for v, d in zip(vals_in, dims_in)]
  if len(axis_name) > 1:
    remaining_axis_names = tuple(n for n in axis_name if n != frame.name)
    vals_out = prim.bind(*vals_out, axis_name=remaining_axis_names,
                         axis_index_groups=None)
  return vals_out, [batching.not_mapped] * len(vals_out)

def _replica_groups(axis_env, axis_name, axis_index_groups):
  replica_groups = xla.axis_groups(axis_env, axis_name)
  if axis_index_groups is not None:
    replica_groups = [[axis_group[i] for i in axis_index_group]
                      for axis_group in replica_groups
                      for axis_index_group in axis_index_groups]
  return replica_groups

def _allreduce_translation_rule(prim, c, *args, axis_name, axis_index_groups,
                                axis_env, platform):
  if platform in ("cpu", "tpu"):
    return _notuple_allreduce_translation_rule(
        prim, c, *args, axis_name=axis_name,
        axis_index_groups=axis_index_groups, axis_env=axis_env,
        platform=platform)

  # XLA's tuple all-reduce doesn't support different dtypes in the same
  # allreduce. Instead, we perform once all-reduce for each argument input type.
  args_by_type = collections.defaultdict(lambda: ([], []))
  for i, arg in enumerate(args):
    indices, dtype_args = args_by_type[c.get_shape(arg).numpy_dtype()]
    indices.append(i)
    dtype_args.append(arg)

  # The outputs, in the original argument order.
  out = [None] * len(args)
  replica_groups = _replica_groups(axis_env, axis_name, axis_index_groups)
  replica_groups_protos = xc.make_replica_groups(replica_groups)
  for dtype, (indices, dtype_args) in sorted(args_by_type.items()):
    is_complex = dtypes.issubdtype(dtype, np.complexfloating)
    n = len(dtype_args)
    if is_complex and prim is lax.add_p:
      # TODO(b/141575627): we handle complex-dtype sum-reduction directly as a
      # special case because it's not currently handled by XLA:GPU
      dtype_args = ([xops.Real(x) for x in dtype_args] +
                    [xops.Imag(x) for x in dtype_args])
    scalar = ShapedArray((), c.get_shape(dtype_args[0]).numpy_dtype())
    computation = xla.primitive_subcomputation(prim, scalar, scalar)
    all_reduce = xops.AllReduce(xops.Tuple(c, dtype_args), computation,
                                replica_groups_protos, None, None)
    if is_complex and prim is lax.add_p:
      xs = [xops.Complex(xops.GetTupleElement(all_reduce, i),
                         xops.GetTupleElement(all_reduce, n + i)) for i in range(n)]
    else:
      xs = [xops.GetTupleElement(all_reduce, i) for i in range(n)]
    for i, x in zip(indices, xs):
      out[i] = x
  return xops.Tuple(c, out)

# TODO(b/155446630): An XLA:TPU optimization pass also doesn't support
# tuple all-reduce yet. Meanwhile, rely on deterministic compiler behavior.
def _notuple_allreduce_translation_rule(prim, c, *args, axis_name, axis_env,
                                        axis_index_groups, platform):
  def all_reduce(x):
    replica_groups_protos = xc.make_replica_groups(
        _replica_groups(axis_env, axis_name, axis_index_groups))
    scalar = ShapedArray((), c.get_shape(x).numpy_dtype())
    computation = xla.primitive_subcomputation(prim, scalar, scalar)
    return xops.AllReduce(x, computation, replica_groups_protos, None, None)

  if prim is not lax.add_p:
    outs = [all_reduce(x) for x in args]
  else:
    # TODO(b/141575627): we handle complex-dtype sum-reduction directly as a
    # special case because it's not currently handled by XLA:GPU
    outs = [xops.Complex(all_reduce(xops.Real(x)), all_reduce(xops.Imag(x)))
            if dtypes.issubdtype(c.get_shape(x).numpy_dtype(), np.complexfloating)
            else all_reduce(x) for x in args]
  return xops.Tuple(c, outs)

def _psum_transpose_rule(cts, *args, axis_name, axis_index_groups):
  nonzero_out_cts, treedef = tree_util.tree_flatten(cts)
  nonzero_in_cts = psum_p.bind(*nonzero_out_cts, axis_name=axis_name,
                               axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, nonzero_in_cts)

psum_p = core.Primitive('psum')
psum_p.multiple_results = True
psum_p.def_abstract_eval(lambda *args, **params: map(raise_to_shaped, args))
pxla.soft_pmap_rules[psum_p] = \
    partial(_allreduce_soft_pmap_rule, psum_p, lax._reduce_sum)
xla.parallel_translations[psum_p] = partial(_allreduce_translation_rule, lax.add_p)  # type: ignore
ad.deflinear2(psum_p, _psum_transpose_rule)
pxla.multi_host_supported_collectives.add(psum_p)
batching.primitive_batchers[psum_p] = partial(_collective_batcher, psum_p)
batching.collective_rules[psum_p] = \
  partial(_batched_reduction_collective,
          psum_p,
          lambda v, d: v.sum(d),
          lambda v, axis_size: axis_size * v)

# We set a special bind rule for psum so that psum(1, 'i') can be evaluated at
# tracing time.
@psum_p.def_custom_bind
def psum_bind(*args, axis_name, axis_index_groups):
  if all(not isinstance(x, core.Tracer) for x in args):
    if axis_index_groups is not None:
      size = len(axis_index_groups[0])
    elif isinstance(axis_name, (list, tuple)):
      size = prod([core.axis_frame(name).size for name in axis_name])  # type: ignore
    else:
      size = core.axis_frame(axis_name).size  # type: ignore
    return tuple(size * x for x in args)
  return core.Primitive.bind(
      psum_p, *args, axis_name=axis_name, axis_index_groups=axis_index_groups)


pmax_p = core.Primitive('pmax')
pmax_p.multiple_results = True
pmax_p.def_abstract_eval(lambda *args, **params: map(raise_to_shaped, args))
xla.parallel_translations[pmax_p] = partial(_allreduce_translation_rule, lax.max_p)
pxla.multi_host_supported_collectives.add(pmax_p)
batching.primitive_batchers[pmax_p] = partial(_collective_batcher, pmax_p)
batching.collective_rules[pmax_p] = \
  partial(_batched_reduction_collective, pmax_p,
          lambda v, d: v.max(d), lambda v, axis_size: v)


pmin_p = core.Primitive('pmin')
pmin_p.multiple_results = True
pmin_p.def_abstract_eval(lambda *args, **params: map(raise_to_shaped, args))
xla.parallel_translations[pmin_p] = partial(_allreduce_translation_rule, lax.min_p)
pxla.multi_host_supported_collectives.add(pmin_p)
batching.primitive_batchers[pmin_p] = partial(_collective_batcher, pmin_p)
batching.collective_rules[pmin_p] = \
  partial(_batched_reduction_collective, pmin_p,
          lambda v, d: v.min(d), lambda v, axis_size: v)


def _ppermute_translation_rule(c, x, *, axis_name, axis_env, perm, platform):
  replica_groups = _replica_groups(axis_env, axis_name, None)
  group_size = len(replica_groups[0])
  srcs, dsts = unzip2((src % group_size, dst % group_size) for src, dst in perm)
  if not (len(srcs) == len(set(srcs)) and len(dsts) == len(set(dsts))):
    msg = "ppermute sources and destinations must be unique, got {}."
    raise ValueError(msg.format(perm))

  full_perm = []
  for grp in replica_groups:
    grp = list(sorted(grp))
    full_perm.extend((grp[src], grp[dst]) for src, dst in perm)
  return xops.CollectivePermute(x, full_perm)

def _ppermute_transpose_rule(t, x, perm, axis_name):
  srcs, dsts = unzip2(perm)
  inverse_perm = list(zip(dsts, srcs))
  return [ppermute(t, axis_name=axis_name, perm=inverse_perm)]

def _ppermute_batcher(frame, vals_in, dims_in, axis_name, perm):
  assert len(perm) == frame.size, "Permutation doesn't match the axis size!"
  assert axis_name == frame.name, "ppermute batcher called with wrong axis name"
  (v,), (d,) = vals_in, dims_in
  assert d is not batching.not_mapped
  perm_indices = [None] * frame.size
  for src, dst in perm:
    perm_indices[src] = dst
  return lax_numpy.take(v, perm_indices, d), d

ppermute_p = core.Primitive('ppermute')
ppermute_p.def_abstract_eval(lambda x, **params: raise_to_shaped(x))
ad.deflinear2(ppermute_p, _ppermute_transpose_rule)
xla.parallel_translations[ppermute_p] = _ppermute_translation_rule
pxla.multi_host_supported_collectives.add(ppermute_p)
batching.primitive_batchers[ppermute_p] = partial(_collective_batcher, ppermute_p)
batching.collective_rules[ppermute_p] = _ppermute_batcher


def _moveaxis(src, dst, x):
  perm = [i for i in range(x.ndim) if i != src]
  perm.insert(dst, src)
  return lax.transpose(x, perm)

def _all_to_all_via_all_gather(x, *, axis_name, split_axis, concat_axis, axis_index_groups):
  global_full = all_gather(x, axis_name, axis_index_groups=axis_index_groups)
  idx = axis_index(axis_name)
  if axis_index_groups:
    idx = idx % len(axis_index_groups[0])
  local_slice = lax.dynamic_index_in_dim(global_full, idx, split_axis + 1, keepdims=False)
  return _moveaxis(0, concat_axis, local_slice)

def _all_to_all_translation_rule(c, x, *, split_axis, concat_axis, axis_name,
                                 axis_index_groups, axis_env, platform):
  # Workaround for AllToAll not being implemented on CPU.
  replica_groups = _replica_groups(axis_env, axis_name, axis_index_groups)
  if len(replica_groups[0]) == 1:
    return x
  elif platform != 'tpu':
    warnings.warn("all_to_all (and pswapaxes) are only implemented properly for TPUs. All other "
                  "backends emulate it using a very slow and memory intensive algorithm, so expect "
                  "significant slowdowns.")
    lowering = xla.lower_fun(_all_to_all_via_all_gather, multiple_results=False, parallel=True)
    return lowering(c, x,
                    split_axis=split_axis, concat_axis=concat_axis, axis_name=axis_name,
                    axis_index_groups=axis_index_groups, axis_env=axis_env, platform=platform)
  else:
    split_count = len(replica_groups[0])
    if not all(split_count == len(g) for g in replica_groups):
      raise ValueError('Replica groups must be equally sized')
    replica_groups_protos = xc.make_replica_groups(replica_groups)
    if concat_axis == split_axis:
      return xops.AllToAll(x, split_axis, concat_axis, split_count,
                           replica_groups_protos)
    else:
      if concat_axis < split_axis:
        split_axis += 1
      elif split_axis < concat_axis:
        concat_axis += 1
      x = xla.lower_fun(partial(lax.expand_dims, dimensions=(concat_axis,)), multiple_results=False)(c, x)
      x = xops.AllToAll(x, split_axis, concat_axis, split_count, replica_groups_protos)
      x = xla.lower_fun(partial(lax.squeeze, dimensions=(split_axis,)), multiple_results=False)(c, x)
      return x

def _all_to_all_transpose_rule(cts, x, axis_name, split_axis, concat_axis, axis_index_groups):
  return (all_to_all(
      cts,
      axis_name=axis_name,
      split_axis=concat_axis,
      concat_axis=split_axis,
      axis_index_groups=axis_index_groups),)


def _all_to_all_batcher(vals_in, dims_in, *, axis_name, split_axis, concat_axis, axis_index_groups):
  x, = vals_in
  d, = dims_in
  if d <= split_axis:
    split_axis += 1
  if d <= concat_axis:
    concat_axis += 1
  # Note: At this point split_axis and concat_axis are adjusted to the extra
  #       dimension and we have d != split_axis and d != concat_axis.
  if split_axis < d < concat_axis:
    d -= 1
  elif concat_axis < d < split_axis:
    d += 1
  result = all_to_all_p.bind(
      x,
      axis_name=axis_name,
      split_axis=split_axis,
      concat_axis=concat_axis,
      axis_index_groups=axis_index_groups)
  return result, d

def _all_to_all_batched_collective(frame, vals_in, dims_in,
                                   axis_name, split_axis, concat_axis,
                                   axis_index_groups):
  if isinstance(axis_name, (list, tuple)) and len(axis_name) > 1:
    raise NotImplementedError("update after #4835")  # TODO(mattjj,apaszke)
  x, = vals_in
  d, = dims_in
  split_axis_adj = split_axis + (1 if d <= split_axis else 0)
  concat_axis_adj = concat_axis + (1 if split_axis_adj <= concat_axis else 0)
  if d < split_axis_adj < concat_axis_adj:
    split_axis_adj -= 1
  elif concat_axis_adj < split_axis_adj < d:
    split_axis_adj += 1
  return _moveaxis(d, concat_axis_adj, x), split_axis_adj

def _all_to_all_abstract_eval(x, axis_name, split_axis, concat_axis, axis_index_groups):
  input_aval = raise_to_shaped(x)
  shape = list(input_aval.shape)
  size = shape.pop(split_axis)
  shape.insert(concat_axis, size)
  return ShapedArray(tuple(shape), input_aval.dtype, weak_type=False)

all_to_all_p = core.Primitive('all_to_all')
all_to_all_p.def_abstract_eval(_all_to_all_abstract_eval)
xla.parallel_translations[all_to_all_p] = _all_to_all_translation_rule
ad.deflinear2(all_to_all_p, _all_to_all_transpose_rule)
pxla.multi_host_supported_collectives.add(all_to_all_p)
batching.primitive_batchers[all_to_all_p] = _all_to_all_batcher
batching.collective_rules[all_to_all_p] = _all_to_all_batched_collective


def _expand(dim, size, index, x):
  shape = list(x.shape)
  shape.insert(dim, size)
  out = lax.full(shape, lax._const(x, 0))
  return lax.dynamic_update_index_in_dim(out, x, index, dim)

def _allgather(x, dim, size, index, axis_name, axis_index_groups=None):
  outs = tree_util.tree_map(partial(_expand, dim, size, index), x)
  return psum(outs, axis_name, axis_index_groups=axis_index_groups)

def all_gather(x, axis_name, *, axis_index_groups=None):
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

  Returns:
    Array(s) representing the result of an all-gather along the axis
    ``axis_name``. Shapes are the same as ``x.shape``, but with a leading
    dimension of the axis_size.

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
  [[ 0.  1.  2.  3.]
   [ 4.  5.  6.  7.]
   [ 8.  9. 10. 11.]
   [12. 13. 14. 15.]]
  >>> y = jax.pmap(lambda x: jax.lax.all_gather(
  ... x, 'i', axis_index_groups=[[0, 2], [3, 1]]))(x)
  >>> print(y)
  [[[ 0.  1.  2.  3.]
    [ 8.  9. 10. 11.]]
   [[12. 13. 14. 15.]
    [ 4.  5.  6.  7.]]
   [[ 0.  1.  2.  3.]
    [ 8.  9. 10. 11.]]
   [[12. 13. 14. 15.]
    [ 4.  5.  6.  7.]]
  """

  index = axis_index(axis_name)
  if axis_index_groups is not None:
    indices = np.array(axis_index_groups).flatten()
    axis_index_to_group_index = indices.argsort() % len(axis_index_groups[0])
    index = lax_numpy.array(axis_index_to_group_index)[index]

  axis_size = psum(1, axis_name, axis_index_groups=axis_index_groups)

  return _allgather(x, 0, axis_size, index, axis_name, axis_index_groups)


def _axis_index_translation_rule(c, *, axis_name, axis_env, platform):
  axis_pos = list(axis_env.names).index(axis_name)
  nreplicas = axis_env.nreps // prod(axis_env.sizes)
  div = xb.constant(c, np.array(nreplicas * prod(axis_env.sizes[axis_pos+1:]),
                                dtype=np.uint32))
  mod = xb.constant(c, np.array(axis_env.sizes[axis_pos], dtype=np.uint32))
  unsigned_index = xops.Rem(xops.Div(xops.ReplicaId(c), div), mod)
  return xops.ConvertElementType(unsigned_index, xb.dtype_to_etype(np.int32))

def _axis_index_soft_pmap_rule(vals, mapped, chunk_size, *, axis_name):
  assert not vals and not mapped
  idx = axis_index(axis_name)  # type: ignore
  return idx * chunk_size + np.arange(chunk_size, dtype=np.int32), True

axis_index_p = core.Primitive('axis_index')
xla.parallel_translations[axis_index_p] = _axis_index_translation_rule
pxla.soft_pmap_rules[axis_index_p] = _axis_index_soft_pmap_rule  # type: ignore
axis_index_p.def_abstract_eval(
    lambda *args, **params: ShapedArray((), np.int32))
pxla.multi_host_supported_collectives.add(axis_index_p)

# Axis index doesn't get any arguments, so that the default bind would have no
# way to call into a data-dependency based trace such as vmap. Each trace that
# wants to bind an axis name has to additionally implement `process_axis_index`
# and put its main trace on the axis env stack.
def _axis_index_bind(*, axis_name):
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  inner_size = 1
  index = 0
  for name in reversed(axis_name):
    frame = core.axis_frame(name)
    if frame.main_trace is not None:
      trace = frame.main_trace.with_cur_sublevel()
      name_idx = trace.process_axis_index(frame)
    else:
      name_idx = core.Primitive.bind(axis_index_p, axis_name=name)
    index += name_idx * inner_size
    inner_size *= psum(1, name)
  return index
axis_index_p.def_custom_bind(_axis_index_bind)

def _process_axis_index(self, frame):
  return batching.BatchTracer(self, lax_numpy.arange(frame.size, dtype=np.int32), 0)
batching.BatchTrace.process_axis_index = _process_axis_index  # type: ignore


pdot_p = core.Primitive('pdot')

@pdot_p.def_impl
def _pdot_impl(x, y, *, axis_name, pos_contract, pos_batch):
  if axis_name: raise NameError(f"unbound axis name: {axis_name[0]}")
  return lax.dot_general(x, y, [pos_contract, pos_batch])

@pdot_p.def_abstract_eval
def _pdot_abstract_eval(x, y, *, axis_name, pos_contract, pos_batch):
  # TODO: avals with names, check inputs are mapped along axis_name, eliminate
  return lax.dot_general_p.abstract_eval(
      x, y, dimension_numbers=[pos_contract, pos_batch],
      precision=None)

def _pdot_vmap_collective_rule(frame, vals_in, dims_in, *, axis_name,
                               pos_contract, pos_batch):
  x, y = vals_in
  x_dim, y_dim = dims_in
  x_pos_contract, y_pos_contract = pos_contract
  x_pos_contract = [x_dim] + [d + (d >= x_dim) for d in x_pos_contract]
  y_pos_contract = [y_dim] + [d + (d >= y_dim) for d in y_pos_contract]
  x_pos_batch, y_pos_batch = pos_batch
  x_pos_batch = [d + (d >= x_dim) for d in x_pos_batch]
  y_pos_batch = [d + (d >= y_dim) for d in y_pos_batch]
  remaining_axis_names = tuple(n for n in axis_name if n != frame.name)
  out = pdot_p.bind(x, y, axis_name=remaining_axis_names,
                    pos_contract=[x_pos_contract, y_pos_contract],
                    pos_batch=[x_pos_batch, y_pos_batch])
  return out, None
batching.collective_rules[pdot_p] = _pdot_vmap_collective_rule

def _pdot_vmap_batching_rule(vals_in, dims_in, *, axis_name, pos_contract,
                             pos_batch):
  x, y = vals_in
  (pos_contract, pos_batch), result_batch_dim = lax._dot_general_batch_dim_nums(
      (x.ndim, y.ndim), dims_in, [pos_contract, pos_batch])
  out = pdot_p.bind(x, y, axis_name=axis_name, pos_contract=pos_contract,
                    pos_batch=pos_batch)
  return out, result_batch_dim
batching.primitive_batchers[pdot_p] = _pdot_vmap_batching_rule

def _pdot_translation_rule(c, x, y, *, axis_name, pos_contract, pos_batch,
                           axis_env, platform):
  assert axis_name
  local_out = lax._dot_general_translation_rule(
      c, x, y, dimension_numbers=[pos_contract, pos_batch], precision=None)
  out_tup = xla.parallel_translations[psum_p](
      c, local_out, axis_name=axis_name, axis_index_groups=None,
      axis_env=axis_env, platform=platform)
  out, = xla.xla_destructure(c, out_tup)
  return out
xla.parallel_translations[pdot_p] = _pdot_translation_rule

def _pdot_transpose_lhs(g, y, *, axis_name, pos_contract, pos_batch):
  # TODO: avals with names, call pbroadcast with axis_name
  return lax._dot_general_transpose_lhs(
      g, y, dimension_numbers=[pos_contract, pos_batch], precision=None)
def _pdot_transpose_rhs(g, x, *, axis_name, pos_contract, pos_batch):
  # TODO: avals with names, call pbroadcast with axis_name
  return lax._dot_general_transpose_rhs(
      g, x, dimension_numbers=[pos_contract, pos_batch], precision=None)
ad.defbilinear(pdot_p, _pdot_transpose_lhs, _pdot_transpose_rhs)

pxla.multi_host_supported_collectives.add(pdot_p)


@config.register_omnistaging_disabler
def omnistaging_disabler() -> None:
  global axis_index

  psum_p.bind = partial(core.Primitive.bind, psum_p)  # type: ignore
  psum_p.def_impl(partial(pxla.apply_parallel_primitive, psum_p))  # type: ignore
  pxla.parallel_pure_rules[psum_p] = lambda *args, shape: (x * prod(shape) for x in args)  # type: ignore

  def _axis_index_bind(*, axis_name):
    dynamic_axis_env = pxla._thread_local_state.dynamic_axis_env
    frame = dynamic_axis_env[axis_name]
    sizes = dynamic_axis_env.sizes[:dynamic_axis_env.index(frame)+1]
    nreps = dynamic_axis_env.nreps
    trace = frame.pmap_trace

    out_aval = ShapedArray((), np.int32)
    out_tracer = pe.JaxprTracer(trace, pe.PartialVal.unknown(out_aval), None)
    eqn = pe.new_eqn_recipe([], [out_tracer], axis_index_p,
                            dict(nreps=nreps, sizes=sizes, axis_name=axis_name),
                            source_info_util.current())
    out_tracer.recipe = eqn

    return out_tracer

  def _axis_index_translation_rule(c, nreps, sizes, axis_name):
    div = xb.constant(c, np.array(nreps // prod(sizes), dtype=np.uint32))
    mod = xb.constant(c, np.array(sizes[-1], dtype=np.uint32))
    unsigned_index = xops.Rem(xops.Div(xops.ReplicaId(c), div), mod)
    return xops.ConvertElementType(unsigned_index, xb.dtype_to_etype(np.int32))

  axis_index_p.def_custom_bind(_axis_index_bind)
  axis_index_p.def_abstract_eval(
      lambda *args, **params: ShapedArray((), np.int32))
  xla.translations[axis_index_p] = _axis_index_translation_rule
