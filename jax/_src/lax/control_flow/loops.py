# Copyright 2022 The JAX Authors.
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
"""Module for the loop primitives."""
from functools import partial
import itertools
import operator

from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar

import jax
import weakref
from jax import core
from jax import linear_util as lu
from jax.config import config
from jax.core import ConcreteArray, ShapedArray, raise_to_shaped
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
import jax._src.pretty_printer as pp
from jax.tree_util import (tree_flatten, tree_unflatten, treedef_is_leaf,
                           tree_map)
from jax._src import ad_checkpoint
from jax._src import ad_util
from jax._src import api
from jax._src import dtypes
from jax._src import source_info_util
from jax._src import util
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lax import windowed_reductions
from jax._src.lax.control_flow import conditionals
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo
from jax._src.traceback_util import api_boundary
from jax._src.util import (
    extend_name_stack,
    partition_list,
    safe_map,
    safe_zip,
    split_list,
    unzip2,
    weakref_lru_cache,
    )
import numpy as np

from jax._src.lax.control_flow.common import (
    _abstractify,
    _avals_short,
    _check_tree_and_avals,
    _initial_style_jaxpr,
    _make_closed_jaxpr,
    _prune_zeros,
    _typecheck_param,
    allowed_effects,
    )

_map = safe_map
zip = safe_zip

T = TypeVar('T')
Array = Any
BooleanNumeric = Any  # A bool, or a Boolean array.

### Helper functions

def _promote_weak_typed_inputs(in_vals, in_avals, out_avals):
  """Promote weakly-typed in_vals to be compatible with out_avals.

  Args:
    in_vals : flattened list of input values.
    in_avals : corresponding list of avals.
    out_avals : list of target output avals.
  Returns:
    in_vals_new : flattened list of modified in_vals with no weak types.
    changed : bool; true if in_vals required modification.
  """
  if len(in_vals) != len(in_avals) or len(in_avals) != len(out_avals):
    # Calling function is responsible for catching this.
    return in_vals, False
  weak_mismatches = [i for i, (a1, a2) in enumerate(zip(in_avals, out_avals))
                    if getattr(a1, 'weak_type', False) and not core.typematch(a1, a2)]
  if not weak_mismatches:
    return in_vals, False
  for i in weak_mismatches:
    new_dtype = dtypes.result_type(in_vals[i], out_avals[i])
    in_vals[i] = lax.convert_element_type(in_vals[i], new_dtype)
  return in_vals, True


### scan

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')

@api_boundary
def scan(f: Callable[[Carry, X], Tuple[Carry, Y]],
         init: Carry,
         xs: X,
         length: Optional[int] = None,
         reverse: bool = False,
         unroll: int = 1,
         early_exit: Optional[Callable[[Carry], bool]] = None,
         ) -> Tuple[Carry, Y]:
  """Scan a function over leading array axes while carrying along state.

  The `Haskell-like type signature`_ in brief is

  .. code-block:: haskell

    scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])

  where we use [t] here to denote the type t with an additional leading axis.
  That is, if t is an array type then [t] represents the type with an additional
  leading axis, and if t is a pytree (container) type with array leaves then [t]
  represents the type with the same pytree structure and corresponding leaves
  each with an additional leading axis.

  When ``a`` is an array type or None, and ``b`` is an array type, the semantics
  of ``scan`` are given roughly by this Python implementation::

    def scan(f, init, xs, length=None):
      if xs is None:
        xs = [None] * length
      carry = init
      ys = []
      for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
      return carry, np.stack(ys)

  Unlike that Python version, both ``a`` and ``b`` may be arbitrary pytree
  types, and so multiple arrays can be scanned over at once and produce multiple
  output arrays. (None is actually an empty pytree.)

  Also unlike that Python version, ``scan`` is a JAX primitive and is lowered to
  a single XLA While HLO. That makes it useful for reducing compilation times
  for jit-compiled functions, since native Python loop constructs in an ``@jit``
  function are unrolled, leading to large XLA computations.

  Finally, the loop-carried value ``carry`` must hold a fixed shape and dtype
  across all iterations (and not just be consistent up to NumPy rank/shape
  broadcasting and dtype promotion rules, for example). In other words, the type
  ``c`` in the type signature above represents an array with a fixed shape and
  dtype (or a nested tuple/list/dict container data structure with a fixed
  structure and arrays with fixed shape and dtype at the leaves).

  Args:
    f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
      that ``f`` accepts two arguments where the first is a value of the loop
      carry and the second is a slice of ``xs`` along its leading axis, and that
      ``f`` returns a pair where the first element represents a new value for
      the loop carry and the second represents a slice of the output.
    init: an initial loop carry value of type ``c``, which can be a scalar,
      array, or any pytree (nested Python tuple/list/dict) thereof, representing
      the initial loop carry value. This value must have the same structure as
      the first element of the pair returned by ``f``.
    xs: the value of type ``[a]`` over which to scan along the leading axis,
      where ``[a]`` can be an array or any pytree (nested Python
      tuple/list/dict) thereof with consistent leading axis sizes.
    length: optional integer specifying the number of loop iterations, which
      must agree with the sizes of leading axes of the arrays in ``xs`` (but can
      be used to perform scans where no input ``xs`` are needed).
    reverse: optional boolean specifying whether to run the scan iteration
      forward (the default) or in reverse, equivalent to reversing the leading
      axes of the arrays in both ``xs`` and in ``ys``.
    unroll: optional positive int specifying, in the underlying operation of the
      scan primitive, how many scan iterations to unroll within a single
      iteration of a loop.
    early_exit: optional callable. Called at each step; if it returns True
      then the scan is halted early.

  Returns:
    A pair of type ``(c, [b])`` where the first element represents the final
    loop carry value and the second element represents the stacked outputs of
    the second output of ``f`` when scanned over the leading axis of the inputs.

  .. _Haskell-like type signature: https://wiki.haskell.org/Type_signature
  """
  if not callable(f):
    raise TypeError("lax.scan: f argument should be a callable.")
  if early_exit is None: early_exit = lambda carry: False
  if not callable(early_exit):
    raise TypeError("lax.scan: early_exit argument should be a callable or None.")
  xs_flat, xs_tree = tree_flatten(xs)

  try:
    lengths = [x.shape[0] for x in xs_flat]
  except AttributeError as err:
    msg = "scan got value with no leading axis to scan over: {}."
    raise ValueError(
      msg.format(', '.join(str(x) for x in xs_flat
                           if not hasattr(x, 'shape')))) from err

  if length is not None:
    length = int(length)
    if not all(length == l for l in lengths):
      msg = ("scan got `length` argument of {} which disagrees with "
             "leading axis sizes {}.")
      raise ValueError(msg.format(length, [x.shape[0] for x in xs_flat]))
  else:
    unique_lengths = set(lengths)
    if len(unique_lengths) > 1:
      msg = "scan got values with different leading axis sizes: {}."
      raise ValueError(msg.format(', '.join(str(x.shape[0]) for x in xs_flat)))
    elif len(unique_lengths) == 0:
      msg = "scan got no values to scan over and `length` not provided."
      raise ValueError(msg)
    else:
      length, = unique_lengths

  if config.jax_disable_jit:
    try:
      if length == 0:
        raise ValueError("zero-length scan is not supported in disable_jit() mode because "
                         "the output type is unknown.")
      carry = init
      ys = []
      maybe_reversed = reversed if reverse else lambda x: x
      for i in maybe_reversed(range(length)):
        if early_exit(carry):
          break
        else:
          xs_slice = [_index_array(i, x) for x in xs_flat]
          carry, y = f(carry, tree_unflatten(xs_tree, xs_slice))
          expand_y = tree_map(lambda yi: yi[None], y)
          ys.append(expand_y)
      if len(ys) != length:
        if len(ys) == 0:
          raise ValueError("zero-length scan (due to early exit) is not supported in "
                           "disable_jit() mode because the output type is unknown.")
        pad_array = lambda x: jax.numpy.zeros((length - len(ys),) + x.shape, dtype=x.dtype)
        y_like = tree_map(lambda yi: yi[0], ys[0])
        padding = tree_map(pad_array, y_like)
        ys.append(padding)
      concatenate = lambda *ys: jax.numpy.concatenate(ys)
      stacked_y = tree_map(concatenate, *maybe_reversed(ys))
      return carry, stacked_y
    except core.ConcretizationTypeError:
      # e.g. due to vmap or checkpoint. In this case fall back to primitive version
      pass

  def aug_f(carry, x):
    out = f(carry, x)
    if not hasattr(out, "__len__") or len(out) != 2:
      msg = "scan body output must be a pair, got {}."
      raise TypeError(msg.format(out))
    carry, y = out
    done = early_exit(carry)
    return done, carry, y

  xs_avals = [core.raise_to_shaped(core.get_aval(x)) for x in xs_flat]
  x_avals = [core.mapped_aval(length, 0, aval) for aval in xs_avals]

  def _create_jaxpr(init):
    init_flat, init_tree = tree_flatten(init)
    in_flat, in_tree = tree_flatten((init, xs))

    carry_avals = tuple(_map(_abstractify, init_flat))
    jaxpr, consts, out_tree = _initial_style_jaxpr(
        aug_f, in_tree, (*carry_avals, *x_avals), "scan")
    out_tree_children = out_tree.children()
    carry_avals_out = jaxpr.out_avals[1:1+out_tree_children[1].num_leaves]
    return init_flat, carry_avals, carry_avals_out, init_tree, in_flat, jaxpr, consts, out_tree, out_tree_children

  # The carry input and output avals must match exactly. However, we want to account for
  # the case when init contains weakly-typed values (e.g. Python scalars), with avals that
  # may not match the output despite being compatible by virtue of their weak type.
  # To do this, we compute the jaxpr in two passes: first with the raw inputs, and if
  # necessary, a second time with modified init values.
  init_flat, carry_avals, carry_avals_out, init_tree, *rest = _create_jaxpr(init)
  new_init_flat, changed = _promote_weak_typed_inputs(init_flat, carry_avals, carry_avals_out)
  if changed:
    new_init = tree_unflatten(init_tree, new_init_flat)
    init_flat, carry_avals, carry_avals_out, init_tree, *rest = _create_jaxpr(new_init)
  in_flat, jaxpr, consts, out_tree, out_tree_children = rest

  _check_tree_and_avals("scan carry output and input",
                        # Extract the subtree and avals for the first element of the return tuple
                        out_tree_children[1], carry_avals_out,
                        init_tree, carry_avals)
  disallowed_effects = jaxpr.effects - allowed_effects
  if disallowed_effects:
    raise NotImplementedError(
        f'Effects not supported in `scan`: {disallowed_effects}')

  # use default precision with no weak types
  int_ = dtypes.canonicalize_dtype(int)
  start = jax.numpy.array(0, dtype=int_)
  stop = jax.numpy.array(length, dtype=int_)
  done = jax.numpy.array(early_exit(init), dtype=np.bool_)
  out = scan_p.bind(start, stop, done, *consts, *in_flat,
                    reverse=reverse, length=length, jaxpr=jaxpr,
                    num_consts=len(consts), num_carry=len(init_flat),
                    linear=(False,) * (3 + len(consts) + len(in_flat)),
                    unroll=unroll)
  out = out[2:]  # trim off start and stop
  _, carry, ys = tree_unflatten(out_tree, out)  # trim off done
  return carry, ys  # type: ignore[return-value]

def _scan_impl_unrolled(start, stop, done, consts, init, xs, *,
                        reverse, length, linear, f_impl, x_avals, y_avals):
  if reverse:
    j = stop
  else:
    j = start - 1
  carry = init
  ys = []

  false = lax._const(done, False)
  inf_y_avals = _map(partial(_empty_array, None), y_avals)

  for i in range(length):
    if reverse:
      i = length - i - 1
    oob = (i < start) | (i >= stop)
    # Don't index into forbidden regions
    j = lax.select(done, j, lax.max(lax.min(i, stop - 1), start))
    x = _map(partial(_dynamic_index_array, j), xs)
    done2, carry, y = conditionals.cond(oob | done,
                                        lambda *_: (false, carry, inf_y_avals),
                                        f_impl,
                                        consts, carry, x)
    done = done | done2
    ys.append(y)

  if reverse:
    new_start = j
    new_stop = stop
  else:
    new_start = start
    new_stop = j + 1

  ys = list(reversed(ys)) if reverse else ys
  ys = list(zip(*ys))
  ys = _map(_stack, y_avals, ys)
  return new_start, new_stop, done, carry, ys

def _scan_impl_loop(start, stop, done, consts, init, xs, *,
                    reverse, length, linear, f_impl, x_avals, y_avals):
  def cond_fun(vals):
    i, done, _, _ = vals
    if reverse:
      not_oob = i >= start
    else:
      not_oob = i < stop
    return not_oob & lax.bitwise_not(done)

  def body_fun(vals):
    i, _, carry, ys = vals
    x = _map(partial(_dynamic_index_array, i), xs)
    done, carry_out, y_updates = f_impl(consts, carry, x)
    ys_out = _map(partial(_update_array, i), ys, y_updates)
    if reverse:
      i_next = i - 1
    else:
      i_next = i + 1
    return i_next, done, carry_out, ys_out

  ys_init = _map(partial(_empty_array, length), y_avals)
  if length == 0:
    return start, stop, done, init, ys_init
  else:
    i_init = stop - 1 if reverse else start
    init_val = (i_init, done, init, ys_init)
    i_final, done, carry, ys = while_loop(cond_fun, body_fun, init_val)
    if reverse:
      new_start = i_final + 1
      new_stop = stop
    else:
      new_start = start
      new_stop = i_final
    return new_start, new_stop, done, carry, ys

def _scan_impl_block(consts, carry, xs, *,
                     reverse, length, linear, f_impl, x_avals, y_avals):
    def aug_f_impl(consts, carry, x):
      num_steps, *carry = carry
      done, carry, y = f_impl(consts, carry, x)
      return done, [num_steps + 1, *carry], y
    [start, stop], xs = split_list(xs, [2])
    done = False
    _, _, done, carry, ys = _scan_impl_unrolled(start, stop, done, consts, carry, xs,
                                                reverse=reverse, length=length,
                                                linear=linear, f_impl=aug_f_impl,
                                                x_avals=x_avals, y_avals=y_avals)
    return done, carry, ys

def _scan_impl_block_unrolled(start, stop, done, consts, init, xs, *,
                              reverse, length, linear, block_length,
                              f_impl, x_avals, y_avals):
  num_blocks, rem = divmod(length, block_length)
  assert rem == 0

  dtype = lax.dtype(start)
  assert dtype == lax.dtype(stop)
  if num_blocks == 0:
    starts = stops = lax.full((0,), 0, dtype=dtype)
    block_start = 0
    block_stop = 0
  else:
    zeros = lax.full((num_blocks,), 0, dtype=dtype)
    fulls = lax.full((num_blocks,), block_length, dtype=dtype)

    block_start, subblock_start = divmod(start, block_length)
    block_stop, subblock_stop = divmod(stop, block_length)

    iota = lax.iota(dtype, num_blocks)
    pred_start = iota < block_start
    pred_stop = iota < block_stop

    starts = lax.select(pred_start, fulls, zeros)
    stops = lax.select(pred_stop, fulls, zeros)
    starts = starts.at[block_start].set(subblock_start)
    # i.e.
    # if block_stop != num_blocks:
    #   stops.at[block_stop].set(subblock_stop)
    block_stop2 = lax.min(block_stop, num_blocks - 1)
    stops = stops.at[block_stop2].set(
        lax.select(block_stop == num_blocks, stops[block_stop2], subblock_stop))

  partition = partial(_partition_leading, num_blocks, block_length)
  xs_block = [starts, stops] + _map(partition, x_avals, xs)

  prepend_aval = partial(_prepend_dim_to_aval, block_length)
  x_block_avals = _map(prepend_aval, x_avals)
  y_block_avals = _map(prepend_aval, y_avals)

  f_impl_block = partial(
      _scan_impl_block, reverse=reverse, length=block_length,
      linear=linear, f_impl=f_impl, x_avals=x_avals, y_avals=y_avals)

  carry = [0] + init
  _, _, new_done, [num_steps, *carry], ys_blocks = \
      _scan_impl_loop(
          block_start, block_stop, done, consts, carry, xs_block,
          reverse=reverse, length=num_blocks, linear=linear,
          f_impl=f_impl_block, x_avals=x_block_avals, y_avals=y_block_avals)

  if reverse:
    new_start = stop - num_steps
    new_stop = stop
  else:
    new_start = start
    new_stop = start + num_steps

  combine = partial(_combine_leading, num_blocks, block_length)
  ys = _map(combine, y_avals, ys_blocks)
  return new_start, new_stop, new_done, carry, ys

def _scan_impl(*args, reverse, length, num_consts, num_carry, jaxpr, linear,
               unroll):
  _, _, x_avals = split_list(jaxpr.in_avals, [num_consts, num_carry])
  _, _, y_avals = split_list(jaxpr.out_avals, [num_carry, 1])

  def f_impl(consts, carry, x):
    out_flat = core.jaxpr_as_fun(jaxpr)(*consts, *carry, *x)
    [done], carry_out, y_updates = split_list(out_flat, [1, num_carry])
    return done, carry_out, y_updates

  [start, stop, done], consts, init, xs = split_list(args, [3, num_consts, num_carry])
  assert start.shape == ()
  assert stop.shape == ()
  assert done.shape == ()

  if unroll == 1:
    new_start, new_stop, new_done, carry, ys = _scan_impl_loop(
        start, stop, done, consts, init, xs, reverse=reverse, length=length,
        linear=linear, f_impl=f_impl, x_avals=x_avals, y_avals=y_avals)
    return (new_start, new_stop, new_done, *carry, *ys)

  num_blocks, rem = divmod(length, unroll)
  length_div = num_blocks * unroll

  if rem > 0:
    if reverse:
      split = partial(_split_leading_dim, rem)
      xs_rem, xs = unzip2(_map(split, x_avals, xs))
    else:
      split = partial(_split_leading_dim, length_div)
      xs, xs_rem = unzip2(_map(split, x_avals, xs))

  new_start, new_stop, new_done, carry, ys = _scan_impl_block_unrolled(
      lax.min(start, length_div), lax.min(stop, length_div), done, consts, init, xs,
      reverse=reverse, length=length_div, linear=linear, block_length=unroll,
      f_impl=f_impl, x_avals=x_avals, y_avals=y_avals)

  if rem > 0:
    start_rem = lax.max(0, start - length_div)
    stop_rem = lax.max(0, stop - length_div)
    new_start_rem, new_stop_rem, new_done, carry, ys_rem = _scan_impl_unrolled(
        start_rem, stop_rem, done, consts, carry, xs_rem, reverse=reverse, length=rem,
        linear=linear, f_impl=f_impl, x_avals=x_avals, y_avals=y_avals)
    new_start = lax.select(start < length_div,
                           new_start,
                           new_start_rem + length_div)
    new_stop = lax.select(stop <= length_div,
                          new_stop,
                          new_stop_rem + length_div)
    if reverse:
      ys = _map(_concatenate, y_avals, ys_rem, ys)
    else:
      ys = _map(_concatenate, y_avals, ys, ys_rem)

  return (new_start, new_stop, new_done, *carry, *ys)

def _stack(aval, vals):
  vals = [lax.expand_dims(x, (0,)) for x in vals]
  return lax.concatenate(vals, 0)

def _concatenate(aval, x1, x2):
  return lax.concatenate([x1, x2], 0)

def _split_leading_dim(i, aval, x):
  assert x.ndim >= 1
  return (slicing.slice_in_dim(x, 0, i),
          slicing.slice_in_dim(x, i, x.shape[0]))

def _dynamic_index_array(i, x):
  return slicing.dynamic_index_in_dim(x, i, keepdims=False)

def _index_array(i, x):
  return slicing.index_in_dim(x, i, keepdims=False)

def _empty_array(sz, aval):
  if sz is None:
    shape = aval.shape
  else:
    shape = (sz, *aval.shape)
  return lax.broadcast(lax.empty(aval.dtype), shape)

def _update_array(i, xs, x):
  return slicing.dynamic_update_index_in_dim(xs, x, i, 0)

def _partition_leading(sz0, sz1, aval, x):
  assert x.ndim >= 1
  assert x.shape[0] == sz0 * sz1
  return lax.reshape(x, (sz0, sz1, *x.shape[1:]))

def _combine_leading(sz0, sz1, aval, x):
  assert x.ndim >= 2
  assert x.shape[0] == sz0
  assert x.shape[1] == sz1
  return lax.collapse(x, 0, 2)

def _prepend_dim_to_aval(sz, aval):
  return core.unmapped_aval(sz, core.no_axis_name, 0, aval)

def _is_literal_false(x):
  return type(x) is core.Literal and x.val is False

_literal_false = core.Literal(False, core.ShapedArray((), np.dtype(bool)))

def _scan_abstract_eval(*args, reverse, length, num_consts, num_carry, jaxpr,
                        linear, unroll):
  [start_aval, stop_aval, done_aval] = args[:3]
  _, carry_avals, y_avals = split_list(jaxpr.out_avals, [1, num_carry])
  ys_avals = _map(partial(_prepend_dim_to_aval, length), y_avals)
  return [start_aval, stop_aval, done_aval] + carry_avals + ys_avals, jaxpr.effects

def _scan_jvp(primals, tangents, reverse, length, jaxpr, num_consts, num_carry,
              linear, unroll):
  num_xs = len(jaxpr.in_avals) - num_carry - num_consts
  num_ys = len(jaxpr.out_avals) - 1 - num_carry
  tangents = tangents[3:]  # trim off start, stop, done
  nonzeros = [type(t) is not ad_util.Zero for t in tangents]
  const_nz, init_nz, xs_nz = split_list(nonzeros, [num_consts, num_carry])

  # Fixpoint computation of which carry are not ad.zero: either
  # non-zero from init, or the carry out is non-zero. Each iteration promotes
  # at least one carry to non-zero. We need at most len(carry) iterations,
  # but we need one last iteration to prepare the jaxpr based on the final
  # carry_nz.
  carry_nz = init_nz
  for _ in range(1 + len(carry_nz)):
    nonzeros = const_nz + carry_nz + xs_nz
    jaxpr_jvp, nonzeros_out = ad.jvp_jaxpr(
        jaxpr, nonzeros, instantiate=[False] + carry_nz + [False] * num_ys)
    carry_nz_out = nonzeros_out[1:1+num_carry]
    if carry_nz_out == carry_nz:
      break
    else:
      carry_nz = _map(operator.or_, carry_nz, carry_nz_out)
  else:
    assert False, "Fixpoint not reached"

  tangents = [ad.instantiate_zeros(t) if nz else t
              for t, nz in zip(tangents, nonzeros)]

  [start, stop, done], consts, init, xs = split_list(
      primals, [3, num_consts, num_carry])
  all_tangents = split_list(tangents, [num_consts, num_carry])
  consts_dot, init_dot, xs_dot = _map(_prune_zeros, all_tangents)

  jaxpr_jvp_rearranged = ad.rearrange_binders(
      jaxpr_jvp,
      [num_consts, num_carry, num_xs], [len(consts_dot), len(init_dot), len(xs_dot)],
      [1, num_carry, num_ys], [0, len(init_dot), sum(nonzeros_out) - len(init_dot)])

  [start_linear, stop_linear, done_linear], consts_linear, init_linear, xs_linear = \
                                 split_list(linear, [3, num_consts, num_carry])
  jaxpr_jvp_linear = tuple([start_linear, stop_linear, done_linear]
                           + consts_linear + [True] * len(consts_dot)
                           + init_linear + [True] * len(init_dot)
                           + xs_linear + [True] * len(xs_dot))

  out_flat = scan_p.bind(
      start, stop, done, *consts, *consts_dot, *init, *init_dot, *xs, *xs_dot,
      reverse=reverse, length=length, jaxpr=jaxpr_jvp_rearranged,
      num_consts=num_consts + len(consts_dot),
      num_carry=num_carry + len(init_dot),
      linear=jaxpr_jvp_linear, unroll=unroll)

  [start, stop, done], carry, carry_dot, ys, ys_dot = \
                 split_list(out_flat, [3, num_carry, len(init_dot), num_ys])
  primals_out = [start, stop, done, *carry, *ys]
  tangents_out_iter = iter(carry_dot + ys_dot)
  tangents_out = [next(tangents_out_iter) if nz else ad_util.Zero.from_value(p)
                  for p, nz in zip(primals_out, (False, False, *nonzeros_out))]
  assert next(tangents_out_iter, None) is None
  return primals_out, tangents_out

def _scan_partial_eval(trace, *tracers, reverse, length, num_consts, num_carry,
                       jaxpr, linear, unroll):
  num_ys = len(jaxpr.out_avals) - 1 - num_carry
  unknowns = [not t.pval.is_known() for t in tracers]
  [start_uk, stop_uk, done_uk], const_uk, init_uk, xs_uk = \
                            split_list(unknowns, [3, num_consts, num_carry])
  if done_uk or all([not uk for uk in unknowns]) or all(unknowns):
    # If conditional is unknown, or all inputs are known, or all are unknown,
    # just do the default processing.
    params = dict(reverse=reverse, length=length, num_consts=num_consts,
                  num_carry=num_carry, jaxpr=jaxpr, linear=linear, unroll=unroll)
    return trace.default_process_primitive(scan_p, tracers, params)
  del unknowns
  assert not start_uk
  assert not stop_uk

  # Fixpoint computation of which carry elements are unknown. Each iteration
  # promotes at least one carry to unknown. We need at most len(carry)
  # iterations, but we need one last iteration to prepare the jaxpr based on the
  # final carry_uk.
  carry_uk = init_uk
  for _ in range(1 + len(carry_uk)):
    unknowns = const_uk + carry_uk + xs_uk
    jaxpr_known, jaxpr_unknown, out_uk, res_avals = pe.partial_eval_jaxpr_nounits(
        jaxpr, unknowns, instantiate=[False] + carry_uk + [False] * num_ys)
    [done_uk], carry_uk_out, ys_uk = split_list(out_uk, [1, num_carry])
    if carry_uk_out == carry_uk:
      break
    else:
      carry_uk = _map(operator.or_, carry_uk, carry_uk_out)
  else:
    assert False, "Fixpoint not reached"
  if done_uk:
    raise NotImplementedError
  num_res = len(res_avals)
  del res_avals, carry_uk_out
  # At this point, the signatures of each jaxpr look like:
  # jaxpr :: (consts, carry, x) -> (done, carry, y)
  # jaxpr_known :: (known_consts, known_carry, known_x] -> (done, known_carry, known_y, residuals)
  # jaxpr_unknown :: (residuals, unknown_consts, unknown_carry, unknown_x) -> (unknown_carry, unknown_y)

  # Instantiate those inputs which must be treated as unknown from the fixpoint.
  tracers = [trace.instantiate_const(t) if uk else t
             for t, uk in zip(tracers, (False, False, False, *unknowns))]

  # The residual inputs and outputs of the jaxprs produced haven't yet been
  # adapted to the scan calling convention; in particular, jaxpr_known has its
  # residual outputs all at the end, meaning they're extensive outputs (which is
  # fully general but may be wasteful for residuals which are loop-invariant)
  # while jaxpr_unknown has its corresponding residual inputs at the front (just
  # as a convention with partial_eval_jaxpr_nounits), making them constant
  # inputs. To make them consistent, we move the residual inputs on
  # jaxpr_unknown to the end, even though we may move some back in the sequel.
  jaxpr_unknown = pe.move_binders_to_back(
      jaxpr_unknown, [True] * num_res + [False] * sum(unknowns))
  # Now
  # jaxpr_unknown :: (unknown_consts, unknown_carry, unknown_x, residuals) -> (unknown_carry, unknown_y)

  # At this point, all residuals are treated as extensive outputs of jaxpr_known
  # (and extensive inputs to jaxpr_unknown). But residuals that are loop-
  # invariant can be hoisted out of the scan, rather than letting them get
  # broadcast (as in e.g. scanning multiplication by a constant matrix; we don't
  # want to broadcast the matrix!). So, outside the loop we perform a partial
  # evaluation with known 'const' inputs (but all other inputs unknown).
  const_pvals = [pe.PartialVal.known(t.pval.get_known())
                 for t in tracers[3:3+num_consts] if t.pval.is_known()]
  other_pvals = [pe.PartialVal.unknown(aval)
                 for aval in jaxpr_known.in_avals[len(const_pvals):]]
  with source_info_util.reset_name_stack():
    jaxpr_known_, invar_pvals_out, jaxpr_known_consts = pe.trace_to_jaxpr_nounits(
        lu.wrap_init(core.jaxpr_as_fun(jaxpr_known)), const_pvals + other_pvals,
        instantiate=[True] * (len(out_uk) - sum(out_uk)) + [False] * num_res)
  jaxpr_known = pe.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr_known_), ())
  # jaxpr_known :: (jaxpr_known_consts, known_carry, known_x) -> (done, known_carry, known_y, extensive_res)
  # invar_pvals_out :: (done, known_carry, known_y, res)
  # jaxpr_known_consts :: (known_consts, intensive_res)

  # The above trace_to_jaxpr_nounits call computed loop-invariant residuals
  # (known values in invar_pvals_out) and also computed loop-invariant values
  # needed by the new jaxpr_known (in jaxpr_known_consts, which replace the
  # previous consts). We need to collect the computed intensive residuals, and
  # move corresponding intensive residual binders in jaxpr_unknown to the front.
  res_pvals = invar_pvals_out[len(invar_pvals_out) - num_res:]
  intensive_res = [pval.get_known() for pval in res_pvals if pval.is_known()]
  jaxpr_unknown = pe.move_binders_to_front(
      jaxpr_unknown,
      [False] * sum(unknowns) + [pval.is_known() for pval in res_pvals])
  del const_pvals, other_pvals, invar_pvals_out, jaxpr_known_, res_pvals
  # jaxpr_unknown :: (intensive_res, unknown_consts, unknown_carry, unknown_x, extensive_res) -> (unknown_carry, unknown_y)
  # We use `jaxpr_known_consts` when we call scan_p.bind with jaxpr_known, and
  # we use `intensive_res` when we build the jaxpr eqn with jaxpr_unknown.

  # As another optimization, for any extensive inputs that are just forwarded to
  # extensive outputs, to avoid a copy (which would be looping over
  # dynamic-update-slice) we'd rather forward the input tracer/value. That means
  # pruning some outputs from jaxpr_known here, and updating `out_flat` below.
  fwds_known = pe._jaxpr_forwarding(jaxpr_known.jaxpr)
  # Prune fwds_known to include only extensive input to extensive output.
  fwds_known = [in_idx if out_idx >= 1 + num_carry - sum(carry_uk) and
                in_idx is not None and
                in_idx >= len(jaxpr_known_consts) + num_carry - sum(carry_uk)
                else None for out_idx, in_idx in enumerate(fwds_known)]
  # Drop any extensive output we can instead get by forwarding an input.
  # TODO(mattjj): use pe.dce_jaxpr here, though need a fixpoint
  jaxpr_known_, () = jaxpr_known.jaxpr, jaxpr_known.consts
  jaxpr_known_.outvars = [x for x, i in zip(jaxpr_known_.outvars, fwds_known)
                          if i is None]
  jaxpr_known = core.ClosedJaxpr(jaxpr_known_, ())
  del jaxpr_known_
  # We use `fwds_known` below when forming the output of scanning jaxpr_known.

  # jaxpr_known :: (jaxpr_known_consts, known_carry, known_x) -> (done, known_carry, unfwd_known_y, unfwd_extensive_res)

  # Run the known part of the scan (if it has any outputs or effects).
  known_inputs = ([t.pval.get_known() for t in tracers[:3]]  # start, stop, done
                  + list(jaxpr_known_consts)
                  + [t.pval.get_known() for t in tracers[3+num_consts:]
                     if t.pval.is_known()])
  if not jaxpr_known.out_avals and not jaxpr_known.effects:
    out_known = []
  else:
    linear_known = [False] * len(known_inputs)  # conservative!
    out_start, out_stop, *out_known = scan_p.bind(
        *known_inputs, reverse=reverse, length=length, jaxpr=jaxpr_known,
        num_consts=len(jaxpr_known_consts), num_carry=num_carry - sum(carry_uk),
        linear=tuple(linear_known), unroll=unroll)
    del linear_known
  # Complete the known output by filling in forwarded values using fwds_known.
  out_known_iter = iter(out_known)
  out_known = [next(out_known_iter) if f is None
               else _maybe_put(known_inputs[f + 3]) for f in fwds_known]
  # out_known :: (done, known_carry, known_y, extensive_res)
  assert next(out_known_iter, None) is None
  del known_inputs, out_known_iter, fwds_known

  # Split known outputs from residuals.
  carry_ys_uk = carry_uk + ys_uk
  [out_done], out_known, extensive_res = \
        split_list(out_known, [1, len(carry_ys_uk) - sum(carry_ys_uk)])
  assert len(intensive_res) + len(extensive_res) == num_res
  # out_known :: (known_carry, known_y)

  # Unknown part never uses early-exit. This was assumed known, above.
  # Instead we use out_start and out_stop to match the same region of
  # iteration that the known part used.
  jaxpr_unknown = jaxpr_unknown.replace(jaxpr=jaxpr_unknown.jaxpr.replace(
        outvars=[_literal_false] + jaxpr_unknown.jaxpr.outvars))
  # jaxpr_unknown :: (intensive_res, unknown_consts, unknown_carry, unknown_x, extensive_res) -> (done=False, unknown_carry, unknown_y)

  # Create input tracers for jaxpr_unknown bind.
  unknown_inputs = [t for t in tracers[3:] if not t.pval.is_known()]
  start_stop_done_tracers = _map(trace.new_instantiated_const,
                                 (out_start, out_stop, jax.numpy.array(False, dtype=bool)))
  intensive_res = _map(trace.new_instantiated_const, intensive_res)
  extensive_res = _map(trace.new_instantiated_const, extensive_res)
  # Create output tracers for jaxpr_unknown bind, adapting extensive shapes.
  start_stop_done_avals = [core.get_aval(x) for x in (out_start, out_stop, out_done)]
  _, carry_avals, y_avals = split_list(jaxpr_unknown.out_avals, [1, sum(carry_uk)])
  ys_avals = [core.unmapped_aval(length, core.no_axis_name, 0, y_aval)
              for y_aval in y_avals]
  out_tracers = [pe.JaxprTracer(trace, pe.PartialVal.unknown(a), None)
                 for a in itertools.chain(start_stop_done_avals, carry_avals, ys_avals)]
  del carry_avals, y_avals
  # Create equation.
  linear_unknown = tuple([False, False, False] +
                         [False] * len(intensive_res) +
                         [l for l, uk in zip(linear[3:], unknowns) if uk] +
                         [False] * len(extensive_res))
  name_stack = source_info_util.current_name_stack()[len(trace.name_stack):]
  source = source_info_util.current().replace(name_stack=name_stack)
  assert len(out_tracers) == 2 + len(jaxpr_unknown.out_avals)
  eqn = pe.new_eqn_recipe(
        [*start_stop_done_tracers, *intensive_res, *unknown_inputs, *extensive_res],
        out_tracers,
        scan_p,
        dict(reverse=reverse, length=length, unroll=unroll,
             jaxpr=jaxpr_unknown, linear=linear_unknown,
             num_consts=len(intensive_res) + sum(const_uk),
             num_carry=sum(carry_uk)),
        jaxpr_unknown.effects,
        source)
  for t in out_tracers: t.recipe = eqn

  # Merge known and unknown outputs into final result.
  return [out_start, out_stop, out_done] + util.merge_lists(carry_ys_uk, out_known, out_tracers[3:])

def _maybe_put(x):
  if isinstance(x, np.ndarray):
    return jax.device_put(x, jax.devices('cpu')[0])
  else:
    return x

def _scan_transpose(reduce_axes, cts, *args, reverse, length, num_consts,
                    num_carry, jaxpr, linear, unroll):
  [start_lin, stop_lin, done_lin], consts_lin, init_lin, xs_lin = \
                                 split_list(linear, [3, num_consts, num_carry])
  assert not start_lin
  assert not stop_lin
  assert not done_lin
  # we've only implemented transposing scans with specific lin/nonlin patterns
  num_ires = len(consts_lin) - sum(consts_lin)
  num_eres = len(xs_lin) - sum(xs_lin)
  if consts_lin != [False] * num_ires + [True] * (len(consts_lin) - num_ires):
    raise NotImplementedError
  if xs_lin != [True] * (len(xs_lin) - num_eres) + [False] * num_eres:
    raise NotImplementedError
  if not all(init_lin):
    pass  # TODO(mattjj): error check https://github.com/google/jax/issues/1963

  [start, stop, done], consts, _, xs = split_list(args, [3, num_consts, num_carry])
  ires, _ = split_list(consts, [num_ires])
  _, eres = split_list(xs, [sum(xs_lin)])
  assert not any(ad.is_undefined_primal(r) for r in ires)
  assert not any(ad.is_undefined_primal(r) for r in eres)

  _, carry_avals, y_avals = split_list(jaxpr.out_avals, [1, num_carry])
  ys_avals = _map(partial(_prepend_dim_to_aval, length), y_avals)
  _, ct_carry, ct_ys = split_list(cts, [3, num_carry])
  ct_carry = _map(ad.instantiate_zeros_aval, carry_avals, ct_carry)
  ct_ys = _map(ad.instantiate_zeros_aval, ys_avals, ct_ys)
  ct_consts = _map(ad_util.zeros_like_aval, jaxpr.in_avals[num_ires:num_consts])

  # we've only implemented transposing scans without early exit
  if not _is_literal_false(jaxpr.jaxpr.outvars[0]):
    raise NotImplementedError
  jaxpr = jaxpr.replace(jaxpr=jaxpr.jaxpr.replace(outvars=jaxpr.jaxpr.outvars[1:]))
  #       jaxpr :: [ires, T d] -> [T c] -> [T a, eres] -> ([T c], [T b])
  # jaxpr_trans :: [ires] -> [CT d, CT c] -> [CT b, eres] -> ([CT d, CT c], [CT a])
  jaxpr_trans = _transpose_scan_jaxpr(
      num_ires, num_consts - num_ires, num_eres, jaxpr, reduce_axes)
  # no early exit
  newoutvars = [_literal_false] + jaxpr_trans.jaxpr.outvars
  jaxpr_trans = jaxpr_trans.replace(jaxpr=jaxpr_trans.jaxpr.replace(outvars=newoutvars))

  linear_trans = ([False, False, False] +
                  [False] * num_ires +
                  [True] * (len(ct_consts) + len(ct_carry) + len(ct_ys)) +
                  [False] * num_eres)

  outs = scan_p.bind(
      start, stop, done, *ires, *ct_consts, *ct_carry, *ct_ys, *eres, reverse=not reverse,
      length=length, jaxpr=jaxpr_trans, num_consts=num_ires,
      num_carry=num_consts-num_ires+num_carry, linear=tuple(linear_trans),
      unroll=unroll)
  _, ct_consts, ct_init, ct_xs = split_list(outs, [3, num_consts - num_ires, num_carry])
  return [None, None, None] + [None] * num_ires + ct_consts + ct_init + ct_xs + [None] * num_eres

# transpose_scan_jaxpr :: ([res1, c, a, res2] -> b)
#                         -> ([res1, CT c, CT b, res2] -> [CT c, CT a])
def _transpose_scan_jaxpr(num_res1, num_c, num_res2, jaxpr, reduce_axes):
  num_a = len(jaxpr.in_avals) - num_res1 - num_c - num_res2
  # TODO: allow input cotangent avals to be batched relative to jaxpr.in_avals
  # if an axis isn't reduced
  res1_avals, c_avals, a_avals, res2_avals = split_list(
      jaxpr.in_avals, [num_res1, num_c, num_a])
  num_b = len(jaxpr.out_avals)
  b_avals = list(jaxpr.out_avals)

  @lu.wrap_init
  def transposed(*res1_cbar_bbar_res2):
    res1, c_bar, b_bar, res2 = split_list(
        res1_cbar_bbar_res2, [num_res1, num_c, num_b])
    primals = (res1 + [ad.UndefinedPrimal(aval) for aval in c_avals] +
               [ad.UndefinedPrimal(aval) for aval in a_avals] + res2)
    cbar_abar = ad.backward_pass(jaxpr.jaxpr, reduce_axes, False, jaxpr.consts,
                                 primals, b_bar)
    _, new_c_bar, a_bar, _ = split_list(cbar_abar, [num_res1, num_c, num_a])
    a_bar = _map(ad.instantiate_zeros_aval, a_avals, a_bar)
    c_bar = _map(ad.instantiate_zeros_aval, c_avals,
                _map(ad.add_tangents, c_bar, new_c_bar))
    return c_bar + a_bar
  return _make_closed_jaxpr(transposed, res1_avals + c_avals + b_avals + res2_avals)


def _scan_batching_rule(axis_size, axis_name, main_type, args, dims, reverse, length,
                        jaxpr, num_consts, num_carry, linear, unroll):
  num_ys = len(jaxpr.out_avals) - 1 - num_carry
  orig_batched = [d is not batching.not_mapped for d in dims]
  [start, stop, done], consts, init, xs = split_list(args, [3, num_consts, num_carry])
  [start_batched, stop_batched, done_batched], const_batched, init_batched, xs_batched = \
                                  split_list(orig_batched, [3, num_consts, num_carry])
  [start_bdim, stop_bdim, done_bdim], consts_bdims, init_bdims, xs_bdims = \
                                          split_list(dims, [3, num_consts, num_carry])
  del orig_batched, args, dims

  # Heuristic for when to use a more complicated body function. (See below.)
  # This is a sufficient but not necessary condition for needing the more complicated
  # body function; we will unnecessarily use this more complicated body function if
  # using a nontrivial early-exit condition that doesn't exhibit batch dependence.
  # Catching that would involve more tracing though, so we trade runtime in that edge
  # case for improved tracetime in the general case.
  aug_body = start_batched | stop_batched | done_batched | (
        not _is_literal_false(jaxpr.jaxpr.outvars[0]))

  if aug_body:
    batch_start = start if isinstance(start, int) else start.min()
    batch_stop = stop if isinstance(stop, int) else stop.max()
    batch_done = done if isinstance(done, bool) else done.all()
  else:
    batch_start = start
    batch_stop = stop
    batch_done = done

  if aug_body:
    # TODO(kidger): do something a bit more elegant here, this will
    # increase the size of the carry multiple times under nested vmaps

    # Add `start` and `stop` to `consts`.
    # Add `done` to `carry`.
    # Add a step counter to `xs`.
    [start_lin, stop_lin, done_lin], consts_lin, init_lin, xs_lin = \
                                          split_list(linear, [3, num_consts, num_carry])
    linear = ([start_lin, stop_lin, done_lin] +
              consts_lin + [False, False] +
              [False, False] + init_lin +
              [False] + xs_lin)
    linear = tuple(linear)
    del start_lin, stop_lin, done_lin, consts_lin, init_lin, xs_lin
    orig_jaxpr = jaxpr
    orig_num_consts = num_consts
    orig_num_carry = num_carry
    consts = consts + [start, stop]
    const_batched = const_batched + [start_batched, stop_batched]
    consts_bdims = consts_bdims + [start_bdim, stop_bdim]
    num_consts = num_consts + 2
    if reverse:
      j_init = batch_stop
    else:
      j_init = batch_start - 1
    init = [done, j_init] + init
    init_batched = [done_batched, False] + init_batched
    init_bdims = [done_bdim, batching.not_mapped] + init_bdims
    num_carry = 2 + num_carry
    xs = [lax.iota(lax.dtype(start), length)] + xs
    xs_batched = [False] + xs_batched
    xs_bdims = [batching.not_mapped] + xs_bdims

    # New body function, as written for a single batch element.
    # Handles the fact that we may iterate over a [start, stop)
    # range that is larger than this batch element wants, and that we
    # may not respect the `done` condition for this batch element.
    # (Because in each case, we have to iterate over the greatest
    # interval across all batch elements.)
    @lu.wrap_init
    def new_aug_f(*args):
      consts, [start, stop, done, j], carry, [i], x = \
            split_list(args, [orig_num_consts, 4, orig_num_carry, 1])
      outs = core.jaxpr_as_fun(orig_jaxpr)(*consts, *carry, *x)
      [done2], carry2, y = split_list(outs, [1, orig_num_carry])
      oob = (i < start) | (i >= stop)
      pred = oob | done
      j = lax.select(pred, j, lax.max(lax.min(i, stop - 1), start))
      carry = [lax.select(pred, c, c2) for c, c2 in zip(carry, carry2)]
      y = [lax.select(pred, _empty_array(None, y), y) for y in y]
      done = lax.select(pred, done, done | done2)
      # First `done` is the early-exit; second `done` is a carry
      return [done, done, j] + carry + y

    new_args = consts + init + [x[0] for x in xs]
    new_bdims = consts_bdims + init_bdims + xs_bdims
    new_args0 = [x if d is batching.not_mapped else x[d]
                 for x, d in zip(new_args, new_bdims)]
    new_args0 = _map(_abstractify, new_args0)
    jaxpr = _make_closed_jaxpr(new_aug_f, new_args0)
    del j_init, new_aug_f, new_args, new_bdims, new_args0

  # Fixpoint computation of which carry are batched: either
  # batched from init, or the carry out is batched. Each iteration promotes
  # at least one carry to batched. We need at most len(carry) iterations,
  # but we need one last iteration to prepare the jaxpr based on the final
  # carry_batched.
  carry_batched = init_batched
  for _ in range(1 + len(carry_batched)):
    batched = const_batched + carry_batched + xs_batched
    jaxpr_batched, batched_out = batching.batch_jaxpr(
        jaxpr, axis_size, batched,
        instantiate=[False] + carry_batched + [False] * num_ys,
        axis_name=axis_name,
        main_type=main_type)
    [done_batched_out], carry_batched_out, ys_batched = split_list(batched_out, [1, num_carry])
    if carry_batched_out == carry_batched:
      break
    else:
      carry_batched = _map(operator.or_, carry_batched, carry_batched_out)
  else:
    assert False, "Fixpoint not reached"

  if done_batched_out:
    # TODO(kidger): this is inefficient: we should be able to append to
    # the existing jaxpr instead of needlessly retracing it just to add a
    # new equation to the end.
    @lu.wrap_init
    def new_aug_f(*args):
      done, *rest = core.jaxpr_as_fun(jaxpr_batched)(*args)
      return (done.all(), *rest)
    jaxpr_batched = _make_closed_jaxpr(new_aug_f, jaxpr_batched.in_avals)
    del new_aug_f

  new_consts = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0
                else x for x, d in zip(consts, consts_bdims)]
  new_init = [batching.broadcast(x, axis_size, 0) if now_batched and not was_batched
              else batching.moveaxis(x, d, 0) if now_batched else x
              for x, d, was_batched, now_batched in
              zip(init, init_bdims, init_batched, carry_batched)]
  new_xs = [batching.moveaxis(x, d, 1) if d is not batching.not_mapped and d != 1
            else x for x, d in zip(xs, xs_bdims)]
  new_args = [batch_start, batch_stop, batch_done] + new_consts + new_init + new_xs

  outs = scan_p.bind(
      *new_args, reverse=reverse, length=length, jaxpr=jaxpr_batched,
      num_consts=num_consts, num_carry=num_carry, linear=linear, unroll=unroll)

  if aug_body:
    _, out_carry, out_ys = split_list(outs, [3, num_carry])
    [out_done, out_j], out_carry = split_list(out_carry, [2])
    if reverse:
      out_start = out_j
      out_stop = stop
    else:
      out_start = start
      out_stop = out_j + 1
    outs = [out_start, out_stop, out_done, *out_carry, *out_ys]
    [done_batched, j_batched], carry_batched = split_list(carry_batched, [2])
    if reverse:
      start_stop_done_batched = [j_batched, stop_batched, done_batched]
    else:
      start_stop_done_batched = [start_batched, j_batched, done_batched]
    start_stop_done_bdims = [0 if b else batching.not_mapped
                             for b in start_stop_done_batched]
  else:
    out_start = start
    out_stop = stop
    out_done = done
    start_stop_done_bdims = [batching.not_mapped] * 3

  carry_bdims = [0 if b else batching.not_mapped for b in carry_batched]
  ys_bdims = [1 if b else batching.not_mapped for b in ys_batched]
  out_bdims = start_stop_done_bdims + carry_bdims + ys_bdims
  for o, d in zip(outs, out_bdims):
    assert d is batching.not_mapped or d in range(np.ndim(o))
  return outs, out_bdims

def _masked_scan_jaxpr(jaxpr, num_consts, num_carry):
  fun = core.jaxpr_as_fun(jaxpr)

  @lu.wrap_init
  def masked(*args):
    [dynamic_length], consts, [i], carry, xs = split_list(
        args, [1, num_consts, 1, num_carry])
    out = fun(*(consts + carry + xs))
    new_carry, ys = split_list(out, [num_carry])
    new_carry = [lax.select(i < dynamic_length, new_c, c)
                 for new_c, c in zip(new_carry, carry)]
    return [i + 1] + new_carry + ys

  aval = ShapedArray((), dtypes.canonicalize_dtype(dtypes.int_))
  const_avals, carry_avals, x_avals = split_list(jaxpr.in_avals, [num_consts, num_carry])
  return _make_closed_jaxpr(masked, [aval] + const_avals + [aval] + carry_avals + x_avals)

def _scan_padding_rule(in_avals, out_avals, *args, jaxpr, **params):
  padded_jaxpr = core.ClosedJaxpr(*pe.pad_jaxpr(jaxpr.jaxpr, jaxpr.consts))
  return scan_p.bind(*args, jaxpr=padded_jaxpr, **params)

def _scan_dce_rule(used_outputs: List[bool], eqn: core.JaxprEqn
                   ) -> Tuple[List[bool], core.JaxprEqn]:
  jaxpr = eqn.params['jaxpr']
  num_consts, num_carry = eqn.params['num_consts'], eqn.params['num_carry']
  num_xs = len(jaxpr.in_avals) - num_consts - num_carry
  _, used_carry_out, used_extensive_out = split_list(used_outputs, [3, num_carry])
  for i in range(1 + num_carry):
    used_outputs = [True] + used_carry_out + used_extensive_out
    jaxpr_dce, used_inputs = pe.dce_jaxpr(
        jaxpr.jaxpr, used_outputs,
        instantiate=[False] * num_consts + used_carry_out + [False] * num_xs)
    used_consts, used_carry_in, used_extensive_in = \
        split_list(used_inputs, [num_consts, num_carry])
    if list(used_carry_in) == list(used_carry_out):
      break
    else:
      used_carry_out = _map(operator.or_, used_carry_out, used_carry_in)
  else:
    assert False, "Fixpoint not reached"
  if config.jax_enable_checks: core.check_jaxpr(jaxpr_dce)

  start_stop_done_linear, rest_linear = split_list(eqn.params["linear"], [3])
  new_linear = [l for l, u in zip(rest_linear, used_inputs) if u]
  new_linear = start_stop_done_linear + new_linear
  new_params = dict(eqn.params, num_consts=sum(used_consts),
                    num_carry=sum(used_carry_in), linear=tuple(new_linear),
                    jaxpr=core.ClosedJaxpr(jaxpr_dce, jaxpr.consts))
  # TODO(mattjj,sharadmv): don't assume effects are never DCE'd?
  new_eqn = pe.new_jaxpr_eqn(
      [v for v, used in zip(eqn.invars, [True, True, True] + used_inputs) if used],
      [v for v, used in zip(eqn.outvars, [True, True] + used_outputs) if used],
      eqn.primitive, new_params, eqn.effects, eqn.source_info)
  assert len(new_eqn.invars ) == 3 + len(new_params['jaxpr'].in_avals )
  assert len(new_eqn.outvars) == 2 + len(new_params['jaxpr'].out_avals)
  return [True, True, True] + used_inputs, new_eqn

# TODO(mattjj): de-duplicate code with _scan_partial_eval
def _scan_partial_eval_custom(saveable, unks_in, inst_in, eqn):
  jaxpr = eqn.params['jaxpr']
  num_consts, num_carry = eqn.params['num_consts'], eqn.params['num_carry']
  num_ys = len(jaxpr.out_avals) - 1 - num_carry

  [start_uk, stop_uk, done_uk], const_uk, carry_uk, xs_uk = split_list(unks_in, [3, num_consts, num_carry])
  if done_uk:
    # If conditional is unknown, just do the default processing.
    eqn_known = None
    eqn_staged = eqn
    unks_out = [True] * len(eqn.outvars)
    inst_out = [True] * len(eqn.outvars)
    new_vars = [x for inst, x in zip(inst_in, eqn.invars)
                if type(x) is core.Var and not inst]
    return eqn_known, eqn_staged, unks_out, inst_out, new_vars
  assert not start_uk
  assert not stop_uk

  # Fixpoint (trivial on 'inst_in', since we might as well make all inputs
  # available as DCE can subsequently prune any unused ones)
  for _ in range(1 + len(carry_uk)):
    unks_in = const_uk + carry_uk + xs_uk
    jaxpr_known_, jaxpr_staged_, unks_out, inst_out, num_res = \
        pe.partial_eval_jaxpr_custom(
            jaxpr.jaxpr, in_unknowns=unks_in, in_inst=True,
            ensure_out_unknowns=[False] + carry_uk + [False] * num_ys,
            ensure_out_inst=True, saveable=saveable)
    [done_uk], carry_uk_out, ys_uk = split_list(unks_out, [1, num_carry])
    if carry_uk_out == carry_uk:
      break
    else:
      carry_uk = _map(operator.or_, carry_uk, carry_uk_out)
  else:
    assert False, "Fixpoint not reached"
  if done_uk:
    raise NotImplementedError
  jaxpr_known  = core.ClosedJaxpr(jaxpr_known_ , jaxpr.consts)
  jaxpr_staged = core.ClosedJaxpr(jaxpr_staged_, jaxpr.consts)

  # jaxpr :: (consts, carry, x) -> (done, carry, y)
  # jaxpr_known :: (known_consts, known_carry, known_x) -> (done, known_carry, known_y, residuals)
  # jaxpr_staged :: (residuals, consts, carry, x) -> (done, carry, y)

  # Staged computation does not use early-exit; we use the
  # start and stop output from the known part instead.
  jaxpr_staged.jaxpr.outvars = [_literal_false] + jaxpr_staged.jaxpr.outvars[1:]
  # jaxpr_staged :: (residuals, consts, carry, x) -> (done=False, carry, y)

  # Move all residual binders to the back of jaxpr_staged so they're extensive.
  # TODO(mattjj): make jaxpr_staged only take instantiated inputs
  res_avals = jaxpr_staged.in_avals[:num_res]
  jaxpr_staged = pe.move_binders_to_back(
      jaxpr_staged, [True] * num_res + [False] * len(jaxpr.in_avals))
  # jaxpr_staged :: (consts, carry, x, residuals) -> (done=False, carry, y)

  # Instantiate all inputs (b/c jaxpr_staged takes all inputs, corresponding to
  # passing in_inst argument to partial_eval_jaxpr_custom above).
  new_inst = [x for x, inst in zip(eqn.invars, inst_in)
              if type(x) is core.Var and not inst]
  del inst_in

  # As an optimization, hoist loop-invariant residuals out of the loop rather
  # than using extensive outputs for them. See _scan_partial_eval for comments.
  num_const_known = len(const_uk) - sum(const_uk)
  num_carry_known = len(carry_uk) - sum(carry_uk)
  num_xs_known    = len(   xs_uk) - sum(   xs_uk)
  jaxpr_known_hoist, jaxpr_known_loop, loop_dep, consts_known_lp_avals = \
      pe.partial_eval_jaxpr_nounits(
          jaxpr_known,
          [False] * num_const_known + [True] * (num_carry_known + num_xs_known),
          [True] * (len(unks_out) - sum(unks_out)) + [False] * num_res)
  # jaxpr_known_hoist :: known_consts -> (intensive_res, loop_consts)
  # jaxpr_known_loop :: (loop_consts, known_carry, known_x) -> (done, known_carry, known_y, extensive_res)

  # jaxpr_known_hoist produces intensive residuals followed by the constants for
  # jaxpr_known_loop. We adjust jaxpr_staged to accept intensive res as consts.
  loop_dep_res = loop_dep[len(loop_dep) - num_res:]
  jaxpr_staged = pe.move_binders_to_front(
      jaxpr_staged, [False] * len(jaxpr.in_avals) + _map(operator.not_, loop_dep_res))
  num_intensive_res = len(loop_dep_res) - sum(loop_dep_res)
  del jaxpr_known, loop_dep, num_carry_known, num_xs_known, const_uk
  # jaxpr_staged :: (intensive_res, consts, carry, x, extensive_res) -> (done=False, carry, y)

  # Create residual variables.
  intensive_avals, ext_avals_mapped = partition_list(loop_dep_res, res_avals)
  ext_avals = [core.unmapped_aval(eqn.params['length'], core.no_axis_name, 0, a)
               for a in ext_avals_mapped]
  newvar = core.gensym()
  intensive_res = _map(newvar, intensive_avals)
  extensive_res = _map(newvar, ext_avals)

  # Create known eqn, which is a closed_call_p combining evaluation of
  # jaxpr_known_hoist and a scan of jaxpr_known_loop.
  ins_known, _ = partition_list([False, False, False] + unks_in, eqn.invars)
  out_binders_known, _ = partition_list([False, False] + unks_out, eqn.outvars)
  start_stop_done_binders, out_binders_known = split_list(out_binders_known, [3])
  start_stop_done_binders = [newvar(v.aval) if type(b) is core.DropVar else b
                             for v, b in zip(eqn.invars[:3], start_stop_done_binders)]
  start_binder, stop_binder, done_binder = start_stop_done_binders
  # jaxpr_known_loop takes as input constants output as res by jaxpr_known_hoist
  # (corresponding to consts_known_lp_avals) followed by known carry and xs.
  linear_known_ = [l for l, uk in zip(eqn.params['linear'][3:], unks_in) if not uk]
  _, linear_known_ = split_list(linear_known_, [num_const_known])
  linear_known = [False, False, False] + [False] * len(consts_known_lp_avals) + linear_known_
  params_known = dict(eqn.params, jaxpr=jaxpr_known_loop,
                      num_consts=len(consts_known_lp_avals),
                      num_carry=len(carry_uk)-sum(carry_uk),
                      linear=tuple(linear_known))

  @lu.wrap_init
  def known(*ins_known):
    start_stop_done, consts_known_hoist, ins_known_lp = split_list(ins_known, [3, num_const_known])
    out_hoist = core.jaxpr_as_fun(jaxpr_known_hoist)(*consts_known_hoist)
    intensive_res, consts_known_lp = split_list(out_hoist, [num_intensive_res])
    out_start, out_stop, out_done, *out_loop = scan_p.bind(
        *start_stop_done, *consts_known_lp, *ins_known_lp, **params_known)
    return [out_start, out_stop, out_done, *intensive_res, *out_loop]
  call_jaxpr_, _, call_jaxpr_consts = pe.trace_to_jaxpr_dynamic(
      known, [v.aval for v in ins_known])
  call_jaxpr = core.ClosedJaxpr(call_jaxpr_, call_jaxpr_consts)
  eqn_known = pe.new_jaxpr_eqn(
      ins_known,
      [start_binder, stop_binder, done_binder, *intensive_res, *out_binders_known, *extensive_res],
      core.closed_call_p, dict(call_jaxpr=call_jaxpr), call_jaxpr.effects,
      eqn.source_info)

  # Create the staged eqn.
  linear_staged = ([False, False, False] +
                   [False] * len(intensive_res) +
                   list(eqn.params['linear'][3:]) +
                   [False] * len(extensive_res))
  params_staged = dict(eqn.params, jaxpr=jaxpr_staged,
                       num_consts=len(intensive_res) + eqn.params['num_consts'],
                       linear=tuple(linear_staged))
  assert all(inst_out)  # justifies the use of eqn.outvars on the next line
  out_binders_staged = [core.DropVar(x.aval) for x in eqn.outvars[:3]] + eqn.outvars[3:]
  eqn_staged = pe.new_jaxpr_eqn(
      [start_binder, stop_binder, _literal_false, *intensive_res, *eqn.invars[3:], *extensive_res],
      out_binders_staged, eqn.primitive,
      params_staged, jaxpr_staged.effects,
      eqn.source_info)

  new_vars = [start_binder, stop_binder, done_binder, *new_inst, *intensive_res, *extensive_res]
  return eqn_known, eqn_staged, [False, False] + unks_out, [True] * len(eqn.outvars), new_vars

def _scan_typecheck(bind_time, *in_atoms, reverse, length, num_consts, num_carry,
                    jaxpr, linear, unroll):
  avals = [x.aval for x in in_atoms]
  tc = partial(_typecheck_param, 'scan')
  tc(reverse, 'reverse', 'bool', type(reverse) is bool)
  tc(num_consts, 'num_consts', 'non-negative int',
     type(num_consts) is int and num_consts >= 0)
  tc(num_carry, 'num_carry', 'non-negative int',
     type(num_carry) is int and num_carry >= 0)
  tc(jaxpr, 'jaxpr', 'ClosedJaxpr', type(jaxpr) is core.ClosedJaxpr)
  tc(linear, 'linear', 'tuple of bool',
     type(linear) is tuple and all(type(x) is bool for x in linear))
  tc(unroll, 'unroll', 'positive int', type(unroll) is int and unroll > 0)

  tc(length, 'length', 'non-negative int',
     type(length) is int and length >= 0)

  if len(linear) != len(avals):
    raise core.JaxprTypeError(
      f'scan param linear has length {len(linear)} for {len(avals)} operands')

  [start_aval, stop_aval, done_aval], const_avals, init_avals, x_avals = \
        split_list(avals, [3, num_consts, num_carry])
  const_avals_jaxpr, init_avals_jaxpr, x_avals_jaxpr = split_list(
      jaxpr.in_avals, [num_consts, num_carry])
  _, carry_avals_jaxpr, y_avals_mapped = split_list(jaxpr.out_avals, [1, num_carry])
  x_avals_mapped = _map(partial(core.mapped_aval, length, 0), x_avals)
  y_avals = [core.unmapped_aval(length, core.no_axis_name, 0, a)
             for a in y_avals_mapped]

  if start_aval.shape != () or not np.issubdtype(start_aval, np.integer):
    raise core.JaxprTypeError('scan start not an integer scalar')
  if stop_aval.shape != () or not np.issubdtype(stop_aval, np.integer):
    raise core.JaxprTypeError('scan stop not an integer scalar')
  if done_aval.shape != () or not np.issubdtype(done_aval, np.bool_):
    raise core.JaxprTypeError('scan done not a boolean scalar')
  if not all(_map(core.typematch, init_avals_jaxpr, carry_avals_jaxpr)):
    raise core.JaxprTypeError(
      f'scan input carry input and output types mismatch: '
      f'\n{_avals_short(init_avals_jaxpr)}\nvs\n{_avals_short(carry_avals_jaxpr)}')
  if not all(_map(core.typecompat, const_avals_jaxpr, const_avals)):
    raise core.JaxprTypeError(
      f'scan jaxpr takes input const types\n{_avals_short(const_avals_jaxpr)},\n'
      f'called with consts of type\n{_avals_short(const_avals)}')
  if not all(_map(core.typecompat, init_avals_jaxpr, init_avals)):
    raise core.JaxprTypeError(
      f'scan jaxpr takes input carry types\n{_avals_short(init_avals_jaxpr)},\n'
      f'called with initial carry of type\n{_avals_short(init_avals)}')
  if not all(_map(core.typecompat, x_avals_jaxpr, x_avals_mapped)):
    raise core.JaxprTypeError(
      f'scan jaxpr takes input sequence types\n{_avals_short(x_avals_jaxpr)},\n'
      f'called with sequence of type\n{_avals_short(x_avals)}')
  return [start_aval, stop_aval, done_aval, *init_avals, *y_avals], jaxpr.effects

def _scan_pp_rule(eqn, context, settings):
  printed_params = dict(eqn.params)
  del printed_params['linear']
  if 3 + eqn.params['num_consts'] + eqn.params['num_carry'] == len(eqn.invars):
    del printed_params['length']
  if printed_params['unroll'] == 1:
    del printed_params['unroll']
  if printed_params['num_carry'] == 0:
    del printed_params['num_carry']
  if printed_params['num_consts'] == 0:
    del printed_params['num_consts']
  if not printed_params['reverse']:
    del printed_params['reverse']
  lhs = core.pp_vars(eqn.outvars, context, print_shapes=settings.print_shapes)
  rhs = [pp.text(eqn.primitive.name),
         core.pp_kv_pairs(sorted(printed_params.items()), context, settings),
         pp.text(" ") + core.pp_vars(eqn.invars, context)]
  annotation = (source_info_util.summarize(eqn.source_info)
                if settings.source_info else None)
  return [lhs, pp.text(" = ", annotation=annotation), *rhs]


def scan_bind(*args, **params):
  if config.jax_enable_checks:
    avals = _map(core.get_aval, args)
    in_atoms = [core.Var(0, '', a) for a in avals]  # dummies
    _scan_typecheck(True, *in_atoms, **params)
    core.check_jaxpr(params['jaxpr'].jaxpr)
  return core.AxisPrimitive.bind(scan_p, *args, **params)

scan_p = core.AxisPrimitive("scan")
scan_p.multiple_results = True
scan_p.def_custom_bind(scan_bind)
scan_p.def_impl(partial(xla.apply_primitive, scan_p))
scan_p.def_effectful_abstract_eval(_scan_abstract_eval)
ad.primitive_jvps[scan_p] = _scan_jvp
ad.reducing_transposes[scan_p] = _scan_transpose
pe.custom_partial_eval_rules[scan_p] = _scan_partial_eval
xla.register_initial_style_primitive(scan_p)
mlir.register_lowering(scan_p,
                       mlir.lower_fun(_scan_impl, multiple_results=True))
batching.axis_primitive_batchers[scan_p] = _scan_batching_rule
core.custom_typechecks[scan_p] = partial(_scan_typecheck, False)
pe.partial_eval_jaxpr_custom_rules[scan_p] = _scan_partial_eval_custom
pe.padding_rules[scan_p] = _scan_padding_rule
pe.dce_rules[scan_p] = _scan_dce_rule
# TODO(mattjj,frostig): un-comment this pp rule
# core.pp_eqn_rules[scan_p] = _scan_pp_rule

### while_loop

@api_boundary
def while_loop(cond_fun: Callable[[T], BooleanNumeric],
               body_fun: Callable[[T], T],
               init_val: T,
               max_steps: Optional[int] = None) -> T:
  """Call ``body_fun`` repeatedly in a loop while ``cond_fun`` is True.

  The `Haskell-like type signature`_ in brief is

  .. code-block:: haskell

    while_loop :: (a -> Bool) -> (a -> a) -> a -> a

  The semantics of ``while_loop`` are given by this Python implementation::

    def while_loop(cond_fun, body_fun, init_val):
      val = init_val
      while cond_fun(val):
        val = body_fun(val)
      return val

  Unlike that Python version, ``while_loop`` is a JAX primitive and is lowered
  to a single XLA While HLO. That makes it useful for reducing compilation times
  for jit-compiled functions, since native Python loop constructs in an ``@jit``
  function are unrolled, leading to large XLA computations.

  Also unlike the Python analogue, the loop-carried value ``val`` must hold a
  fixed shape and dtype across all iterations (and not just be consistent up to
  NumPy rank/shape broadcasting and dtype promotion rules, for example). In
  other words, the type ``a`` in the type signature above represents an array
  with a fixed shape and dtype (or a nested tuple/list/dict container data
  structure with a fixed structure and arrays with fixed shape and dtype at the
  leaves).

  Another difference from using Python-native loop constructs is that
  ``while_loop`` is not reverse-mode differentiable (because XLA computations
  require static bounds on memory requirements) unless ``max_steps`` is passed.

  Args:
    cond_fun: function of type ``a -> Bool``.
    body_fun: function of type ``a -> a``.
    init_val: value of type ``a``, a type that can be a scalar, array, or any
      pytree (nested Python tuple/list/dict) thereof, representing the initial
      loop carry value.
    max_steps: a bound on the maximum number of steps. If passed, then after
      this many steps the loop will finish unconditionally. Must be passed for
      ``while_loop`` to be reverse-mode differentiable. Note that memory usage
      will grow proportional to the value of ``max_steps``.

  Returns:
    The output from the final iteration of body_fun, of type ``a``.

  .. _Haskell-like type signature: https://wiki.haskell.org/Type_signature
  """

  if max_steps is not None:
    def f(carry, _):
      return body_fun(carry), None

    def early_exit(carry):
      return lax.bitwise_not(cond_fun(carry))

    final_val, _ = scan(f, init_val, xs=None, length=max_steps, early_exit=early_exit)
    return final_val

  if not (callable(body_fun) and callable(cond_fun)):
    raise TypeError("lax.while_loop: body_fun and cond_fun arguments should be callable.")
  if config.jax_disable_jit:
    try:
      val = init_val
      while cond_fun(val):
        val = body_fun(val)
      return val
    except core.ConcretizationTypeError:
      # Can't run this while_loop in Python (e.g. because there's a vmap
      # transformation on it), so we fall back to the primitive version.
      pass

  def _create_jaxpr(init_val):
    init_vals, in_tree = tree_flatten((init_val,))
    init_avals = tuple(_map(_abstractify, init_vals))
    cond_jaxpr, cond_consts, cond_tree = _initial_style_jaxpr(
        cond_fun, in_tree, init_avals, "while_cond")
    body_jaxpr, body_consts, body_tree = _initial_style_jaxpr(
        body_fun, in_tree, init_avals, "while_loop")
    if not treedef_is_leaf(cond_tree) or len(cond_jaxpr.out_avals) != 1:
      msg = "cond_fun must return a boolean scalar, but got pytree {}."
      raise TypeError(msg.format(cond_tree))
    pred_aval = cond_jaxpr.out_avals[0]
    if (not isinstance(pred_aval, ShapedArray)
        or pred_aval.strip_weak_type().strip_named_shape() != ShapedArray((), np.bool_)):
      msg = "cond_fun must return a boolean scalar, but got output type(s) {}."
      raise TypeError(msg.format(cond_jaxpr.out_avals))
    return init_vals, init_avals, body_jaxpr, in_tree, cond_jaxpr, cond_consts, body_consts, body_tree

  # The body input and output avals must match exactly. However, we want to account for
  # the case when init contains weakly-typed values (e.g. Python scalars), with avals that
  # may not match the output despite being compatible by virtue of their weak type.
  # To do this, we compute the jaxpr in two passes: first with the raw inputs, and if
  # necessary, a second time with modified init values.
  init_vals, init_avals, body_jaxpr, in_tree, *rest = _create_jaxpr(init_val)
  new_init_vals, changed = _promote_weak_typed_inputs(init_vals, init_avals, body_jaxpr.out_avals)
  if changed:
    new_init_val, = tree_unflatten(in_tree, new_init_vals)
    init_vals, init_avals, body_jaxpr, in_tree, *rest = _create_jaxpr(new_init_val)
  cond_jaxpr, cond_consts, body_consts, body_tree = rest

  in_tree_children = in_tree.children()
  assert len(in_tree_children) == 1
  _check_tree_and_avals("body_fun output and input",
                        body_tree, body_jaxpr.out_avals,
                        in_tree_children[0], init_avals)
  effects = core.join_effects(cond_jaxpr.effects, body_jaxpr.effects)
  disallowed_effects = effects - allowed_effects
  if disallowed_effects:
    raise NotImplementedError(
        f'Effects not supported in `while`: {disallowed_effects}')
  outs = while_p.bind(*cond_consts, *body_consts, *init_vals,
                      cond_nconsts=len(cond_consts), cond_jaxpr=cond_jaxpr,
                      body_nconsts=len(body_consts), body_jaxpr=body_jaxpr)
  return tree_unflatten(body_tree, outs)

def _while_loop_abstract_eval(*args, cond_jaxpr, body_jaxpr, **kwargs):
  del args, kwargs
  joined_effects = core.join_effects(cond_jaxpr.effects, body_jaxpr.effects)
  disallowed_effects = joined_effects - allowed_effects
  if disallowed_effects:
    raise NotImplementedError(
        f'Effects not supported in `while`: {disallowed_effects}')
  return _map(raise_to_shaped, body_jaxpr.out_avals), joined_effects


def _while_loop_batching_rule(axis_size, axis_name, main_type, args, dims,
                              cond_nconsts, cond_jaxpr,
                              body_nconsts, body_jaxpr):
  orig_batched = [d is not batching.not_mapped for d in dims]
  cconst_bat, bconst_bat, init_bat = split_list(orig_batched, [cond_nconsts, body_nconsts])
  cconsts, bconsts, init = split_list(args, [cond_nconsts, body_nconsts])
  cconst_dims, bconst_dims, init_dims = split_list(dims, [cond_nconsts, body_nconsts])

  carry_bat = init_bat
  # Fixpoint computation of which carry are batched: either
  # batched from init, or the carry out is batched. Each iteration promotes
  # at least one carry to batched. We need at most len(carry) iterations to
  # reach a fixpoint.
  for _ in range(1 + len(carry_bat)):
    _, carry_bat_out = batching.batch_jaxpr(
        body_jaxpr, axis_size, bconst_bat + carry_bat, instantiate=carry_bat,
        axis_name=axis_name, main_type=main_type)
    if carry_bat == carry_bat_out:
      break
    carry_bat = safe_map(operator.or_, carry_bat, carry_bat_out)
  else:
    assert False, "Fixpoint not reached"

  # Knowing how the carry is batched now, we can determine if the predicate is
  # batched.
  _, (pred_bat,) = batching.batch_jaxpr(
      cond_jaxpr, axis_size, cconst_bat + carry_bat, instantiate=False,
      axis_name=axis_name, main_type=main_type)

  if pred_bat:
    # If the predicate is batched, we have to batch *all* of the carry
    # regardless of if the body needs it.
    carry_bat = [True] * len(carry_bat)
    carry_dims = [0] * len(carry_bat)
    body_jaxpr_batched, _ = batching.batch_jaxpr_axes(
        body_jaxpr, axis_size, bconst_dims + carry_dims,
        carry_dims, axis_name=axis_name, main_type=main_type)
    cond_jaxpr_batched, _ = batching.batch_jaxpr_axes(
        cond_jaxpr, axis_size, cconst_dims + carry_dims, [0],
        axis_name=axis_name, main_type=main_type)
  else:
    # If the predicate is not batched, we can look at the `cond_jaxpr`'s out
    # shape to determine the rank of the predicate. From this rank we pick the
    # dims of the carry to be batched to ensure that the predicate shape is a
    # prefix of the carry in and out shapes. We can then batch the `body_jaxpr`
    # according to these new batch dims.
    cond_rank = len(cond_jaxpr.out_avals[0].shape)
    carry_dims = [cond_rank if b else None for b in carry_bat]
    body_jaxpr_batched, _ = batching.batch_jaxpr_axes(
        body_jaxpr, axis_size, bconst_dims + carry_dims, carry_dims,
        axis_name=axis_name, main_type=main_type)
    # Now we need to rebatch the `cond_jaxpr` according to the new dims of the
    # carry.
    cond_jaxpr_batched, _ = batching.batch_jaxpr_axes(
        cond_jaxpr, axis_size, cconst_dims + carry_dims, (None,),
        axis_name=axis_name, main_type=main_type)

  # To prepare the `init` to the `while_p`, we broadcast values if they are
  # unbatched and need to have an out axis. If their current batch axis does not
  # match the one it needs to be for the translation rule to work, we move it
  # into place.
  new_init = []
  for x, old_axis, new_axis in zip(init, init_dims, carry_dims):
    if old_axis is batching.not_mapped and new_axis is not batching.not_mapped:
      new_init.append(batching.broadcast(x, axis_size, new_axis))
    elif old_axis is batching.not_mapped and new_axis is batching.not_mapped:
      new_init.append(x)
    else:
      assert new_axis is not batching.not_mapped
      new_init.append(batching.moveaxis(x, old_axis, new_axis))

  outs = while_p.bind(*(cconsts + bconsts + new_init),
                      cond_nconsts=cond_nconsts, cond_jaxpr=cond_jaxpr_batched,
                      body_nconsts=body_nconsts, body_jaxpr=body_jaxpr_batched)
  return outs, carry_dims

def _while_loop_jvp(primals, tangents, cond_nconsts, cond_jaxpr, body_nconsts,
                    body_jaxpr):
  nonzeros = [type(t) is not ad_util.Zero for t in tangents]
  cconst_nz, bconst_nz, init_nz = split_list(nonzeros, [cond_nconsts, body_nconsts])

  carry_nz = init_nz
  for _ in range(1 + len(carry_nz)):
    body_nonzeros = bconst_nz + carry_nz
    body_jvp, nonzeros_out = ad.jvp_jaxpr(
        body_jaxpr, body_nonzeros, instantiate=carry_nz)
    if nonzeros_out == carry_nz:
      break
    carry_nz = _map(operator.or_, carry_nz, nonzeros_out)
  else:
    assert False, "Fixpoint not reached"

  nonzeros = cconst_nz + body_nonzeros
  tangents = [ad.instantiate_zeros(t) if nz else t
              for t, nz in zip(tangents, nonzeros)]

  cconst, bconst, init = split_list(primals, [cond_nconsts, body_nconsts])
  _, bconst_dot, init_dot = split_list(tangents, [cond_nconsts, body_nconsts])
  bconst_dot = _prune_zeros(bconst_dot)
  init_dot = _prune_zeros(init_dot)

  num_carry = len(primals) - cond_nconsts - body_nconsts

  body_jvp_rearranged = ad.rearrange_binders(
      body_jvp,
      [body_nconsts, num_carry], [len(bconst_dot), len(init_dot)],
      [num_carry], [len(init_dot)])

  newvar = core.gensym([cond_jaxpr.jaxpr])
  invars_aug = (
      cond_jaxpr.jaxpr.invars + [newvar(core.get_aval(x)) for x in init_dot])
  cond_jaxpr_augmented = core.Jaxpr(cond_jaxpr.jaxpr.constvars,
                                    invars_aug,
                                    cond_jaxpr.jaxpr.outvars,
                                    cond_jaxpr.jaxpr.eqns,
                                    cond_jaxpr.jaxpr.effects)
  cond_jaxpr_augmented = core.ClosedJaxpr(cond_jaxpr_augmented, cond_jaxpr.consts)

  out = while_p.bind(
      *(cconst + bconst + bconst_dot + init + init_dot),
      cond_nconsts=cond_nconsts,
      cond_jaxpr=cond_jaxpr_augmented,
      body_nconsts=len(bconst) + len(bconst_dot),
      body_jaxpr=body_jvp_rearranged)

  out_carry, out_carry_dot = split_list(out, [num_carry])
  out_tangents_iter = iter(out_carry_dot)
  out_tangents = [next(out_tangents_iter) if nz else ad_util.Zero.from_value(p)
                  for p, nz in zip(out_carry, nonzeros_out)]
  return out_carry, out_tangents

def _while_partial_eval(trace: pe.JaxprTrace, *tracers: pe.Tracer, cond_nconsts: int,
                        cond_jaxpr: pe.ClosedJaxpr, body_nconsts: int,
                        body_jaxpr: pe.ClosedJaxpr) -> Sequence[pe.Tracer]:
  # As long as some carry (and hence output) are known and the output of
  # `cond_jaxpr` is known, we use a portion of the loop body to compute the
  # known outputs of the `while_loop`. For the unknown outputs we generate a
  # jaxpr to run the whole while, including recomputing the known parts,
  # basically like building in checkpointing/rematieralization. This means that
  # we don't actually save any computation by partial evaluation if there are
  # unknown outputs.
  #
  # What this achieves is twofold: jax.linearize works, and we can give a proper
  # error for reverse differentiation of `while`.

  unknowns = [not t.pval.is_known() for t in tracers]
  params = dict(cond_nconsts=cond_nconsts, cond_jaxpr=cond_jaxpr,
                body_nconsts=body_nconsts, body_jaxpr=body_jaxpr)

  cond_consts_uk, body_consts_uk, carry_init_uk = \
      split_list(unknowns, [cond_nconsts, body_nconsts])

  # Fixpoint computation of unknown carry. Each iteration promotes at least one
  # carry to unknown. We need one last iteration to prepare the jaxpr.
  carry_uk = carry_init_uk
  for _ in range(1 + len(carry_uk)):
    body_jaxpr_known, _, carry_out_uk, body_res_avals = pe.partial_eval_jaxpr_nounits(  # type: ignore
        body_jaxpr, body_consts_uk + carry_uk, instantiate=carry_uk)
    if carry_out_uk == carry_uk:
      break
    else:
      carry_uk = _map(operator.or_, carry_uk, carry_out_uk)
  else:
    assert False, "Fixpoint not reached"

  cond_jaxpr_known, _, cond_uk, _ = pe.partial_eval_jaxpr_nounits(  # type: ignore
      cond_jaxpr, cond_consts_uk + carry_uk, instantiate=False)

  if cond_uk[0] or all([not uk for uk in unknowns]) or all(unknowns):
    # If conditional is unknown, or all inputs are known, or all are unknown,
    # just do the default processing.
    return trace.default_process_primitive(while_p, tracers, params)

  # Run the known part of the while.
  in_consts = [t.pval.get_known() for uk, t in
               zip(cond_consts_uk + body_consts_uk + carry_uk, tracers)
               if not uk]
  cond_nconsts_known = len(cond_consts_uk) - sum(cond_consts_uk)
  body_nconsts_known = len(body_consts_uk) - sum(body_consts_uk)
  num_known_outs = len(carry_uk) - sum(carry_uk)
  # TODO(mattjj): use pe.dce_jaxpr to drop res computations and not just outputs
  body_jaxpr_known.jaxpr.outvars = body_jaxpr_known.jaxpr.outvars[:num_known_outs]
  out_known = while_p.bind(
      *in_consts, cond_nconsts=cond_nconsts_known, cond_jaxpr=cond_jaxpr_known,
      body_nconsts=body_nconsts_known, body_jaxpr=body_jaxpr_known)
  del body_jaxpr_known

  # Run the whole while_loop to get all the outputs, then merge with known ones
  out_tracers_ = trace.default_process_primitive(while_p, tracers, params)
  out_tracers = [t for t, uk in zip(out_tracers_, carry_uk) if uk]
  return util.merge_lists(carry_uk, out_known, out_tracers)

# TODO(mattjj): de-duplicate code with _while_partial_eval
def _while_partial_eval_custom(saveable, unks_in, inst_in, eqn):
  del saveable  # We can't save any residuals anyway (w/o dynamic shapes)!
  cond_jaxpr = eqn.params['cond_jaxpr']
  cond_nconsts = eqn.params['cond_nconsts']
  body_jaxpr = eqn.params['body_jaxpr']
  body_nconsts = eqn.params['body_nconsts']

  cond_consts_uk, body_consts_uk, carry_init_uk = \
      split_list(unks_in, [cond_nconsts, body_nconsts])

  # Fixpoint to compute known part of the body (trivial on 'inst_in', since we
  # make all inputs available as DCE can subsequently prune any unused ones)
  carry_uk = carry_init_uk
  for _ in range(1 + len(carry_uk)):
    body_unks_in = body_consts_uk + carry_uk
    jaxpr_known_, _, carry_uk_out, _, num_res = \
        pe.partial_eval_jaxpr_custom(
            body_jaxpr.jaxpr, in_unknowns=body_unks_in, in_inst=True,
            ensure_out_unknowns=carry_uk, ensure_out_inst=True,
            saveable=ad_checkpoint.nothing_saveable)
    if carry_uk_out == carry_uk:
      break
    else:
      carry_uk = _map(operator.or_, carry_uk, carry_uk_out)
  else:
    assert False, "Fixpoint not reached"
  assert not num_res
  body_jaxpr_known = core.ClosedJaxpr(jaxpr_known_, body_jaxpr.consts)
  del jaxpr_known_, carry_uk_out, num_res

  # Instantiate all inputs (b/c jaxpr_staged will take all inputs).
  new_inst = [x for x, inst in zip(eqn.invars, inst_in)
              if type(x) is core.Var and not inst]

  # Compute the known part of cond_fun (basically pruning inputs on known side).
  cond_unks_in = cond_consts_uk + carry_uk
  cond_jaxpr_known_, _, [cond_uk], _, _ = \
      pe.partial_eval_jaxpr_custom(
          cond_jaxpr.jaxpr, cond_unks_in, in_inst=True,
          ensure_out_unknowns=False, ensure_out_inst=True,
          saveable=ad_checkpoint.nothing_saveable)
  # NOTE(mattjj): I think it should be impossible for the condition to be
  # unknown, but asserting that caused a test failure in diffrax. So
  # we handle it: if it is unknown, stage out the whole cond function.
  if cond_uk:
    return None, eqn, [True] * len(carry_uk), [True] * len(carry_uk), new_inst
  cond_jaxpr_known = core.ClosedJaxpr(cond_jaxpr_known_, cond_jaxpr.consts)
  del cond_uk

  # Build the known eqn.
  ins_known, _ = partition_list(unks_in, eqn.invars)
  out_binders_known, _ = partition_list(carry_uk, eqn.outvars)
  params_known = dict(cond_jaxpr=cond_jaxpr_known, body_jaxpr=body_jaxpr_known,
                      cond_nconsts=len(cond_consts_uk) - sum(cond_consts_uk),
                      body_nconsts=len(body_consts_uk) - sum(body_consts_uk))
  effects_known = core.join_effects(cond_jaxpr_known.effects,
                                    body_jaxpr_known.effects)
  eqn_known = pe.new_jaxpr_eqn(ins_known, out_binders_known, while_p,
                               params_known, effects_known, eqn.source_info)

  # Staged eqn is same as input eqn.
  eqn_staged = eqn

  unks_out = carry_uk
  inst_out = [True] * len(unks_out)
  return eqn_known, eqn_staged, unks_out, inst_out, new_inst

def _while_transpose_error(*_, **kwargs):
  raise ValueError("Reverse-mode differentiation does not work for "
                   "lax.while_loop or lax.fori_loop. "
                   "Try using lax.scan instead.")

# For a while loop with ordered effects in the cond, we need a special
# lowering. Fundamentally, we'd like to rewrite a while loop that looks like
# this:
# ```
# while cond(x):
#   x = body(x)
# ```
# into something that looks like this:
# ```
# while True:
#   token, pred = cond(token, x)
#   if not pred:
#     break
#   token, x = body(token, x)
# ```
# Unfortunately, with an MHLO while we can't (1) return multiple values
# from a `cond` and (2) can't break a while loop. We thus adopt the
# following rewrite strategy:
# ```
# def new_cond(pred, token, x):
#   return pred
# token, pred = cond(token, x)
# while new_cond(pred, token, x):
#   token, x = body(token, x)
#   token, pred = cond(token, x)
# ```
def _while_lowering(ctx, *args, cond_jaxpr, body_jaxpr, cond_nconsts,
                    body_nconsts):
  pred_aval = cond_jaxpr.out_avals[0]
  batched = bool(pred_aval.shape)
  cond_ordered_effects = [eff for eff in cond_jaxpr.effects if eff in
                          core.ordered_effects]
  if cond_ordered_effects:
    def cond(args):
      # Pred can be batched
      pred = core.eval_jaxpr(cond_jaxpr.jaxpr, cond_jaxpr.consts, *args)[0]
      if batched:
        pred = lax._reduce_or(pred, tuple(range(len(pred_aval.shape))))
      return pred
    def body(args):
      return tuple(core.eval_jaxpr(body_jaxpr.jaxpr, body_jaxpr.consts, *args))
    def new_cond(pred_args):
      pred, _ = pred_args
      return pred
    def new_body(pred_args):
      _, args  = pred_args
      args = body(args)
      pred = cond(args)
      return pred, args
    def fun(*args):
      pred = cond(args)
      _, out = while_loop(new_cond, new_body, (pred, args))
      return out
    return mlir.lower_fun(fun)(ctx, *args)

  loop_carry_types = _map(mlir.aval_to_ir_types, ctx.avals_in)
  body_effects = [eff for eff in body_jaxpr.effects
                  if eff in core.ordered_effects]
  num_tokens = len(body_effects)
  tokens = [ctx.tokens_in.get(eff) for eff in body_effects]
  token_types = [mlir.token_type() for _ in tokens]
  loop_carry_types = [*token_types, *loop_carry_types]
  flat_loop_carry_types = util.flatten(loop_carry_types)
  args = [*tokens, *args]

  flat_args = mlir.flatten_lowering_ir_args(args)
  while_op = mhlo.WhileOp(flat_loop_carry_types, flat_args)

  # Loop condition
  cond_block = while_op.regions[0].blocks.append(*flat_loop_carry_types)
  name_stack = extend_name_stack(ctx.module_context.name_stack, 'while')
  with ir.InsertionPoint(cond_block):
    flat_cond_args = [
        cond_block.arguments[i] for i in range(len(flat_loop_carry_types))
    ]
    cond_args = util.unflatten(flat_cond_args, _map(len, loop_carry_types))
    # Remove tokens from cond args
    cond_args = cond_args[num_tokens:]
    x, _, z = util.split_list(cond_args, [cond_nconsts, body_nconsts])
    cond_ctx = ctx.module_context.replace(
        name_stack=xla.extend_name_stack(name_stack, 'cond'))
    ((pred,),), _ = mlir.jaxpr_subcomp(cond_ctx, cond_jaxpr.jaxpr, mlir.TokenSet(),
                                    _map(mlir.ir_constants, cond_jaxpr.consts),
                                    *(x + z))
    if batched:
      pred_ctx = mlir.LoweringRuleContext(
          module_context=ctx.module_context,
          primitive=None,
          avals_in=[pred_aval],
          avals_out=[pred_aval.update(shape=())],
          tokens_in=mlir.TokenSet(),
          tokens_out=None)
      pred, = lax._unary_reduce_lower(
          mhlo.OrOp,
          lambda dtype: np.array(False, dtype),
          pred_ctx,
          pred,
          axes=tuple(range(len(pred_aval.shape))))
    mhlo.ReturnOp([pred])

  # Loop body
  body_block = while_op.regions[1].blocks.append(*flat_loop_carry_types)
  with ir.InsertionPoint(body_block):
    flat_body_args = [
        body_block.arguments[i] for i in range(len(flat_loop_carry_types))
    ]
    body_args = util.unflatten(flat_body_args, _map(len, loop_carry_types))
    # Tokens are at the front of the args list to the while loop
    token_args, body_args = util.split_list(body_args, [num_tokens])
    tokens_in = mlir.TokenSet(zip(body_effects, token_args))
    x, y, z = util.split_list(body_args, [cond_nconsts, body_nconsts])
    body_ctx = ctx.module_context.replace(
        name_stack=xla.extend_name_stack(name_stack, 'body'))
    new_z, tokens_out = mlir.jaxpr_subcomp(body_ctx, body_jaxpr.jaxpr,
        tokens_in, _map(mlir.ir_constants, body_jaxpr.consts), *(y + z))
    out_tokens = [tokens_out.get(eff) for eff in body_effects]
    if batched:
      body_pred_ctx = ctx.module_context.replace(
          name_stack=xla.extend_name_stack(name_stack,
                                           'body_pred'))
      ((body_pred,),), _ = mlir.jaxpr_subcomp(
          body_pred_ctx, cond_jaxpr.jaxpr, mlir.TokenSet(),
          _map(mlir.ir_constants, cond_jaxpr.consts), *(x + z))
      new_z = _map(
          partial(_pred_bcast_select_mhlo, pred_aval, body_pred), new_z, z,
          body_jaxpr.out_avals)

    mhlo.ReturnOp([*util.flatten(out_tokens), *util.flatten(x),
                   *util.flatten(y), *util.flatten(new_z)])

  outputs = util.unflatten(while_op.results, _map(len, loop_carry_types))
  tokens, _, _, z = util.split_list(outputs, [num_tokens, cond_nconsts, body_nconsts])
  if tokens:
    ctx.set_tokens_out(mlir.TokenSet(zip(body_effects, tokens)))
  return z

def _while_typecheck(*in_atoms, cond_jaxpr, body_jaxpr, cond_nconsts,
                     body_nconsts):
  # TODO(frostig,mattjj): check cond_jaxpr, body_jaxpr types
  joined_effects = core.join_effects(cond_jaxpr.effects, body_jaxpr.effects)
  if joined_effects - allowed_effects:
    raise NotImplementedError(
        f'Effects not supported in `while`: {joined_effects - allowed_effects}')
  return body_jaxpr.out_avals, joined_effects

while_p = core.AxisPrimitive('while')
while_p.multiple_results = True
while_p.def_impl(partial(xla.apply_primitive, while_p))
while_p.def_effectful_abstract_eval(_while_loop_abstract_eval)
ad.primitive_jvps[while_p] = _while_loop_jvp
pe.custom_partial_eval_rules[while_p] = _while_partial_eval
xla.register_initial_style_primitive(while_p)
ad.primitive_transposes[while_p] = _while_transpose_error
batching.axis_primitive_batchers[while_p] = _while_loop_batching_rule
pe.partial_eval_jaxpr_custom_rules[while_p] = _while_partial_eval_custom
mlir.register_lowering(while_p, _while_lowering)
core.custom_typechecks[while_p] = _while_typecheck


def _pred_bcast_select_mhlo(
    pred_aval: core.ShapedArray, pred: ir.Value, xs: Sequence[ir.Value],
    ys: Sequence[ir.Value], x_y_aval: core.AbstractValue) -> Sequence[ir.Value]:
  if x_y_aval is core.abstract_token:
    x, = xs
    y, = ys
    return [mhlo.AfterAllOp(mlir.aval_to_ir_type(x_y_aval), [x, y]).result]
  else:
    assert isinstance(x_y_aval, core.ShapedArray), x_y_aval
    x, = xs
    y, = ys
    assert x.type == y.type, (x.type, y.type)
    assert (pred_aval.shape == x_y_aval.shape[:len(pred_aval.shape)]), (
            pred_aval.shape, x_y_aval)
    x_y_type = mlir.aval_to_ir_type(x_y_aval)
    bcast_pred_type = ir.RankedTensorType.get(
        x_y_type.shape, mlir.dtype_to_ir_type(np.dtype(np.bool_)))
    bcast_pred = mhlo.BroadcastInDimOp(
        bcast_pred_type, pred,
        mlir.dense_int_elements(list(range(len(pred_aval.shape))))).result
    return mhlo.SelectOp(bcast_pred, x, y).results

### fori_loop

def _fori_cond_fun(loop_carry):
  i, upper, _ = loop_carry
  return lax.lt(i, upper)

@weakref_lru_cache
def _fori_body_fun(body_fun):
  body_fun = weakref.ref(body_fun)
  def while_body_fun(loop_carry):
    i, upper, x = loop_carry
    return lax.add(i, lax._const(i, 1)), upper, body_fun()(i, x)
  return while_body_fun

@weakref_lru_cache
def _fori_scan_body_fun(body_fun):
  body_fun = weakref.ref(body_fun)
  def scanned_fun(loop_carry, _):
    i, x = loop_carry
    return (i + 1, body_fun()(i, x)), None
  return scanned_fun

@api_boundary
def fori_loop(lower, upper, body_fun, init_val):
  """Loop from ``lower`` to ``upper`` by reduction to :func:`jax.lax.while_loop`.

  The `Haskell-like type signature`_ in brief is

  .. code-block:: haskell

    fori_loop :: Int -> Int -> ((Int, a) -> a) -> a -> a

  The semantics of ``fori_loop`` are given by this Python implementation::

    def fori_loop(lower, upper, body_fun, init_val):
      val = init_val
      for i in range(lower, upper):
        val = body_fun(i, val)
      return val

  As the Python version suggests, setting ``upper <= lower`` will produce no
  iterations. Negative or custom increments are not supported.

  Unlike that Python version, ``fori_loop`` is implemented in terms of either a
  call to :func:`jax.lax.while_loop` or a call to :func:`jax.lax.scan`. If the
  trip count is static (meaning known at tracing time, perhaps because ``lower``
  and ``upper`` are Python integer literals) then the ``fori_loop`` is
  implemented in terms of ``scan`` and reverse-mode autodiff is supported;
  otherwise, a ``while_loop`` is used and reverse-mode autodiff is not
  supported.  See those functions' docstrings for more information.

  Also unlike the Python analogue, the loop-carried value ``val`` must hold a
  fixed shape and dtype across all iterations (and not just be consistent up to
  NumPy rank/shape broadcasting and dtype promotion rules, for example). In
  other words, the type ``a`` in the type signature above represents an array
  with a fixed shape and dtype (or a nested tuple/list/dict container data
  structure with a fixed structure and arrays with fixed shape and dtype at the
  leaves).

  Args:
    lower: an integer representing the loop index lower bound (inclusive)
    upper: an integer representing the loop index upper bound (exclusive)
    body_fun: function of type ``(int, a) -> a``.
    init_val: initial loop carry value of type ``a``.

  Returns:
    Loop value from the final iteration, of type ``a``.

  .. _Haskell-like type signature: https://wiki.haskell.org/Type_signature
  """
  if not callable(body_fun):
    raise TypeError("lax.fori_loop: body_fun argument should be callable.")
  # TODO(phawkins): perhaps do more type checking here, better error messages.
  lower_dtype = dtypes.canonicalize_dtype(lax.dtype(lower))
  upper_dtype = dtypes.canonicalize_dtype(lax.dtype(upper))
  if lower_dtype != upper_dtype:
    msg = ("lower and upper arguments to fori_loop must have equal types, "
           "got {} and {}")
    raise TypeError(msg.format(lower_dtype.name, upper_dtype.name))

  # If we can specialize on the trip count, call scan instead of a while_loop
  # to enable efficient reverse-mode differentiation.
  if (isinstance(core.get_aval(lower), ConcreteArray) and
      isinstance(core.get_aval(upper), ConcreteArray)):
    try:
      lower_ = int(lower)
      upper_ = int(upper)
    except TypeError:
      use_scan = False
    else:
      use_scan = True
  else:
    use_scan = False

  if use_scan:
    if config.jax_disable_jit and upper_ == lower_:
      # non-jit implementation of scan does not support length=0
      return init_val

    (_, result), _ = scan(_fori_scan_body_fun(body_fun), (lower_, init_val),
                          None, length=upper_ - lower_)
  else:
    _, _, result = while_loop(_fori_cond_fun, _fori_body_fun(body_fun),
                              (lower, upper, init_val))
  return result

### map and miscellanous rules

@api_boundary
def map(f, xs):
  """Map a function over leading array axes.

  Like Python's builtin map, except inputs and outputs are in the form of
  stacked arrays. Consider using the ``jax.vmap`` transform instead, unless you
  need to apply a function element by element for reduced memory usage or
  heterogeneous computation with other control flow primitives.

  When ``xs`` is an array type, the semantics of ``map`` are given by this
  Python implementation::

    def map(f, xs):
      return np.stack([f(x) for x in xs])

  Like ``scan``, ``map`` is implemented in terms of JAX primitives so many of
  the same advantages over a Python loop apply: ``xs`` may be an arbitrary
  nested pytree type, and the mapped computation is compiled only once.

  Args:
    f: a Python function to apply element-wise over the first axis or axes of
      ``xs``.
    xs: values over which to map along the leading axis.

  Returns:
    Mapped values.
  """
  g = lambda _, x: ((), f(x))
  _, ys = scan(g, (), xs)
  return ys

def _rng_bit_generator_batching_rule(batched_args, batch_dims, *, shape, dtype, algorithm):
  """Calls RBG in a loop and stacks the results."""
  key, = batched_args
  bd, = batch_dims
  if bd is batching.not_mapped:
    return lax.rng_bit_generator_p.bind(key, shape=shape, dtype=dtype,
                                        algorithm=algorithm), (None, None)
  key = batching.moveaxis(key, bd, 0)
  map_body = lambda k: lax.rng_bit_generator_p.bind(k, shape=shape, dtype=dtype, algorithm=algorithm)
  stacked_keys, stacked_bits = map(map_body, key)
  return (stacked_keys, stacked_bits), (0, 0)

batching.primitive_batchers[lax.rng_bit_generator_p] = _rng_bit_generator_batching_rule  # type: ignore

### associative_scan

@api_boundary
def associative_scan(fn: Callable, elems, reverse: bool = False, axis: int = 0):
  """Performs a scan with an associative binary operation, in parallel.

  For an introduction to associative scans, see [BLE1990]_.

  Args:
    fn: A Python callable implementing an associative binary operation with
      signature ``r = fn(a, b)``. Function `fn` must be associative, i.e., it
      must satisfy the equation
      ``fn(a, fn(b, c)) == fn(fn(a, b), c)``.

      The inputs and result are (possibly nested Python tree structures of)
      array(s) matching ``elems``. Each array has a dimension in place
      of the ``axis`` dimension. `fn` should be applied elementwise over
      the ``axis`` dimension (for example, by using :func:`jax.vmap` over the
      elementwise function.)

      The result ``r`` has the same shape (and structure) as the two inputs
      ``a`` and ``b``.
    elems: A (possibly nested Python tree structure of) array(s), each with
      an ``axis`` dimension of size ``num_elems``.
    reverse: A boolean stating if the scan should be reversed with respect to
      the ``axis`` dimension.
    axis: an integer identifying the axis over which the scan should occur.

  Returns:
    A (possibly nested Python tree structure of) array(s) of the same shape
    and structure as ``elems``, in which the ``k``'th element of ``axis`` is the
    result of recursively applying ``fn`` to combine the first ``k`` elements
    of ``elems`` along ``axis``. For example, given ``elems = [a, b, c, ...]``,
    the result would be ``[a, fn(a, b), fn(fn(a, b), c), ...]``.

  Example 1: partial sums of an array of numbers:

  >>> lax.associative_scan(jnp.add, jnp.arange(0, 4))
  DeviceArray([0, 1, 3, 6], dtype=int32)

  Example 2: partial products of an array of matrices

  >>> mats = jax.random.uniform(jax.random.PRNGKey(0), (4, 2, 2))
  >>> partial_prods = lax.associative_scan(jnp.matmul, mats)
  >>> partial_prods.shape
  (4, 2, 2)

  Example 3: reversed partial sums of an array of numbers

  >>> lax.associative_scan(jnp.add, jnp.arange(0, 4), reverse=True)
  DeviceArray([6, 6, 5, 3], dtype=int32)

  .. [BLE1990] Blelloch, Guy E. 1990. "Prefix Sums and Their Applications.",
    Technical Report CMU-CS-90-190, School of Computer Science, Carnegie Mellon
    University.
  """
  if not callable(fn):
    raise TypeError("lax.associative_scan: fn argument should be callable.")
  elems_flat, tree = tree_flatten(elems)

  if reverse:
    elems_flat = [lax.rev(elem, [axis]) for elem in elems_flat]

  def combine(a_flat, b_flat):
    # Lower `fn` to operate on flattened sequences of elems.
    a = tree_unflatten(tree, a_flat)
    b = tree_unflatten(tree, b_flat)
    c = fn(a, b)
    c_flat, _ = tree_flatten(c)
    return c_flat

  # Check that all inputs have a consistent leading dimension `num_elems`.
  axis = util.canonicalize_axis(axis, elems_flat[0].ndim)
  num_elems = int(elems_flat[0].shape[axis])
  if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
    raise ValueError('Array inputs to associative_scan must have the same '
                     'first dimension. (saw: {})'
                     .format([elem.shape for elem in elems_flat]))


  # Summary of algorithm:
  #
  # Consider elements of `_scan(elems)` at odd indices. That's the same as first
  # summing successive pairs of elements of `elems` and performing a scan on
  # that half sized tensor. We perform the latter scan by recursion.
  #
  # Now consider the even elements of `_scan(elems)`. These can be computed
  # from the odd elements of `_scan(elems)` by adding each odd element of
  # `_scan(elems)` to the matching even element in the original `elems`.
  #
  # We return the odd and even elements interleaved.
  #
  # For the base case of the recursion we return the first element
  # of `elems` followed by the sum of the first two elements computed as
  # a (small two-down-to-one) reduction step.
  def _scan(elems):
    """Perform scan on `elems`."""

    num_elems = elems[0].shape[axis]

    if num_elems < 2:
      return elems

    # Combine adjacent pairs of elements.
    reduced_elems = combine(
      [slicing.slice_in_dim(elem, 0, -1, stride=2, axis=axis) for elem in elems],
      [slicing.slice_in_dim(elem, 1, None, stride=2, axis=axis)
       for elem in elems])

    # Recursively compute scan for partially reduced tensors.
    odd_elems = _scan(reduced_elems)

    if num_elems % 2 == 0:
      even_elems = combine(
        [slicing.slice_in_dim(e, 0, -1, axis=axis) for e in odd_elems],
        [slicing.slice_in_dim(e, 2, None, stride=2, axis=axis) for e in elems])
    else:
      even_elems = combine(
        odd_elems,
        [slicing.slice_in_dim(e, 2, None, stride=2, axis=axis) for e in elems])

    # The first element of a scan is the same as the first element
    # of the original `elems`.
    even_elems = [
      lax.concatenate([slicing.slice_in_dim(elem, 0, 1, axis=axis), result],
                      dimension=axis)
      for (elem, result) in zip(elems, even_elems)]
    return list(_map(partial(_interleave, axis=axis), even_elems, odd_elems))

  scans = _scan(elems_flat)

  if reverse:
    scans = [lax.rev(scanned, [axis]) for scanned in scans]

  return tree_unflatten(tree, scans)

def _interleave(a, b, axis):
  """Given two Tensors of static shape, interleave them along the first axis."""
  assert a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
  a_pad = [(0, 0, 0)] * a.ndim
  b_pad = [(0, 0, 0)] * b.ndim
  a_pad[axis] = (0, 1 if a.shape[axis] == b.shape[axis] else 0, 1)
  b_pad[axis] = (1, 0 if a.shape[axis] == b.shape[axis] else 1, 1)
  op = lax.bitwise_or if a.dtype == np.bool_ else lax.add
  return op(lax.pad(a, lax._const(a, 0), a_pad),
            lax.pad(b, lax._const(b, 0), b_pad))

### Cumulative reductions.

def cumsum(operand: Array, axis: int = 0, reverse: bool = False) -> Array:
  """Computes a cumulative sum along `axis`."""
  return cumsum_p.bind(operand, axis=int(axis), reverse=bool(reverse))

def cumprod(operand: Array, axis: int = 0, reverse: bool = False) -> Array:
  """Computes a cumulative product along `axis`."""
  return cumprod_p.bind(operand, axis=int(axis), reverse=bool(reverse))

def cummax(operand: Array, axis: int = 0, reverse: bool = False) -> Array:
  """Computes a cumulative maximum along `axis`."""
  return cummax_p.bind(operand, axis=int(axis), reverse=bool(reverse))

def cummin(operand: Array, axis: int = 0, reverse: bool = False) -> Array:
  """Computes a cumulative minimum along `axis`."""
  return cummin_p.bind(operand, axis=int(axis), reverse=bool(reverse))

def _cumred_shape_rule(x, *, axis: int, reverse: bool):
  if axis < 0 or axis >= x.ndim:
    raise ValueError(
        f"axis {axis} is out of bounds for array of shape {x.shape}")
  return x.shape

def _cumsum_transpose_rule(t, operand, *, axis: int, reverse: bool):
  return [cumsum(t, axis=axis, reverse=not reverse)]



def cumred_reduce_window_impl(window_reduce: Callable, x, *, axis: int,
                              reverse: bool):
  n = x.shape[axis]
  if n == 0:
    return x
  padding = [(0, 0)] * x.ndim
  padding[axis] = (0, n - 1) if reverse else (n - 1, 0)
  strides = [1] * x.ndim
  window_dims = [1] * x.ndim
  window_dims[axis] = n
  return window_reduce(x, window_dims, strides, padding)


def cumred_gpu_impl(window_reduce: Callable, reduce_fn: Callable, x, *,
                    axis: int, reverse: bool):
  # On GPU, reduce_window is executed in a single fusion and associative_scan
  # is split into multiple to materialize intermediate calculations.
  # On small inputs reduce_window is faster being a single fusion,
  # but on larger ones is slower because of O(n^2) complexity.
  # This conservative value of the threshold was obtained via benchmarking.
  if x.shape[axis] > 32:
    return associative_scan(reduce_fn, x, reverse=reverse, axis=axis)
  return cumred_reduce_window_impl(window_reduce, x, axis=axis, reverse=reverse)


def _cumred_batch_rule(prim, batched_args, batch_dims, *, axis: int,
                       reverse: bool):
  operand, = batched_args
  bdim, = batch_dims
  axis = axis if axis < bdim else axis + 1
  return prim.bind(operand, axis=axis, reverse=reverse), bdim

def _cumred_dtype_rule(name, operand, *args, **kw):
  if not dtypes.issubdtype(operand.dtype, np.number):
    raise TypeError("{} does not accept dtype {}. Accepted dtypes are subtypes "
                    "of number.".format(name, np.dtype(operand.dtype).name))
  return dtypes.canonicalize_dtype(operand.dtype)


def _cumulative_reduction_primitive(name, reduce_fn, reduce_window_fn):
  reducer_p = lax.standard_primitive(
    _cumred_shape_rule, partial(_cumred_dtype_rule, name),
    name)
  batching.primitive_batchers[reducer_p] = partial(_cumred_batch_rule,
                                                   reducer_p)

  def register_lowering(fn, platform=None):
    mlir.register_lowering(
        reducer_p,
        mlir.cache_lowering(mlir.lower_fun(fn, multiple_results=False)),
        platform=platform)

  # Default for platforms not treated specially below.
  register_lowering(partial(associative_scan, reduce_fn))
  # On GPU, we choose between window reduction and associative scan
  # based on the input size.
  for platform in ['cuda', 'rocm']:
    register_lowering(
        partial(cumred_gpu_impl, reduce_window_fn, reduce_fn), platform)
  # On TPU, an implementation using reduce_window is handled specially by the
  # compiler and is efficient. On other backends, it is O(n^2).
  register_lowering(partial(cumred_reduce_window_impl, reduce_window_fn), 'tpu')
  return reducer_p

cumsum_p = _cumulative_reduction_primitive("cumsum", lax.add, windowed_reductions._reduce_window_sum)
ad.deflinear2(cumsum_p, _cumsum_transpose_rule)
cumprod_p = _cumulative_reduction_primitive("cumprod", lax.mul, windowed_reductions._reduce_window_prod)
cummax_p = _cumulative_reduction_primitive("cummax", lax.max, windowed_reductions._reduce_window_max)
cummin_p = _cumulative_reduction_primitive("cummin", lax.min, windowed_reductions._reduce_window_min)


def _cumulative_jvp_rule(primals, tangents, *, axis: int, reverse: bool,
                         combine_fn: Callable):
  # Irrespective of backend, we always use the parallel prefix scan
  # implementation when differentiating because reduce_window is not
  # arbitrarily differentiable.
  return api.jvp(partial(associative_scan, combine_fn, axis=axis,
                         reverse=reverse),
                 primals, tangents)

ad.primitive_jvps[cumprod_p] = partial(_cumulative_jvp_rule, combine_fn=lax.mul)
ad.primitive_jvps[cummin_p] = partial(_cumulative_jvp_rule, combine_fn=lax.min)
ad.primitive_jvps[cummax_p] = partial(_cumulative_jvp_rule, combine_fn=lax.max)
