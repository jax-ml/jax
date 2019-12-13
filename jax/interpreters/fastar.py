from contextlib import contextmanager
from functools import reduce
from operator import and_

import numpy as onp

import jax.core as jc
import jax.interpreters.partial_eval as pe
import jax.lax as lax
import jax.linear_util as lu
import jax.numpy as np
import jax.scipy.special as special
from jax import abstract_arrays, jit as jit_
from jax.ad_util import zeros_like_aval
from jax.api_util import flatten_fun_nokwargs
from jax.interpreters import xla
from jax.ops import index_update
from jax.tree_util import (
  tree_flatten, tree_unflatten, tree_map, Partial, register_pytree_node)
from jax.util import curry, safe_zip, safe_map
from jax.util import partial, unzip3, cache

map = safe_map
zip = safe_zip


def true_mask(val):
  return onp.full(onp.shape(val), True, dtype=bool)


def false_mask(val):
  return onp.full(onp.shape(val), False, dtype=bool)


def _mask_all(parray):
  _, mask = parray
  return onp.all(mask)


class HashableMask(object):
  def __init__(self, mask):
    self.mask = mask

  def __hash__(self):
    return hash(self.mask.tostring())

  def __eq__(self, other):
    return onp.all(self.mask == other.mask)


def _to_tree(idxs):
  tree = {}
  for idx in idxs:
    branch = tree
    for i in idx:
      branch = branch.setdefault(i, {})
  return tree


def _contains_rectangle(idx_tree, rectangle):
  """
  Return True if rectangle is contained in idx_tree, else False.
  """
  (start, stop), rectangle = rectangle[0], rectangle[1:]
  return all(
    n in idx_tree
    and (not rectangle or _contains_rectangle(idx_tree[n], rectangle))
    for n in range(start, stop))


def _remove_rectangle(idx_tree, rectangle):
  (start, stop), rectangle = rectangle[0], rectangle[1:]
  new_tree = {}
  for root, branch in idx_tree.items():
    if start <= root < stop:
      if rectangle:
        new_branch = _remove_rectangle(branch, rectangle)
        if new_branch:
          new_tree[root] = new_branch
    else:
      new_tree[root] = branch
  return new_tree


def _find_rectangle(idx_tree):
  """
  Greedily find a rectangle in idx_tree.
  """
  start = min(idx_tree.keys())
  stop = start + 1
  branch = idx_tree[start]
  if branch:
    rect = _find_rectangle(branch)
    while stop in idx_tree and _contains_rectangle(idx_tree[stop], rect):
      stop += 1
    return ((start, stop),) + rect
  else:
    while stop in idx_tree:
      stop += 1
    return (start, stop),


def _mask_to_slices(mask):
  """
  Greedily search for rectangular slices in mask.
  """
  if onp.shape(mask) == ():
    return [()] if mask else []

  rectangles = []
  idx_tree = _to_tree(onp.argwhere(mask))
  while idx_tree:
    rect = _find_rectangle(idx_tree)
    rectangles.append(rect)
    idx_tree = _remove_rectangle(idx_tree, rect)
  return [tuple(slice(s, e) for s, e in rect) for rect in rectangles]


def _get_aval(x):
  return abstract_arrays.raise_to_shaped(jc.get_aval(x))


## Core
init_rules = {}


def _init_ans(prim, *args, **params):
  if prim in init_rules:
    return init_rules[prim](*args, **params)
  else:
    args = [_get_aval(arg) for arg, _ in args]
    abstract_out = prim.abstract_eval(*args, **params)
    if prim.multiple_results:
      return [parray(zeros_like_aval(o), false_mask(o))
              for o in abstract_out]
    else:
      return parray(zeros_like_aval(abstract_out),
                    false_mask(abstract_out))


update_rules = {}


def _get_update(p):
  try:
    return update_rules[p]
  except KeyError:
    raise NotImplementedError(
      "Fastar update rule for '{}' not implemented".format(p))


# Populate the cache
def _firstpass(jaxpr, consts, freevar_vals, args):
  def read(v):
    if type(v) is jc.Literal:
      return parray(v.val, true_mask(v.val))
    else:
      val = env[repr(v)]
      assert isinstance(val, Parray)
      return val

  def write(v, val):
    assert isinstance(val, Parray)
    env[repr(v)] = val

  def delete(v):
    del env[repr(v)]

  def write_subenvs(vs, env):
    subenvs[repr(vs)] = env

  env = {}
  subenvs = {}

  write(jc.unitvar, parray(jc.unit, true_mask(jc.unit)))
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  map(write, jaxpr.freevars, freevar_vals)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    if eqn.bound_subjaxprs:
      subjaxprs, sub_consts, sub_freevar_vals = unzip3([(
        subjaxpr,
        map(read, const_vars),
        map(read, bound_vars))
        for subjaxpr, const_vars, bound_vars
        in eqn.bound_subjaxprs])
      ans, subenvs_ = _init_ans(eqn.primitive, eqn.params, subjaxprs,
                                sub_consts, sub_freevar_vals, in_vals)
      ans, subenvs_ = _get_update(eqn.primitive)(
        eqn.params, subjaxprs, sub_consts, sub_freevar_vals, in_vals,
        subenvs_)

      write_subenvs(tuple(eqn.outvars), subenvs_)
    else:
      ans = _init_ans(eqn.primitive, *in_vals, **eqn.params)
      ans = _get_update(eqn.primitive)(ans, *in_vals, **eqn.params)
    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
  map(delete, jaxpr.constvars)
  map(delete, jaxpr.invars)
  map(delete, jaxpr.freevars)
  return map(read, jaxpr.outvars), (env, subenvs)


@lu.transformation_with_aux
def _inited_fun(jaxpr, in_tree_def, *args):
  consts, freevar_vals, args = tree_unflatten(in_tree_def, args)
  out = yield (jaxpr, consts, freevar_vals, args), {}
  out = out[0], (out[1],)  # Tuple of envs, not singleton env
  out, out_treedef = tree_flatten(out)
  yield out, out_treedef


def _call_init_rule(primitive, params, jaxpr, consts, freevar_vals, in_vals):
  jaxpr, = jaxpr
  consts, = consts
  freevar_vals, = freevar_vals
  all_args, in_treedef = tree_flatten((consts, freevar_vals, in_vals))
  fun = lu.wrap_init(_firstpass)
  fun, out_treedef = _inited_fun(fun, jaxpr, in_treedef)
  out = primitive.bind(fun, *all_args, **params)
  return tree_unflatten(out_treedef(), out)


init_rules[xla.xla_call_p] = partial(_call_init_rule, xla.xla_call_p)


def _fastpass(jaxpr, consts, freevar_vals, args, old_env):
  old_env, old_subenvs = old_env

  def read(v):
    if type(v) is jc.Literal:
      return parray(v.val, true_mask(v.val))
    else:
      val = env[repr(v)]
      assert isinstance(val, Parray)
      return val

  def read_old(v):
    if type(v) is jc.Literal:
      return parray(v.val, true_mask(v.val))
    else:
      val = old_env[repr(v)]
      assert isinstance(val, Parray)
      return val

  def read_old_subenvs(vs):
    return old_subenvs[repr(vs)]

  def write(v, val):
    assert isinstance(val, Parray)
    env[repr(v)] = val

  def write_subenvs(vs, env):
    subenvs[repr(vs)] = env

  def delete(v):
    del env[repr(v)]

  env = {}
  subenvs = {}

  write(jc.unitvar, parray(jc.unit, true_mask(jc.unit)))
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  map(write, jaxpr.freevars, freevar_vals)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    old_outvals = map(read_old, eqn.outvars)
    if eqn.primitive.multiple_results:
      old_ans = old_outvals
    else:
      old_ans = old_outvals[0]
    in_vals = map(read, eqn.invars)
    if eqn.bound_subjaxprs:
      subjaxprs, sub_consts, sub_freevar_vals = unzip3([(
        subjaxpr,
        map(read, const_vars),
        map(read, bound_vars))
        for subjaxpr, const_vars, bound_vars in eqn.bound_subjaxprs])
      old_subenvs_ = read_old_subenvs(tuple(eqn.outvars))
      ans, subenvs_ = _get_update(eqn.primitive)(
        eqn.params, subjaxprs, sub_consts, sub_freevar_vals, in_vals,
        old_subenvs_)
      write_subenvs(tuple(eqn.outvars), subenvs_)
    else:
      ans = _get_update(eqn.primitive)(old_ans, *in_vals, **eqn.params)
    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
  map(delete, jaxpr.constvars)
  map(delete, jaxpr.invars)
  map(delete, jaxpr.freevars)
  return map(read, jaxpr.outvars), (env, subenvs)


@lu.transformation_with_aux
def _updated_fun(jaxpr, in_treedef, *args):
  consts, freevar_vals, args, env = tree_unflatten(in_treedef, args)
  out = yield (jaxpr, consts, freevar_vals, args, env), {}
  out = out[0], (out[1],)  # Tuple of envs, not singleton env
  out, out_treedef = tree_flatten(out)
  yield out, out_treedef


def _call_update_rule(
        primitive, params, jaxpr, consts, freevar_vals, in_vals, env):
  jaxpr, = jaxpr
  consts, = consts
  freevar_vals, = freevar_vals
  env, = env
  all_args, in_treedef = tree_flatten((consts, freevar_vals, in_vals, env))
  fun = lu.wrap_init(_fastpass)
  fun, out_treedef = _updated_fun(fun, jaxpr, in_treedef)
  out = primitive.bind(fun, *all_args, **params)
  return tree_unflatten(out_treedef(), out)


update_rules[xla.xla_call_p] = partial(_call_update_rule, xla.xla_call_p)


class Parray(tuple):
  """
  Array which has been partially evaluated, represented by a pair

      (arr, computed)

  where `computed` is a boolean array with True where arr has been computed,
  False where it hasn't yet. `arr` should be initialized to zeros so that
  np.all(~computed & arr == 0).
  """
  pass


# This isn't a pytree node
class ProtectedParray(Parray):
  pass


def _parray_to_iterable(parray):
  if _protect_parrays_state:
    return (ProtectedParray(parray),), None
  else:
    arr, mask = parray
    return (arr,), HashableMask(mask)


def _iterable_to_parray(mask, arr):
  arr, = arr
  if mask is None:
    return arr
  else:
    return parray(arr, mask.mask)


parray = lambda arr, mask: Parray((arr, mask))
register_pytree_node(Parray, _parray_to_iterable, _iterable_to_parray)

_protect_parrays_state = []


@contextmanager
def _protect_parrays():
  _protect_parrays_state.append(None)
  yield
  _protect_parrays_state.pop()


## High level API
def accelerate_part(fun, jit=True):
  def fast_fun(env, *args):
    ans, env = _update_env(fun, args, env)
    return ans, Partial(fast_fun, env)

  fast_fun = jit_(fast_fun) if jit else fast_fun

  def first_fun(*args):
    ans, env = _init_env(fun, args)
    return ans, Partial(fast_fun, env)

  first_fun = jit_(first_fun) if jit else first_fun
  return first_fun


def accelerate_sections(fixed_point_fun, jit_every=10):
  @jit_
  def accelerated_start(fp_args, x):
    fp = fixed_point_fun(*fp_args)
    x = tree_map(lambda arr: parray(arr, false_mask(arr)), x)
    x, env = _init_env(fp, [x])
    i = 1
    while not _mask_all(x) and i < jit_every:
      x, env = _update_env(fp, [x], env)
      i = i + 1
    if _mask_all(x):
      return x, None
    else:
      return x, Partial(accelerated_section, fp_args, env)

  @jit_
  def accelerated_section(fp_args, env, x):
    fp = fixed_point_fun(*fp_args)
    i = 0
    while not _mask_all(x) and i < jit_every:
      x, env = _update_env(fp, [x], env)
      i = i + 1
    if _mask_all(x):
      return x, None
    else:
      return x, Partial(accelerated_section, fp_args, env)

  return accelerated_start


@cache()
def _fastar_jaxpr(fun, in_tree, in_avals):
  in_pvals = [pe.PartialVal((aval, jc.unit)) for aval in in_avals]
  fun_flat, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  jaxpr, _, consts = pe.trace_to_jaxpr(fun_flat, in_pvals)
  return jaxpr, consts, out_tree()


def _init_env(fun, args):
  with _protect_parrays():
    args_flat, in_tree = tree_flatten(args)
  assert all(type(arg) is ProtectedParray for arg in args_flat)
  args_flat = [Parray(arg) for arg in args_flat]
  avals = tuple(_get_aval(arg) for arg, _ in args_flat)
  jaxpr, consts, out_tree = _fastar_jaxpr(fun, in_tree, avals)
  consts = [parray(const, true_mask(const)) for const in consts]
  ans, env = _firstpass(jaxpr, consts, [], args_flat)
  return tree_unflatten(out_tree, ans), env


def _update_env(fun, args, env):
  with _protect_parrays():
    args_flat, in_tree = tree_flatten(args)
  assert all(type(arg) is ProtectedParray for arg in args_flat)
  args_flat = [Parray(arg) for arg in args_flat]
  avals = tuple(_get_aval(arg) for arg, _ in args_flat)
  jaxpr, consts, out_tree = _fastar_jaxpr(fun, in_tree, avals)
  consts = [parray(const, true_mask(const)) for const in consts]
  ans, env = _fastpass(jaxpr, consts, [], args_flat, env)
  return tree_unflatten(out_tree, ans), env


_init_value = 0


def _unbroadcast_slice(s, shape):
  return tuple(s if dim_sz > 1 else slice(None)
               for s, dim_sz in zip(s[len(s) - len(shape):], shape))


# n-ary elementwise operators with broadcasting
@curry
def _nop_update(op, ans, *args):
  args, arg_masks = zip(*args)
  args = map(np.asarray, args)
  ans, ans_mask = ans
  new_ans_mask = reduce(and_, arg_masks)
  slices = _mask_to_slices(new_ans_mask & ~ ans_mask)
  for s in slices:
    part_args = [a[_unbroadcast_slice(s, np.shape(a))] for a in args]
    ans = index_update(ans, s, op.bind(*part_args))
  return Parray((ans, new_ans_mask))


_nops = [
  lax.abs_p,
  lax.add_p,
  lax.and_p,
  lax.ceil_p,
  lax.cos_p,
  lax.div_p,
  lax.eq_p,
  lax.exp_p,
  lax.expm1_p,
  lax.floor_p,
  lax.ge_p,
  lax.gt_p,
  lax.is_finite_p,
  lax.le_p,
  lax.log_p,
  lax.log1p_p,
  lax.lt_p,
  lax.max_p,
  lax.min_p,
  lax.mul_p,
  lax.ne_p,
  lax.neg_p,
  lax.not_p,
  lax.rem_p,
  lax.select_p,
  lax.sign_p,
  lax.sin_p,
  lax.sub_p,
  lax.tanh_p,
]

for op in _nops:
  update_rules[op] = _nop_update(op)


# Logit and expit use custom_transforms so their primitives have a different
# form.
@curry
def _logexpit_update(op, ans, x, **params):
  x, x_mask = x
  (ans, ans_mask), = ans
  slices = _mask_to_slices(x_mask & ~ ans_mask)
  for s in slices:
    part_x = x[_unbroadcast_slice(s, np.shape(x))]
    ans = index_update(ans, s, op.bind(part_x, **params)[0])
  return [Parray((ans, x_mask))]


for op in [special.expit.prim, special.logit.prim]:
  update_rules[op] = _logexpit_update(op)


# Cheap ops, these are assumed to require little or no computation
@curry
def _cheap_op_update(op, old_out, *args, **params):
  args, args_mask = zip(*args)
  # TODO: use onp equivalents to process the masks
  return Parray((op.bind(*args, **params),
                 onp.bool_(op.bind(*args_mask, **params))))


_cheap_ops = [
  lax.broadcast_p,
  lax.broadcast_in_dim_p,
  lax.concatenate_p,
  lax.convert_element_type_p,
  lax.reshape_p,
  lax.rev_p,
  lax.slice_p,
  lax.tie_in_p,
  lax.transpose_p,
]

for op in _cheap_ops:
  update_rules[op] = _cheap_op_update(op)


def _gather_update(
        old_out, operand, start_indices, dimension_numbers, slice_sizes,
        **params):
  # Treat gather as a cheap op, but need to handle start_indices correctly
  operand, operand_mask = operand
  start_indices, start_indices_mask = start_indices
  assert onp.all(start_indices_mask)
  # Treat gather as a cheap op
  return parray(
    lax.gather_p.bind(
      operand, start_indices, dimension_numbers=dimension_numbers,
      slice_sizes=slice_sizes, **params),
    onp.asarray(lax.gather_p.bind(
      operand_mask, start_indices, dimension_numbers=dimension_numbers,
      slice_sizes=slice_sizes, **params)))


update_rules[lax.gather_p] = _gather_update


def _gather_static_update(
        old_out, operand, start_indices, start_indices_shape,
        dimension_numbers,
        slice_sizes, **params):
  # Treat gather as a cheap op, but need to handle start_indices correctly
  operand, operand_mask = operand
  start_indices = onp.reshape(onp.array(start_indices, dtype=int),
                              start_indices_shape)
  # Treat gather as a cheap op
  return parray(
    lax.gather_p.bind(
      operand, start_indices, dimension_numbers=dimension_numbers,
      slice_sizes=slice_sizes, **params),
    onp.asarray(lax.gather_p.bind(
      operand_mask, start_indices, dimension_numbers=dimension_numbers,
      slice_sizes=slice_sizes, **params)))


update_rules[lax.gather_static_p] = _gather_static_update


# Reductions
@curry
def _reduce_update(op, ans, a, **params):
  a, a_mask = a
  ans, ans_mask = ans
  axes = params['axes']
  new_ans_mask = onp.all(a_mask, axes)
  slices = _mask_to_slices(new_ans_mask & ~ ans_mask)
  for s in slices:
    a_slice = list(s)
    for axis in axes:
      a_slice.insert(axis, slice(None))
    a_part = a[tuple(a_slice)]
    if 'input_shape' in params:
      params['input_shape'] = np.shape(a_part)
    ans = index_update(ans, s, op.bind(a_part, **params))
  return Parray((ans, new_ans_mask))


reduce_ops = [
  lax.reduce_max_p,
  lax.reduce_min_p,
  lax.reduce_sum_p,
]

for op in reduce_ops:
  update_rules[op] = _reduce_update(op)


def _dot_general_update(ans, a, b, dimension_numbers, precision=None):
  a, a_mask = a
  b, b_mask = b
  ansval, ansmask = ans
  (a_cont_dims, b_cont_dims), (a_btch_dims, b_btch_dims) = dimension_numbers

  a_outr_dims = tuple(d for d in range(a.ndim)
                      if d not in a_cont_dims + a_btch_dims)
  b_outr_dims = tuple(d for d in range(b.ndim)
                      if d not in b_cont_dims + b_btch_dims)

  cont_ndim = len(a_cont_dims)
  btch_ndim = len(a_btch_dims)

  # Batch dims to front
  a_mask = onp.moveaxis(a_mask, a_btch_dims, range(btch_ndim))
  a_cont_dims_ = tuple(c + sum(1 for b in a_btch_dims if b > c)
                       for c in a_cont_dims)
  b_mask = onp.moveaxis(b_mask, b_btch_dims, range(len(b_btch_dims)))
  b_cont_dims_ = tuple(c + sum(1 for b in b_btch_dims if b > c)
                       for c in b_cont_dims)

  a_mask = onp.all(a_mask, a_cont_dims_)
  b_mask = onp.all(b_mask, b_cont_dims_)

  new_ansmask = (
          a_mask[(Ellipsis,) + len(b_outr_dims) * (onp.newaxis,)]
          & b_mask[btch_ndim * (slice(None),)
                   + len(a_outr_dims) * (onp.newaxis,)])
  for s in _mask_to_slices(new_ansmask & ~ ansmask):
    s_btch, s_a, s_b = (s[:btch_ndim],
                        s[btch_ndim:btch_ndim + len(a_outr_dims)],
                        s[btch_ndim + len(a_outr_dims):])
    a_slice = tuple((s_btch + s_a + cont_ndim * (slice(None),))[d] for d in
                    onp.argsort(a_btch_dims + a_outr_dims + a_cont_dims))
    b_slice = tuple((s_btch + s_b + cont_ndim * (slice(None),))[d] for d in
                    onp.argsort(b_btch_dims + b_outr_dims + b_cont_dims))
    ansval = index_update(ansval, s,
                          lax.dot_general_p.bind(
                            a[a_slice], b[b_slice],
                            dimension_numbers=dimension_numbers,
                            precision=precision))
  return Parray((ansval, new_ansmask))


update_rules[lax.dot_general_p] = _dot_general_update


def _pad_update(old_out, input, padding_value, padding_config):
  for (lo, hi, interior) in padding_config:
    if interior > 0 and (lo < 0 or hi < 0): raise NotImplementedError(
      "Interior and negative padding on same axis not yet implemented.")

  input, input_mask = input
  padding_value, padding_value_mask = padding_value
  outval, old_outmask = old_out

  unpad_slice = tuple(
    slice(lo if lo > 0 else None,
          -hi if hi > 0 else None,
          None if interior == 0 else interior + 1)
    for (lo, hi, interior) in padding_config)
  unpad = lambda x: x[unpad_slice]

  def pad_slices():
    for index, (lo, hi, interior) in enumerate(padding_config):
      def nonoverlapping(s):
        return unpad_slice[:index] + (s,) + \
               tuple([slice(None)] * (old_outmask.ndim - index - 1))

      if lo > 0:
        yield nonoverlapping(slice(lo))

      for i in range(interior):
        yield nonoverlapping(slice(lo + i + 1, -hi, (interior + 1)))

      if hi > 0:
        yield nonoverlapping(slice(-hi, None))

  old_padding_value_mask = any(onp.any(old_outmask[s]) for s in pad_slices())
  new_padding_value_mask = padding_value_mask and not old_padding_value_mask

  output_mask = old_outmask.copy()

  input_crop_slice = tuple(
    slice(-lo if lo < 0 else None,
          hi if hi < 0 else None)
    for (lo, hi, _) in padding_config)

  cropped_input_mask = input_mask[input_crop_slice]
  output_mask[unpad_slice] = cropped_input_mask
  if new_padding_value_mask:
    for s in pad_slices():
      outval = index_update(outval, s, np.broadcast_to(padding_value,
                                                       output_mask[
                                                         s].shape))
      output_mask[s] = True

  cropped_input_new_mask = cropped_input_mask & ~unpad(old_outmask)
  cropped_input_slices = _mask_to_slices(cropped_input_new_mask)
  for cropped_input_slice in cropped_input_slices:
    output_slice = tuple(
      slice((lo if lo > 0 else 0) + s.start * (interior + 1),
            (lo if lo > 0 else 0) + s.stop * (interior + 1),
            interior + 1)
      for s, (lo, _, interior) in
      zip(cropped_input_slice, padding_config))

    input_slice = tuple(
      slice(s.start + (-lo if lo < 0 else 0),
            s.stop + (-lo if lo < 0 else 0), s.step)
      for s, (lo, _, interior) in
      zip(cropped_input_slice, padding_config))

    # assert np.all(outval[output_slice] == init_value)
    outval = index_update(outval, output_slice, input[input_slice])

  return Parray((outval, output_mask))


update_rules[lax.pad_p] = _pad_update


def _conv_general_dilated_outmask(lhs_mask, rhs_mask, **params):
  # Note: we assume that rhs_mask doesn't change
  lhs_mask, rhs_mask = np.float32(lhs_mask), np.float32(rhs_mask)
  out = onp.array(lax.conv_general_dilated(lhs_mask, rhs_mask, **params))
  full_out = onp.array(lax.conv_general_dilated(
    onp.ones_like(lhs_mask), onp.ones_like(rhs_mask), **params))
  return out == full_out


def _conv_general_dilated_update_slice_op(
        slc, out, lhs, rhs, window_strides, padding,
        lhs_dilation=None, rhs_dilation=None, dimension_numbers=None,
        precision=None):
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  lhs_shape = onp.take(np.shape(lhs), lhs_spec)
  rhs_shape = onp.take(np.shape(rhs), rhs_spec)
  out_slc = [slc[i] for i in out_spec]
  pad_low, _ = onp.transpose(padding)
  window_shape = lax.lax._dilate_shape(rhs_shape, rhs_dilation)[2:]
  out_start, out_stop = onp.transpose([[s.start, s.stop] for s in out_slc])
  out_start_dilated = out_start[2:] * onp.array(window_strides)
  out_stop_dilated = (out_stop[2:] - 1) * onp.array(window_strides) + 1
  lhs_start_dilated = out_start_dilated - pad_low
  lhs_stop_dilated = out_stop_dilated + window_shape - 1 - pad_low
  lhs_start, lhs_stop = onp.clip(
    (onp.array(
      [lhs_start_dilated, lhs_stop_dilated]) - 1) // lhs_dilation + 1, 0,
    lhs_shape[2:])

  if onp.any(lhs_start == lhs_stop):
    return out

  sub_pad_low = onp.maximum(
    onp.multiply(lhs_start, lhs_dilation) - lhs_start_dilated, 0)
  sub_pad_high = onp.maximum(
    lhs_stop_dilated - onp.multiply((lhs_stop - 1), lhs_dilation) - 1, 0)
  sub_padding = zip(sub_pad_low, sub_pad_high)
  lhs_slice = ((out_slc[0], slice(None)) +
               tuple(slice(int(s), int(e)) for s, e in
                     zip(lhs_start, lhs_stop)))
  new = lax.conv_general_dilated(
    lhs[tuple(onp.take(lhs_slice, onp.argsort(lhs_spec)))], rhs,
    window_strides=window_strides, padding=sub_padding,
    lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
    dimension_numbers=dimension_numbers, precision=precision)
  return index_update(out, slc, new)


def _conv_general_dilated_update(old_out, lhs, rhs, window_strides, padding,
                                 lhs_dilation, rhs_dilation, dimension_numbers,
                                 feature_group_count, lhs_shape, rhs_shape,
                                 precision):
  lhs, lhs_mask = lhs
  rhs, rhs_mask = rhs

  if not np.all(rhs_mask) or feature_group_count > 1:
    raise NotImplementedError

  outval, old_outmask = old_out

  outmask = _conv_general_dilated_outmask(
    lhs_mask, rhs_mask, window_strides=window_strides, padding=padding,
    lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
    dimension_numbers=dimension_numbers)

  # if dimension_numbers is not None:
  #   assert dimension_numbers.lhs_spec == tuple(range(np.ndim(lhs)))
  #   assert dimension_numbers.rhs_spec == tuple(range(np.ndim(rhs)))
  #   assert dimension_numbers.out_spec == tuple(range(np.ndim(outval)))

  new_mask = outmask & ~old_outmask
  for slice in _mask_to_slices(new_mask):
    outval = _conv_general_dilated_update_slice_op(
      slice, outval, lhs, rhs, window_strides, padding,
      lhs_dilation, rhs_dilation, dimension_numbers, precision)

  return Parray((outval, outmask))


update_rules[lax.conv_general_dilated_p] = _conv_general_dilated_update


def _device_put_update(old_out, x, **params):
  x, mask = x
  return parray(xla.device_put_p.bind(x, **params), mask)


update_rules[xla.device_put_p] = _device_put_update
