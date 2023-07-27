import string
from functools import partial
from typing import Any, Callable, Sequence
from dataclasses import dataclass

import numpy as np

import jax
import jax.dlpack
from jax import numpy as jnp
from jax._src import core
from jax._src.util import safe_map, safe_zip
map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

import torch
import torch.fx
from torch._inductor.compile_fx import compile_fx

def inductor_jit(f):
  def f_jit(*args, **kwargs):
    jaxpr, out_shape = jax.make_jaxpr(f, return_shape=True)(*args, **kwargs)
    args_flat = jax.tree_util.tree_leaves((args, kwargs))
    out_tree = jax.tree_util.tree_structure(out_shape)
    out_flat = ijit_p.bind(*args_flat, jaxpr=jaxpr)
    return jax.tree_util.tree_unflatten(out_tree, out_flat)
  return f_jit

ijit_p = core.Primitive('inductor_jit')

@ijit_p.def_impl
def _ijit_impl(*args, jaxpr):
  args = map(j2t, args)
  g = torch.fx.symbolic_trace(inductor_interpreter(jaxpr))
  exe = compile_fx(g, args)
  outs = exe(*args)
  return map(t2j, outs)

def j2t(x_jax):
  x_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x_jax))
  return x_torch

def t2j(x_torch):
  x_torch = x_torch.contiguous()  # https://github.com/google/jax/issues/8082
  x_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch))
  return x_jax

Val = Any

@dataclass(frozen=True)
class Context:
  avals_in: Sequence[core.AbstractValue]
  avals_out: Sequence[core.AbstractValue]


def inductor_interpreter(closed_jaxpr):
  jaxpr = closed_jaxpr.jaxpr
  consts = closed_jaxpr.consts
  def run(*args):
    device = args[0].device
    args = [args[i] for i in range(len(jaxpr.invars))]  # fx limitation

    env: dict[core.Var, Val] = {}

    def read(x: core.Atom) -> Val:
      if type(x) is core.Var:
        return env[x]
      val = x.val
      if not isinstance(val, np.ndarray):
        val = np.asarray(x.val)  # TODO: Deal with dtypes!!!
      return torch.from_numpy(val).to(device)

    def write(v: core.Var, x: Val) -> None:
      assert isinstance(x, (torch.Tensor, torch.fx.Proxy))
      env[v] = x

    device_consts = map(lambda x: torch.from_numpy(x).to(device), consts)
    map(write, jaxpr.constvars, device_consts)
    map(write, jaxpr.invars, args)
    for e in jaxpr.eqns:
      in_vals = map(read, e.invars)
      ctx = Context([v.aval for v in e.invars], [v.aval for v in e.outvars])
      if e.primitive not in translations:
        breakpoint()
        raise NotImplementedError(e)
      out_vals = translations[e.primitive](ctx, *in_vals, **e.params)
      map(write, e.outvars, out_vals)
    return map(read, jaxpr.outvars)
  return run

translations: dict[core.Primitive, Callable] = {}


from jax._src.lax import lax
translations[lax.sin_p] = lambda _, x: [torch.sin(x)]
translations[lax.exp_p] = lambda _, x: [torch.exp(x)]
translations[lax.sqrt_p] = lambda _, x: [torch.sqrt(x)]
translations[lax.rsqrt_p] = lambda _, x: [torch.rsqrt(x)]
translations[lax.lt_p] = lambda _, x, y: [torch.lt(x, y)]
translations[lax.integer_pow_p] = lambda _, x, y: [torch.pow(x, y)]
translations[lax.reduce_sum_p] = lambda _, x, axes: [torch.sum(x, axes)]
translations[jax._src.ad_util.stop_gradient_p] = lambda _, x: [x.detach()]
translations[lax.transpose_p] = lambda _, x, permutation: [x.permute(permutation)]

translations[lax.add_p] = lambda _, x, y: [torch.add(x, y)]
translations[lax.sub_p] = lambda _, x, y: [torch.sub(x, y)]
translations[lax.mul_p] = lambda _, x, y: [torch.mul(x, y)]
translations[lax.div_p] = lambda _, x, y: [torch.div(x, y)]
translations[lax.max_p] = lambda _, x, y: [torch.max(x, y)]

# TODO: I don't know why, but torch.max doesn't just work (unlike torch.sum)
def _reduce_max_rule(_, x, axes):
  if len(axes) == 1:
    return [x.max(axes[0])[0]]
  raise NotImplementedError(axes)
translations[lax.reduce_max_p] = _reduce_max_rule

def _reshape_rule(_, x, new_sizes, dimensions):
  if dimensions is not None:
    raise NotImplementedError
  return [x.reshape(new_sizes)]
translations[lax.reshape_p] = _reshape_rule

def _custom_jvp_call_rule(_, *args, call_jaxpr, jvp_jaxpr_thunk, num_consts, symbolic_zeros):
  # TODO: Preserve the rule!!
  if num_consts > 0:
    raise NotImplementedError
  return inductor_interpreter(call_jaxpr)(*args)
translations[jax._src.custom_derivatives.custom_jvp_call_p] = _custom_jvp_call_rule

def _pjit_rule(_, *args, jaxpr, name, **__):
  if jaxpr.consts:
    raise NotImplementedError
  # TODO: How to deal with shardings
  return inductor_interpreter(jaxpr)(*args)
translations[jax._src.pjit.pjit_p] = _pjit_rule

def _select_n_rule(_, b, x, y, *args):
  if args:
    raise NotImplementedError
  return [torch.where(b, y, x)]
translations[lax.select_n_p] = _select_n_rule

def _convert_element_type_rule(_, x, new_dtype, weak_type):
  if weak_type:
    raise NotImplementedError
  if new_dtype == jnp.dtype(jnp.float32):
    return [x.to(torch.float32)]
  raise NotImplementedError(new_dtype)
translations[lax.convert_element_type_p] = _convert_element_type_rule

def _broadcast_in_dim_rule(ctx, x, shape, broadcast_dimensions):
  aval, = ctx.avals_in
  rank_increase = len(shape) - len(aval.shape)
  # Add dimensions at the back
  if broadcast_dimensions == tuple(range(len(aval.shape))):
    return [x.reshape(*aval.shape, *([1] * rank_increase)).broadcast_to(shape)]
  # Add dimensions at the front
  if broadcast_dimensions == tuple(range(rank_increase, len(shape))):
    return [x.broadcast_to(shape)]
  raise NotImplementedError(aval.shape, shape, broadcast_dimensions)
translations[lax.broadcast_in_dim_p] = _broadcast_in_dim_rule

def _dot_general_rule(ctx, x, y, dimension_numbers, precision, preferred_element_type):
  # TODO: precision
  x_aval, y_aval = ctx.avals_in
  source = string.ascii_lowercase
  contracting, batch = dimension_numbers
  lhs_names, source = list(source[:len(x_aval.shape)]), source[len(x_aval.shape):]
  rhs_names, source = list(source[:len(y_aval.shape)]), source[len(y_aval.shape):]
  for l, r in zip(*contracting):
    rhs_names[r] = lhs_names[l]
  for l, r in zip(*batch):
    rhs_names[r] = lhs_names[l]
  batch_names = [lhs_names[l] for l in batch[0]]
  lhs_tensor_names = [n for i, n in enumerate(lhs_names)
                      if i not in contracting[0] and i not in batch[0]]
  rhs_tensor_names = [n for i, n in enumerate(rhs_names)
                      if i not in contracting[1] and i not in batch[1]]
  eqn = ''.join(lhs_names) + ',' + ''.join(rhs_names) + '->' + ''.join((*batch_names, *lhs_tensor_names, *rhs_tensor_names))
  return [torch.einsum(eqn, x, y)]
translations[lax.dot_general_p] = _dot_general_rule

def _gather_rule(ctx, src, idx, dimension_numbers, fill_value, indices_are_sorted, mode, slice_sizes, unique_indices):
  if (mode != jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS or
      fill_value is not None):
    raise NotImplementedError
  src_aval, idx_aval = ctx.avals_in
  if (len(src_aval.shape) != 2 or len(idx_aval.shape) != 2 or
      idx_aval.shape[-1] != 1 or
      dimension_numbers.offset_dims != (1,) or
      dimension_numbers.collapsed_slice_dims != (0,) or
      dimension_numbers.start_index_map != (0,)):
    raise NotImplementedError
  return [torch.index_select(src, 0, idx.squeeze().to(torch.int64))]
translations[lax.slicing.gather_p] = _gather_rule

