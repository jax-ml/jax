from functools import partial

import jax.core as core
import jax.linear_util as lu
import jax.numpy as np
import jax.lax as lax

from jax.util import curry, unzip2
from jax.lax import _abstractify
from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe
from jax.interpreters import ad

def pvals_with_zeros(zero_components, aval):
  if zero_components is True:
    return pe.PartialVal((None, ad.zero))
  elif zero_components is False:
    return pe.PartialVal((aval, core.unit))
  elif isinstance(zero_components, ZeroTuple):
    avals, consts = unzip(map, pvals_with_zeros, zero_components, aval)
    return pe.PartialVal((core.AbstractTuple(avals),
                          core.JaxprTracerTuple(consts)))

def transpose_jaxpr(jaxpr, avals, tangent_components):
  assert False

strip_zeros = partial(ad.strip_zeros, core.unit, core.pack)


@curry
def jaxpr_as_fun(jaxpr, consts, *args):
  return core.eval_jaxpr(jaxpr, consts, (), *args)


def call_initial(f, *args):
  pvals = map(_abstractify, args)
  avals = [aval for (aval, _) in pvals]
  jaxpr, _, consts = pe.trace_to_jaxpr(
      lu.wrap_init(f), pvals, instantiate=True)
  return call_initial_p.bind(core.pack(consts), *args, jaxpr=jaxpr)

def _call_initial_impl(consts, *args, **kwargs):
  jaxpr = kwargs.pop('jaxpr')
  return jaxpr_as_fun(jaxpr)(consts, *args)

def _call_initial_jvp(primals, tangents, jaxpr):
  avals = [aval for (aval, _) in map(_abstractify, primals)]
  where_zeros = map(ad.get_zeros, tangents)
  nonzero_tangents = strip_zeros(where_zeros, tangents)
  jaxpr_jvp, consts, where_zeros_out = ad.jvp_jaxpr(jaxpr, avals, where_zeros)
  primal_out, tangent_out = call_initial_p.bind(
      core.pack(consts), core.pack(primals),
      core.pack(nonzero_tangents), jaxpr=jaxpr_jvp)
  tangent_out_zeros = ad.put_zeros(ad.TangentTuple, where_zeros_out,
                                   tangent_out)
  return primal_out, tangent_out_zeros

def is_const(x):
  if x is None:
    return True
  elif type(x) is pe.JaxprTracerTuple:
    return tuple(map(is_const, x))
  elif isinstance(x, core.AbstractValue):
    return False
  else:
    raise TypeError(type(x))

def as_aval(pv, const):
  if pv is None:
    pv, _ = _abstractify(const)
    return pv
  elif type(pv) is pe.JaxprTracerTuple:
    return map(as_aval, pv, const)
  elif isinstance(pv, core.AbstractValue):
    return pv
  else:
    raise TypeError((pv, const))

def _call_initial_partial_eval(trace, *tracers, **kwargs):
  jaxpr = kwargs.pop('jaxpr')
  in_pvs, in_consts = unzip2([t.pval for t in tracers])
  first_components = map(is_const, in_pvs)
  avals = map(as_aval, in_pvs, in_consts)
  jaxpr_1, jaxpr_2, out_pv, first_components_out = pe.partial_eval_jaxpr(
      jaxpr, avals, first_components)
  out_pv_const, consts = call_initial_p.bind(core.unit, *in_consts, jaxpr=jaxpr_1)
  const_tracers = core.pack(map(trace.new_instantiated_const, consts))
  eqn = core.JaxprEqn((const_tracers,) + tracers, None, call_initial_p, (), False,
                      dict(jaxpr=jaxpr_2))
  return pe.JaxprTracer(trace, pe.PartialVal((out_pv, out_pv_const)), eqn)


def _call_initial_transpose():
  assert False

call_initial_p = core.Primitive("call_initial")
call_initial_p.def_impl(_call_initial_impl)
ad.primitive_jvps[call_initial_p] = _call_initial_jvp
pe.custom_partial_eval_rules[call_initial_p] = _call_initial_partial_eval


###


def demote_aval_rank(xs):
  if isinstance(xs, core.AbstractTuple):
    return core.AbstractTuple(map(demote_aval_rank, xs))
  else:
    return ShapedArray(xs.shape[1:], xs.dtype)

def promote_aval_rank(n, xs):
  if isinstance(xs, core.AbstractTuple):
    return core.AbstractTuple(map(partial(promote_aval_rank, n), xs))
  else:
    return ShapedArray((n,) + xs.shape, xs.dtype)

def leading_dim_size(xs):
  if isinstance(xs, core.JaxTuple):
    return leading_dim_size(xs[0])
  else:
    return xs.shape[0]

def empty_arrays(aval):
  if isinstance(aval, core.AbstractTuple):
    return core.pack(map(empty_arrays, aval))
  else:
    return lax.full(aval.shape, 0, aval.dtype)

def index_arrays(i, aval, xs):
  if isinstance(aval, core.AbstractTuple):
    return core.pack(map(partial(index_arrays, i), aval, xs))
  else:
    return lax.dynamic_index_in_dim(xs, i, keepdims=False)

def update_arrays(i, aval, xs, x):
  if isinstance(aval, core.AbstractTuple):
    return core.pack(map(partial(update_arrays, i), aval, xs, x))
  else:
    return lax.dynamic_update_index_in_dim(xs, x[None, ...], i, axis=0)


# scan :: (a -> c -> (b, c)) -> c -> [a] -> ([b], c)
def scan_initial(f, init, xs):
  carry_pval = carry_aval, _ = _abstractify(init)
  xs_aval, _ = _abstractify(xs)
  x_aval = demote_aval_rank(xs_aval)
  x_pval = pe.PartialVal((x_aval, core.unit))
  jaxpr, pval_out, consts = pe.trace_to_jaxpr(
      lu.wrap_init(f), (carry_pval, x_pval), instantiate=True)
  (y_aval, carry_aval_out), _ = pval_out
  assert carry_aval == carry_aval_out
  consts_aval, _ = _abstractify(core.pack(consts))
  avals = (consts_aval, x_aval, y_aval, carry_aval)
  return scan_initial_p.bind(core.pack(consts), init, xs,
                             avals=avals, jaxpr=jaxpr)


# scan_p :: (d -> a -> c -> (b, c)) -> d -> c -> [a] -> ([b], c)
def _scan_initial_impl(consts, init, xs, avals, jaxpr):
  # TODO maybe can do this work in the traceable, not every impl call
  length = leading_dim_size(xs)
  (_, x_aval, y_aval, _) = avals
  ys_aval = promote_aval_rank(length, y_aval)

  def body_fun(i, vals):
    carry, ys = vals
    x = index_arrays(i, x_aval, xs)
    y, carry_out = jaxpr_as_fun(jaxpr)(consts, x, carry)
    ys_out = update_arrays(i, y_aval, ys, y)
    return (carry_out, ys_out)

  ys_init = empty_arrays(ys_aval)
  carry, ys = lax.fori_loop(0, length, body_fun, (init, ys_init))
  return core.pack((ys, carry))


def _scan_initial_jvp(primals, tangents, avals, jaxpr):
  consts, init, xs = primals
  consts_dot, init_dot, xs_dot = tangents
  consts_aval, x_aval, y_aval, carry_aval = avals

  where_consts_zeros = ad.get_zeros(consts_dot)
  nonzero_consts_dot = strip_zeros(where_consts_zeros, consts_dot)

  where_init_zeros = ad.get_zeros(init_dot)
  nonzero_init_dot = strip_zeros(where_init_zeros, init_dot)

  where_xs_zeros = ad.get_zeros(xs_dot)  # same as where_x_zeros b/c arrays
  nonzero_xs_dot = strip_zeros(where_xs_zeros, xs_dot)

  jaxpr_jvp, new_consts, where_zeros_out = ad.jvp_jaxpr(
      jaxpr, (consts_aval, carry_aval, x_aval),
      (where_consts_zeros, where_init_zeros, where_xs_zeros))
  _, where_carry_zeros = where_zeros_out
  assert not new_consts  # TODO

  import ipdb; ipdb.set_trace()
  assert where_carry_zeros == where_init_zeros  # TODO while

  # TODO we realized consts are tricky... can't just add a new arg every time we
  # jvp like in n-ary call

  # out = scan_initial_p.bind(
  # (ys, ys_dot), (carry, carry_dot) = 

  # primal_out, tangent_out = call_initial_p.bind(
  #     core.pack(consts), core.pack(primals),
  #     core.pack(nonzero_tangents), jaxpr=jaxpr_jvp)
  # tangent_out_zeros = ad.put_zeros(ad.TangentTuple, where_zeros_out,
  #                                  tangent_out)
  # return primal_out, tangent_out_zeros


scan_initial_p = core.Primitive("scan_initial")
scan_initial_p.def_impl(_scan_initial_impl)
ad.primitive_jvps[scan_initial_p] = _scan_initial_jvp
# pe.custom_partial_eval_rules[scan_initial_p] = _scan_initial_partial_eval
