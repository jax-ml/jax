from functools import partial

import jax.core as core
import jax.linear_util as lu
import jax.numpy as np
import jax.lax as lax

from jax.util import curry, unzip2
from jax.lax import _abstractify, _unpack_eqn
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
strip_zeros_aval = partial(ad.strip_zeros, core.AbstractTuple(()), core.AbstractTuple)

def convert_zeros(keep_symbolic, example, tangent):
  if tangent is ad.zero:
    if keep_symbolic:
      return core.unit
    else:
      return ad.zeros_like_jaxval(example)
  elif type(tangent) is ad.TangentTuple:
    return core.pack(map(convert_zeros, keep_symbolic, example, tangent))
  else:
    return tangent

@curry
def jaxpr_as_fun(jaxpr, consts, *args):
  return core.eval_jaxpr(jaxpr, consts, (), *args)


_call_const = pe.gensym('_consts')

def call_initial(f, *args):
  pvals = map(_abstractify, args)
  avals = [aval for (aval, _) in pvals]
  jaxpr, _, consts = pe.trace_to_jaxpr(
      lu.wrap_init(f), pvals, instantiate=True)
  lifted_jaxpr = pe._closure_convert_jaxpr(jaxpr, _call_const)
  lifted_args = (core.pack(consts),) + args
  return call_initial_p.bind(*lifted_args, jaxpr=lifted_jaxpr, consts=())

def _call_initial_impl(*args, **kwargs):
  jaxpr = kwargs.pop('jaxpr')
  consts = kwargs.pop('consts')
  return jaxpr_as_fun(jaxpr, consts)(*args)

def _call_initial_jvp(primals, tangents, jaxpr, consts):
  avals = [aval for (aval, _) in map(_abstractify, primals)]
  where_zeros = map(ad.get_zeros, tangents)
  nonzero_tangents = strip_zeros(where_zeros, tangents)
  jaxpr_jvp, new_consts, where_zeros_out = ad.jvp_jaxpr(jaxpr, consts, avals, where_zeros)
  primal_out, tangent_out = call_initial_p.bind(
      core.pack(primals), core.pack(nonzero_tangents), jaxpr=jaxpr_jvp,
      consts=new_consts)
  tangent_out_zeros = ad.put_zeros(ad.TangentTuple, where_zeros_out, tangent_out)
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
  consts = kwargs.pop('consts')
  in_pvs, in_consts = unzip2([t.pval for t in tracers])
  first_components = map(is_const, in_pvs)
  avals = map(as_aval, in_pvs, in_consts)
  (jaxpr_1, consts_1), (jaxpr_2, consts_2), out_pv, first_components_out = \
      pe.partial_eval_jaxpr(jaxpr, consts, avals, first_components)
  out_const, residuals = call_initial_p.bind(
      *in_consts, jaxpr=jaxpr_1, consts=consts_1)
  residual_tracers = core.pack(map(trace.new_instantiated_const, residuals))
  eqn = core.JaxprEqn((residual_tracers,) + tracers, None, call_initial_p, (),
                      False, dict(jaxpr=jaxpr_2, consts=consts_2))
  return pe.JaxprTracer(trace, pe.PartialVal((out_pv, out_const)), eqn)


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

_scan_const = pe.gensym('_consts')

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
  lifted_jaxpr = pe._closure_convert_jaxpr(jaxpr, _scan_const)
  consts_aval, _ = _abstractify(core.pack(consts))
  avals = (consts_aval, x_aval, y_aval, carry_aval)
  return scan_initial_p.bind(core.pack(consts), init, xs,
                             avals=avals, jaxpr=lifted_jaxpr, true_consts=())


# scan_p :: (d -> a -> c -> (b, c)) -> d -> c -> [a] -> ([b], c)
def _scan_initial_impl(consts, init, xs, avals, jaxpr, true_consts):
  # TODO maybe can do this work in the traceable, not every impl call
  length = leading_dim_size(xs)
  (_, x_aval, y_aval, _) = avals
  ys_aval = promote_aval_rank(length, y_aval)

  def body_fun(i, vals):
    carry, ys = vals
    x = index_arrays(i, x_aval, xs)
    y, carry_out = jaxpr_as_fun(jaxpr, true_consts)(consts, x, carry)
    ys_out = update_arrays(i, y_aval, ys, y)
    return (carry_out, ys_out)

  ys_init = empty_arrays(ys_aval)
  carry, ys = lax.fori_loop(0, length, body_fun, (init, ys_init))
  return core.pack((ys, carry))


def _scan_initial_jvp(primals, tangents, avals, jaxpr, true_consts):
  consts, init, xs = primals
  consts_dot, init_dot, xs_dot = tangents
  consts_aval, x_aval, y_aval, carry_aval = avals

  where_consts_zeros = ad.get_zeros(consts_dot)
  where_init_zeros = ad.get_zeros(init_dot)
  where_xs_zeros = ad.get_zeros(xs_dot)  # same as where_x_zeros b/c arrays

  where_carry_zeros = where_init_zeros
  while True:
    jaxpr_jvp, new_consts, where_zeros_out = ad.jvp_jaxpr2(
        jaxpr, true_consts, (consts_aval, carry_aval, x_aval),
        (where_consts_zeros, where_carry_zeros, where_xs_zeros))
    where_ys_zeros, where_carry_zeros_out = where_zeros_out
    if where_carry_zeros_out == where_carry_zeros:
      break
    else:
      where_carry_zeros = binary_lattice_join(where_carry_zeros_out, where_carry_zeros)

  # convert_zeros is like strip_zeros but uses explicit lattice information to
  # instantiate zeros in some cases, namely in init_dot based on the fixed point
  nonzero_init_dot = convert_zeros(where_carry_zeros, init, init_dot)
  nonzero_consts_dot = convert_zeros(where_consts_zeros, consts, consts_dot)
  nonzero_xs_dot = convert_zeros(where_xs_zeros, xs, xs_dot)

  consts_dual = core.pack((consts, nonzero_consts_dot))
  init_dual = core.pack((init, nonzero_init_dot))
  xs_dual = core.pack((xs, nonzero_xs_dot))

  consts_dual_aval = core.AbstractTuple((consts_aval, strip_zeros_aval(where_consts_zeros, consts_aval)))
  x_dual_aval = core.AbstractTuple((x_aval, strip_zeros_aval(where_xs_zeros, x_aval)))
  y_dual_aval = core.AbstractTuple((y_aval, strip_zeros_aval(where_ys_zeros, y_aval)))
  carry_dual_aval = core.AbstractTuple((carry_aval, strip_zeros_aval(where_carry_zeros, carry_aval)))
  avals = (consts_dual_aval, x_dual_aval, y_dual_aval, carry_dual_aval)

  ys_dual, carry_out_dual = scan_initial_p.bind(
      consts_dual, init_dual, xs_dual, avals=avals, jaxpr=jaxpr_jvp,
      true_consts=new_consts)

  ys, ys_dot = ys_dual
  ys_dot = ad.put_zeros(ad.TangentTuple, where_ys_zeros, ys_dot)

  carry_out, carry_out_dot = carry_out_dual
  carry_out_dot = ad.put_zeros(ad.TangentTuple, where_carry_zeros_out, carry_out_dot)

  return core.pack((ys, carry_out)), ad.TangentTuple((ys_dot, carry_out_dot))

def instantiate_zeros(example, tangent, keep_symbolic):
  if tangent is ad.zero:
    if keep_symbolic:
      return tangent
    else:
      return ad.zeros_like_jaxval(example)
  elif isinstance(tangent, ad.TangentTuple):
    return ad.TangentTuple(map(instantiate_zeros, example, tangent, keep_symbolic))
  else:
    return tangent

def binary_lattice_join(a, b):
  if type(a) is tuple:
    return tuple(map(binary_lattice_join, a, b))
  else:
    return a and b


def _scan_initial_partial_eval(trace, *tracers, **kwargs):
  jaxpr = kwargs.pop('jaxpr')
  true_consts = kwargs.pop('true_consts')
  in_pvs, in_consts = unzip2([t.pval for t in tracers])
  fc_consts, fc_init, fc_xs = map(is_const, in_pvs)
  avals = map(as_aval, in_pvs, in_consts)

  fc_carry = fc_init
  while True:
    first_components = (fc_consts, fc_carry, fc_xs)
    (jaxpr_1, consts_1), (jaxpr_2, consts_2), out_pv, fc_carry_out = \
        pe.partial_eval_jaxpr2(jaxpr, true_consts, avals, first_components)
    if fc_carry_out == fc_carry:
      break
    else:
      fc_carry = binary_lattice_join(fc_carry, fc_carry_out)
  # TODO lift args according to lattice join result fc_carry
  # TODO maybe update tracers too... instantiate_const

  (ys, residuals), out_const = scan_initial_p.bind(
      *in_consts, avals=avals_1, jaxpr=jaxpr_1, true_consts=consts_1)
  residual_tracers = core.pack(map(trace.new_instantiated_const, residuals))
  d, c, a = tracers
  new_tracers = (d, c, core.pack((residual_tracers, a)))
  eqn = core.JaxprEqn(new_tracers, None, scan_initial_p, (), False,
                      dict(jaxpr=jaxpr_2, true_consts=consts_2, avals=avals_2))
  import ipdb; ipdb.set_trace()
  return pe.JaxprTracer((trace, pe.PartialVal((out_pv, out_const)), eqn))


scan_initial_p = core.Primitive("scan_initial")
scan_initial_p.def_impl(_scan_initial_impl)
ad.primitive_jvps[scan_initial_p] = _scan_initial_jvp
pe.custom_partial_eval_rules[scan_initial_p] = _scan_initial_partial_eval
