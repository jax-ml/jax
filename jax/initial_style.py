from functools import partial
from collections import namedtuple

import jax.core as core
import jax.linear_util as lu
import jax.numpy as np
import jax.lax as lax

from jax.util import curry, unzip2
from jax.api_util import pytree_to_jaxtupletree
from jax.lax import _abstractify, _unpack_eqn, _pack_eqn
from jax.abstract_arrays import ShapedArray
from jax.interpreters import partial_eval as pe
from jax.interpreters import ad
from jax import ad_util


def _convert_zeros(keep_symbolic, example, tangent):
  if tangent is ad.zero:
    if keep_symbolic:
      return core.unit
    else:
      return ad.zeros_like_jaxval(example)
  elif type(tangent) is ad.TangentTuple:
    return core.pack(map(_convert_zeros, keep_symbolic, example, tangent))
  else:
    return tangent

def _is_const(x):
  if x is None:
    return True
  elif type(x) is pe.JaxprTracerTuple:
    return tuple(map(_is_const, x))
  elif isinstance(x, core.AbstractValue):
    return False
  else:
    raise TypeError(type(x))


def _demote_aval_rank(xs):
  if isinstance(xs, core.AbstractTuple):
    return core.AbstractTuple(map(_demote_aval_rank, xs))
  else:
    return ShapedArray(xs.shape[1:], xs.dtype)

def _promote_aval_rank(n, xs):
  if isinstance(xs, core.AbstractTuple):
    return core.AbstractTuple(map(partial(_promote_aval_rank, n), xs))
  else:
    return ShapedArray((n,) + xs.shape, xs.dtype)

def _leading_dim_size(xs):
  if isinstance(xs, core.JaxTuple):
    return _leading_dim_size(xs[0])
  else:
    return xs.shape[0]

def _empty_arrays(aval):
  if isinstance(aval, core.AbstractTuple):
    return core.pack(map(_empty_arrays, aval))
  else:
    return lax.full(aval.shape, 0, aval.dtype)

def _index_arrays(i, aval, xs):
  if isinstance(aval, core.AbstractTuple):
    return core.pack(map(partial(_index_arrays, i), aval, xs))
  else:
    return lax.dynamic_index_in_dim(xs, i, keepdims=False)

def _update_arrays(i, aval, xs, x):
  if isinstance(aval, core.AbstractTuple):
    return core.pack(map(partial(_update_arrays, i), aval, xs, x))
  else:
    return lax.dynamic_update_index_in_dim(xs, x[None, ...], i, axis=0)

# scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
def scan(f, init, xs):
  carry_pval = carry_aval, _ = _abstractify(init)
  xs_aval, _ = _abstractify(xs)
  x_aval = _demote_aval_rank(xs_aval)
  x_pval = pe.PartialVal((x_aval, core.unit))
  jaxpr, pval_out, consts = pe.trace_to_jaxpr(
      lu.wrap_init(f), (carry_pval, x_pval), instantiate=True)
  (carry_aval_out, y_aval), _ = pval_out
  if carry_aval != carry_aval_out:
    msg = ("scanned function carry output does not match carry input: "
           "input carry is {} and output carry is {}")
    raise TypeError(msg.format(carry_aval, carry_aval_out))
  lifted_jaxpr = pe._closure_convert_jaxpr(jaxpr)
  consts_aval, _ = _abstractify(core.pack(consts))
  in_avals = (consts_aval, carry_aval, x_aval)
  out_aval = core.AbstractTuple((carry_aval, y_aval))
  jaxpr = core.TypedJaxpr(lifted_jaxpr, (), in_avals, out_aval)
  length = _leading_dim_size(xs)
  return scan_p.bind(core.pack(consts), init, xs,
                     forward=True, length=length, jaxpr=jaxpr)


def _scan_impl(consts, init, xs, forward, length, jaxpr):
  _, _, x_aval = jaxpr.in_avals
  _, y_aval = jaxpr.out_aval
  ys_aval = _promote_aval_rank(length, y_aval)

  def body_fun(i, vals):
    idx = i if forward else length - i - 1
    carry, ys = vals
    x = _index_arrays(idx, x_aval, xs)
    carry_out, y = core.jaxpr_as_fun(jaxpr)(consts, carry, x)
    ys_out = _update_arrays(idx, y_aval, ys, y)
    return (carry_out, ys_out)

  ys_init = _empty_arrays(ys_aval)
  carry, ys = lax.fori_loop(0, length, body_fun, (init, ys_init))
  return core.pack((carry, ys))


def _scan_jvp(primals, tangents, forward, length, jaxpr):
  consts, init, xs = primals
  consts_dot, init_dot, xs_dot = tangents
  consts_aval, carry_aval, x_aval = jaxpr.in_avals
  _, y_aval = jaxpr.out_aval

  where_consts_zeros = ad.get_zeros(consts_dot)
  where_init_zeros = ad.get_zeros(init_dot)
  where_xs_zeros = ad.get_zeros(xs_dot)  # same as where_x_zeros b/c arrays

  where_carry_zeros = where_init_zeros
  while True:
    where_zeros = (where_consts_zeros, where_carry_zeros, where_xs_zeros)
    jaxpr_jvp, where_zeros_out = ad.jvp_jaxpr(jaxpr, where_zeros)
    where_carry_zeros_out, where_ys_zeros = where_zeros_out
    if where_carry_zeros_out == where_carry_zeros:
      break
    else:
      where_carry_zeros = _binary_lattice_join(where_carry_zeros_out, where_carry_zeros)

  # convert_zeros is like strip_zeros but uses explicit lattice information to
  # instantiate zeros in some cases, namely in init_dot based on the fixed point
  nonzero_init_dot = _convert_zeros(where_carry_zeros, init, init_dot)
  nonzero_consts_dot = _convert_zeros(where_consts_zeros, consts, consts_dot)
  nonzero_xs_dot = _convert_zeros(where_xs_zeros, xs, xs_dot)

  consts_dual = core.pack((consts, nonzero_consts_dot))
  init_dual = core.pack((init, nonzero_init_dot))
  xs_dual = core.pack((xs, nonzero_xs_dot))

  carry_out_dual, ys_dual = scan_p.bind(
      consts_dual, init_dual, xs_dual,
      forward=forward, length=length, jaxpr=jaxpr_jvp)

  ys, ys_dot = ys_dual
  ys_dot = ad.put_zeros(ad.TangentTuple, where_ys_zeros, ys_dot)

  carry_out, carry_out_dot = carry_out_dual
  carry_out_dot = ad.put_zeros(ad.TangentTuple, where_carry_zeros_out, carry_out_dot)
  return core.pack((carry_out, ys)), ad.TangentTuple((carry_out_dot, ys_dot))

def _binary_lattice_join(a, b):
  t = (type(a), type(b))
  if t == (tuple, tuple):
    return tuple(map(_binary_lattice_join, a, b))
  elif t == (tuple, bool):
    return tuple(map(_binary_lattice_join, a, (b,) * len(a)))
  elif t == (bool, tuple):
    return tuple(map(_binary_lattice_join, (a,) * len(b), b))
  elif t == (bool, bool):
    return a and b
  else:
    raise TypeError((type(a), type(b)))


def _scan_partial_eval(trace, *tracers, **kwargs):
  jaxpr = kwargs.pop('jaxpr')
  length = kwargs.pop('length')
  forward = kwargs.pop('forward')
  assert not kwargs
  in_pvs, in_consts = unzip2([t.pval for t in tracers])
  fc_consts, fc_init, fc_xs = map(_is_const, in_pvs)

  fc_carry = fc_init
  while True:
    first_components = (fc_consts, fc_carry, fc_xs)
    jaxpr_1, jaxpr_2, fc_out = pe.partial_eval_jaxpr(jaxpr, first_components)
    fc_carry_out, fc_ys = fc_out
    if fc_carry_out == fc_carry:
      break
    else:
      fc_carry = _binary_lattice_join(fc_carry, fc_carry_out)

  consts_tracer, init_tracer, xs_tracer = tracers
  lifted_init_tracer = _lift_tracer(trace, init_tracer, fc_carry)
  lifted_tracers = consts_tracer, lifted_init_tracer, xs_tracer
  in_pvs, in_consts = unzip2([t.pval for t in lifted_tracers])

  out_pv = _put_known_pvs(fc_out, jaxpr.out_aval)

  out_carry, (ys, residuals) = scan_p.bind(
      *in_consts, forward=forward, length=length, jaxpr=jaxpr_1)
  out_const = core.pack((out_carry, ys))
  residuals_tracer = trace.new_instantiated_const(core.pack(residuals))
  d, c, a = lifted_tracers
  new_tracers = (d, c, (a, residuals_tracer))
  eqn = core.JaxprEqn(new_tracers, None, scan_p, (), True, False,
                      dict(forward=forward, length=length, jaxpr=jaxpr_2))
  return pe.JaxprTracer(trace, pe.PartialVal((out_pv, out_const)), eqn)

def _lift_tracer(trace, tracer, is_const):
  t = type(is_const)
  if t is bool:
    if not is_const:
      return trace.instantiate_const(tracer)
    else:
      return tracer
  elif t is tuple:
    tracers = map(trace.full_raise, tracer)
    return core.pack(map(partial(_lift_tracer, trace), tracers, is_const))
  else:
    raise TypeError(t)

def _put_known_pvs(is_known, aval):
  if is_known is True:
    return None
  elif is_known is False:
    return aval
  else:
    return pe.JaxprTracerTuple(map(_put_known_pvs, is_known, aval))


def _scan_transpose(ct, consts, init, xs, forward, length, jaxpr):
  assert consts is None and init is None
  assert type(xs) is tuple
  a, res = xs
  assert a is None and res is not None

  # jaxpr :: d -> c -> (a, res) ->  (c, b)
  # jaxpr_lifted :: res -> (d, c, a) -> (c, b)
  # jaxpr_lifted_trans :: res -> (CT c, CT b) -> (CT d, CT c, CT a)
  # jaxpr_trans :: * -> (CT c, CT d) -> (CT b, res) -> ((CT c, CT d), CT a)
  assert type(jaxpr.jaxpr.invars[2]) is tuple  # assume restructuring
  jaxpr_lifted = rearrange_binders(
      lambda d, c, a_res: (a_res[1], (d, c, a_res[0])), jaxpr)
  jaxpr_lifted_trans = _transpose_jaxpr(jaxpr_lifted)
  jaxpr_trans = _move_stuff_and_add_add(jaxpr_lifted_trans)

  c_aval, b_aval = jaxpr.out_aval
  d_aval, c_aval2, _ = jaxpr.in_avals
  assert c_aval == c_aval2
  bs_aval = _promote_aval_rank(length, b_aval)
  ct_d = ad_util.zeros_like_aval(d_aval)
  ct_c, ct_bs = ad.instantiate_zeros_aval(core.AbstractTuple((c_aval, bs_aval)), ct)
  carry_ct = core.pack((ct_c, ct_d))

  # jaxpr_trans :: * -> (CT c, CT d) -> (CT b, res) -> ((CT c, CT d), CT a)
  core.check_jaxpr(jaxpr_trans.jaxpr)
  unit_aval, (ct_c_aval, ct_d_aval), (ct_b_aval, _) = jaxpr_trans.in_avals
  assert core.lattice_join(ct_c_aval, core.get_aval(ct_c)) == ct_c_aval
  assert core.lattice_join(ct_d_aval, core.get_aval(ct_d)) == ct_d_aval

  out = scan_p.bind(
      core.unit, carry_ct, core.pack((ct_bs, res)),
      forward=not forward, length=length, jaxpr=jaxpr_trans)
  (ct_init, ct_consts), ct_as = out
  return ct_consts, ct_init, (ct_as, None)

def rearrange_binders(f, typed_jaxpr):
  jaxpr = typed_jaxpr.jaxpr.copy()
  jaxpr.invars = f(*jaxpr.invars)
  in_avals = f(*typed_jaxpr.in_avals)
  core.skip_checks or core.check_jaxpr(jaxpr)
  return core.TypedJaxpr(jaxpr, typed_jaxpr.literals, in_avals,
                         typed_jaxpr.out_aval)

_scan_newvar = pe.gensym('_scan')

def _move_stuff_and_add_add(typed_jaxpr):
  # jaxpr_lifted_trans :: res -> (CT c, CT b) -> (CT d, CT c, CT a)
  # jaxpr_trans :: * -> (CT c, CT d) -> (CT b, res) -> ((CT c, CT d), CT a)

  res_aval, (CTc_aval, CTb_aval) = typed_jaxpr.in_avals
  CTd_aval, CTc_aval2, CTa_aval = typed_jaxpr.out_aval
  assert CTc_aval == CTc_aval2
  in_avals = (core.AbstractTuple(()), core.AbstractTuple((CTc_aval, CTd_aval)),
              core.AbstractTuple((CTb_aval, res_aval)))
  out_aval = core.AbstractTuple((core.AbstractTuple((CTc_aval, CTd_aval)),
                                 CTa_aval))

  jaxpr = typed_jaxpr.jaxpr.copy()
  # assume the jaxpr isn't restructuring any inputs
  assert not any(type(invar) is tuple for invar in jaxpr.invars)

  # munge input side
  CTc_in = _scan_newvar()
  CTb_in = _scan_newvar()
  CTd_in = _scan_newvar()
  res_in, CTc_CTb_in = jaxpr.invars
  jaxpr.invars = ((), (CTc_in, CTd_in), (CTb_in, res_in))
  jaxpr.eqns = (
      [_pack_eqn([CTc_in, CTb_in], CTc_CTb_in)] +
      jaxpr.eqns)

  # munge output side
  CTd_new = _scan_newvar()
  CTd_sum = _scan_newvar()
  CTc = _scan_newvar()
  CTa = _scan_newvar()
  partial_out = _scan_newvar()
  outvar = _scan_newvar()
  jaxpr.eqns = (
      jaxpr.eqns +
      [_unpack_eqn(jaxpr.outvar, [CTd_new, CTc, CTa]),
       _add_any_eqn(CTd_sum, CTd_new, CTd_in),
       _pack_eqn([CTc, CTd_sum], partial_out),
       _pack_eqn([partial_out, CTa], outvar)])
  jaxpr.outvar = outvar

  # TODO(mattjj): use check_typed_jaxpr
  core.skip_checks or core.check_jaxpr(jaxpr)
  return core.TypedJaxpr(jaxpr, typed_jaxpr.literals,
                         in_avals, out_aval)

def _add_any_eqn(tot, a, b):
  return core.JaxprEqn([a, b], [tot], ad_util.add_jaxvals_p, (), False, False, {})


# transpose_jaxpr :: (res -> a -> b) -> (res -> CT b -> CT a)
def _transpose_jaxpr(jaxpr):
  assert len(jaxpr.in_avals) == 2

  @lu.wrap_init
  def transposed(res, b_bar):
    _, (_, a_bar) = ad.backward_pass(jaxpr.jaxpr, jaxpr.literals, (),
                                     (res, None), b_bar)
    a_bar = ad.instantiate_zeros_aval(jaxpr.in_avals[1], a_bar)
    return a_bar

  transposed_jaxpr = _make_typed_jaxpr(transposed, (jaxpr.in_avals[0], jaxpr.out_aval))
  return transposed_jaxpr

def _make_typed_jaxpr(traceable, in_avals):
  pvals = [pe.PartialVal((aval, core.unit)) for aval in in_avals]
  jaxpr, pval_out, consts = pe.trace_to_jaxpr(traceable, pvals, instantiate=True)
  out_aval, _ = pval_out
  assert isinstance(out_aval, core.AbstractValue)
  return core.TypedJaxpr(jaxpr, consts, in_avals, out_aval)


scan_p = core.Primitive("scan")
scan_p.def_impl(_scan_impl)
ad.primitive_jvps[scan_p] = _scan_jvp
ad.primitive_transposes[scan_p] = _scan_transpose
pe.custom_partial_eval_rules[scan_p] = _scan_partial_eval
