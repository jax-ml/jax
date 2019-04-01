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
    return pe.PartialVal((AbstractTuple(avals), core.JaxprTracerTuple(consts)))

def transpose_jaxpr(jaxpr, avals, tangent_components):
  assert False


@curry
def jaxpr_as_fun(jaxpr, consts, *args):
  return core.eval_jaxpr(jaxpr, consts, (), *args)


def call_initial(f, *args):
  pvals = map(_abstractify, args)
  avals = [aval for (aval, _) in pvals]
  jaxpr, pval_out, consts = pe.trace_to_jaxpr(
      lu.wrap_init(f), pvals, instantiate=True)
  return call_initial_p.bind(core.pack(consts), *args, jaxpr=jaxpr)

def _call_initial_impl(consts, *args, **kwargs):
  jaxpr = kwargs.pop('jaxpr')
  return jaxpr_as_fun(jaxpr)(consts, *args)

def _call_initial_jvp(primals, tangents, jaxpr):
  avals = [aval for (aval, _) in map(_abstractify, primals)]
  zeros = map(ad.get_zeros, tangents)
  jaxpr_jvp, consts, zeros_out = ad.jvp_jaxpr(jaxpr, avals, zeros)
  primal_out, tangent_out = call_initial_p.bind(core.pack(consts),
                                                core.pack(primals),
                                                core.pack(tangents),
                                                jaxpr=jaxpr_jvp)
  tangent_out_zeros = ad.put_zeros(ad.TangentTuple, zeros_out, tangent_out)
  return primal_out, tangent_out_zeros

def _call_initial_partial_eval(trace, *tracers, **kwargs):
  jaxpr = kwargs.pop('jaxpr')
  in_pvs, in_consts = unzip2([t.pval for t in tracers])
  first_components = map(is_const, in_pvs)
  avals = map(as_aval, in_pvs, in_consts)
  jaxpr_1, jaxpr_2, aval_out, first_components_out = partial_eval_jaxpr(
      jaxpr, avals, first_components)
  call_initial_p.bind(*in_consts, jaxpr=jaxpr_2)
  eqn = core.JaxprEqn(tracers, None, call_initial_p, (), False, dict(jaxpr=jaxpr_2))
  return pe.JaxprTracer(self, PartialVal((aval_out, out_const)), eqn)


def _call_initial_transpose():
  assert False

call_initial_p = core.Primitive("scan")
call_initial_p.def_impl(_call_initial_impl)
ad.primitive_jvps[call_initial_p] = _call_initial_jvp
pe.custom_partial_eval_rules[call_initial_p] = _call_initial_partial_eval
