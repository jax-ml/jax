from jax import core
from jax.util import unzip2

from . import ad
from . import partial_eval as pe



def lin(fun, *primals):
  with core.new_master(LinTrace) as master:
    lin_trace = LinTrace(master, core.cur_sublevel())
    jaxpr_trace = pe.JaxprTrace(master, core.cur_sublevel())
    in_pvals = [pe.PartialVal.unknown(core.get_aval(x).at_least_vspace())
                for x in primals]
    in_tangents = [pe.JaxprTracer(jaxpr_trace, pval, pe.LambdaBinding())
                   for pval in in_pvals]
    in_tracers = [LinTracer(lin_trace, p, t) for p, t in zip(primals, in_tangents)]
    out = fun.call_wrapped(*in_tracers)
    out_tracer = lin_trace.full_raise(out)
    out_primal, out_tangent = out_tracer.primal, out_tracer.tangent
    jaxpr, consts, env = pe.tracers_to_jaxpr(in_tangents, [out_tangent])
  print(jaxpr)


class LinTracer(ad.JVPTracer):
  pass

class LinTrace(core.Trace):
  def pure(self, val):
    assert False

  def lift(self, val):
    assert False

  def sublift(self, tracer):
    assert False

  def process_primitive(self, primitive, tracers, params):
    primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
    jvp_rule = ad.primitive_jvps[primitive]
    primal_out, tangent_out = jvp_rule(primals_in, tangents_in, **params)
    return LinTracer(self, primal_out, tangent_out)
