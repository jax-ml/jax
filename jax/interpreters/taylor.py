from functools import partial
from collections import Counter

import numpy as onp
from scipy.special import factorial as fact

from jax import core
from jax.util import unzip2, prod
import jax.linear_util as lu


@lu.transformation
def jet(primals, series):
  with core.new_master(JetTrace) as master:
    trace = JetTrace(master, core.cur_sublevel())
    in_tracers = map(partial(JetTracer, trace), primals, series)
    ans = yield in_tracers, {}
    out_tracer = trace.full_raise(ans)  # TODO multiple outputs
    out_primal, series_out = out_tracer.primal, out_tracer.terms
  yield out_primal, series_out


class JetTracer(core.Tracer):
  __slots__ = ["primal", "terms"]

  def __init__(self, trace, primal, terms):
    assert type(terms) in (ZeroSeries, list, tuple)
    self._trace = trace
    self.primal = primal
    self.terms = terms

  @property
  def aval(self):
    return core.get_aval(self.primal)

  def full_lower(self):
    return self  # TODO symbolic zeros

class JetTrace(core.Trace):

  def pure(self, val):
    return JetTracer(self, val, zero_series)

  def lift(self, val):
    return JetTracer(self, val, zero_series)

  def sublift(self, val):
    return JetTracer(self, val.primal, val.terms)

  def process_primitive(self, primitive, tracers, params):
    primals_in, series_in = unzip2((t.primal, t.terms) for t in tracers)
    order, = {len(terms) for terms in series_in if terms is not zero_series}
    series_in = [[zero_term] * order if s is zero_series else s
                 for s in series_in]
    series_in = [[onp.zeros(onp.shape(x), dtype=onp.result_type(x))
                  if t is zero_term else t for t in series]
                 for x, series in zip(primals_in, series_in)]
    rule = jet_rules[primitive]
    primal_out, terms_out = rule(primals_in, series_in, **params)
    return JetTracer(self, primal_out, terms_out)

  def process_call(self, call_primitive, f, tracers, params):
    assert False  # TODO

  def post_process_call(self, call_primitive, out_tracer, params):
    assert False  # TODO

  def join(self, xt, yt):
    assert False  # TODO?


class ZeroTerm(object): pass
zero_term = ZeroTerm()

class ZeroSeries(object): pass
zero_series = ZeroSeries()


jet_rules = {}

def deflinear(prim):
  jet_rules[prim] = partial(linear_prop, prim)

def linear_prop(prim, primals_in, series_in, **params):
  primal_out = prim.bind(*primals_in, **params)
  series_out = [prim.bind(*terms_in, **params) for terms_in in zip(*series_in)]
  return primal_out, series_out
