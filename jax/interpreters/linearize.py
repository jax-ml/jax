from collections import namedtuple

from jax import core
from jax import ad_util
from jax.util import unzip2, toposort, safe_map, safe_zip

from . import ad
from . import partial_eval as pe

map = safe_map
zip = safe_zip


SubJaxpr = namedtuple('SubJaxpr', ['id', 'invars', 'outvars', 'jaxpr'])
LambdaBinding = namedtuple('SubJaxpr', [])
ZeroLike = namedtuple('ZeroLike', ['val'])
ConstVar = namedtuple('ConstVar', ['val'])


def lin(fun, *primals):
  with core.new_master(LinTrace) as master:
    trace = LinTrace(master, core.cur_sublevel())
    in_tracers = [LinTracer(trace, p, LambdaBinding()) for p in primals]
    out = fun.call_wrapped(*in_tracers)
    out_tracer = trace.full_raise(out)
    tracers_to_jaxpr(in_tracers, [out_tracer])


class LinTracer(core.Tracer):
  __slots__ = ['primal', 'tangent_recipe']

  def __init__(self, trace, primal, tangent_recipe):
    self._trace = trace
    self.primal = primal
    self.tangent_recipe = tangent_recipe

  @property
  def aval(self):
    return core.get_aval(self.primal)

  @property
  def parents(self):
    if isinstance(self.tangent_recipe, SubJaxpr):
      return self.tangent_recipe.invars
    else:
      return []

  def full_lower(self):
    return self

class LinTrace(core.Trace):
  def pure(self, val):
    out = LinTracer(self, val, ZeroLike(val))

  def lift(self, val):
    return self.pure(val)

  def sublift(self, tracer):
    assert False  # TODO

  def process_primitive(self, primitive, tracers, params):
    primals_in, recipes_in = unzip2((t.primal, t.tangent_recipe) for t in tracers)
    pvals_in = [pe.PartialVal.unknown(core.get_aval(p).at_least_vspace())
                for p in primals_in]
    jaxpr_trace = pe.JaxprTrace(self.master, self.sublevel)
    tangents_in = [pe.JaxprTracer(jaxpr_trace, pval, pe.LambdaBinding())
                   for pval in pvals_in]
    jvp_rule = ad.primitive_jvps[primitive]
    primal_out, tangent_out = jvp_rule(primals_in, tangents_in, **params)
    jaxpr, consts, env = pe.tracers_to_jaxpr(tangents_in, [tangent_out])
    assert not env and not consts
    del tangents_in, tangent_out, consts, env
    tracer_out = LinTracer(self, primal_out, None)
    recipe_out = SubJaxpr(object(), tracers, [tracer_out], jaxpr)
    tracer_out.tangent_recipe = recipe_out
    return tracer_out


def tracers_to_jaxpr(in_tracers, out_tracers):
  newvar = core.gensym('')
  t_to_var = {}
  def getvar(t):
    var = t_to_var.get(id(t))
    if var is None:
      var = t_to_var[id(t)] = newvar(t.aval)
    return var
  sorted_tracers = toposort(out_tracers)
  invars = map(getvar, in_tracers)
  eqns = []
  processed_eqn_ids = set()
  for t in sorted_tracers:
    recipe = t.tangent_recipe
    if isinstance(recipe, SubJaxpr):
      if recipe.id not in processed_eqn_ids:
        breakpoint()  # TODO
        processed_eqn_ids.add(recipe.id)
    elif isinstance(recipe, LambdaBinding):
      pass
    elif isinstance(recipe, ZeroLike):
      assert False  # TODO
    elif isinstance(recipe, ConstVar):
      assert False  # TODO
    else:
      raise TypeError(recipe)

  jaxpr = core.Jaxpr((), invars, map(getvar, out_tracers), eqns)
  core.skip_checks or core.check_jaxpr(jaxpr)
  return jaxpr
