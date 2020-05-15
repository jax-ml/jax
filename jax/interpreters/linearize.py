from collections import namedtuple

from jax import core
from jax import ad_util
from jax.util import unzip2, toposort, safe_map, safe_zip
from jax.linear_util import wrap_init

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
    return LinTracer(self, val, ZeroLike(val))

  lift = pure

  def sublift(self, tracer):
    assert False  # TODO

  def instantiate_const(self, val):
    return LinTracer(self, None, ConstVar(val))

  def process_primitive(self, primitive, tracers, params):
    primals_in, recipes_in = unzip2((t.primal, t.tangent_recipe) for t in tracers)
    jvp_rule = ad.primitive_jvps[primitive]
    primal_out, jaxpr, consts = unzip_jvp_rule(jvp_rule, primals_in, params)
    const_tracers = map(self.instantiate_const, consts)
    tracer_out = LinTracer(self, primal_out, None)
    recipe_out = SubJaxpr(object(), [*const_tracers, *tracers], [tracer_out], jaxpr)
    tracer_out.tangent_recipe = recipe_out
    return tracer_out

def unzip_jvp_rule(jvp_rule, primals_in, params):
    @wrap_init
    def linearized(*tangents_in):
      primal_out, tangent_out = jvp_rule(primals_in, tangents_in, **params)
      tangent_out = ad.instantiate_zeros(primal_out, tangent_out)
      return primal_out, tangent_out

    tangent_pvals_in = [pe.PartialVal.unknown(core.get_aval(p).at_least_vspace())
                        for p in primals_in]
    jaxpr, pvals_out, consts = pe.trace_to_jaxpr(linearized, tangent_pvals_in)
    primal_pval_out, tangent_pval_out = pvals_out
    assert primal_pval_out.is_known()
    primal_out = primal_pval_out.get_known()
    assert len(jaxpr.outvars) == 2 and jaxpr.outvars[0] is core.unitvar
    jaxpr.outvars = jaxpr.outvars[1:]
    return primal_out, jaxpr, consts

def tracers_to_jaxpr(in_tracers, out_tracers):
  newvar = core.gensym('')

  t_to_var = {}
  def getvar(t):
    var = t_to_var.get(id(t))
    if var is None:
      var = t_to_var[id(t)] = newvar(t.aval)
    return var

  const_to_var = {}
  def getconstvar(c):
    var = const_to_var.get(id(c))
    if var is None:
      aval = core.raise_to_shaped(core.get_aval(c))
      var = const_to_var[id(c)] = newvar(aval)
    return var

  sorted_tracers = toposort(out_tracers)
  invars = map(getvar, in_tracers)
  eqns = []
  consts = {}
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
      v = t_to_var[id(t)] = getconstvar(recipe.val)
      consts[v] = recipe.val
    else:
      raise TypeError(recipe)

  const_vars, const_vals = unzip2(consts.items())
  jaxpr = core.Jaxpr((), (*constvars, *invars), map(getvar, out_tracers), eqns)
  core.skip_checks or core.check_jaxpr(jaxpr)
  return jaxpr, const_vals
