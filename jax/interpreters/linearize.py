from collections import namedtuple

from jax import core
from jax import ad_util
from jax.util import unzip2, toposort, safe_map, safe_zip
from jax.linear_util import wrap_init

from . import ad
from . import partial_eval as pe
from ..core import new_jaxpr_eqn
from .. import numpy as jnp


map = safe_map
zip = safe_zip


# TODO: Use weakrefs (for out_tracers?)?
SubJaxpr = namedtuple('SubJaxpr', ['id', 'in_tracers', 'out_tracers', 'jaxpr'])
LambdaBinding = namedtuple('LambdaBinding', [])
Zeros = namedtuple('Zeros', ['shape'])
ConstVar = namedtuple('ConstVar', ['val'])


def lin(fun, *primals):
  with core.new_master(LinTrace) as master:
    trace = LinTrace(master, core.cur_sublevel())
    in_tracers = [LinTracer(trace, p, LambdaBinding()) for p in primals]
    out = fun.call_wrapped(*in_tracers)
    out_tracer = trace.full_raise(out)
    return tracers_to_jaxpr(in_tracers, [out_tracer])


class LinTracer(core.Tracer):
  __slots__ = ['primal', 'recipe']

  def __init__(self, trace, primal, recipe):
    self._trace = trace
    self.primal = primal
    self.recipe = recipe  # Recipe for the linearized program

  @property
  def aval(self):
    return core.get_aval(self.primal)

  @property
  def parents(self):
    if isinstance(self.recipe, SubJaxpr):
      return self.recipe.in_tracers
    else:
      return []

  def full_lower(self):
    return self


class LinTrace(core.Trace):
  def pure(self, val):
    return LinTracer(self, val, Zeros(jnp.shape(val)))

  lift = pure

  def sublift(self, tracer):
    assert False  # TODO

  def instantiate_const(self, val):
    # TODO: Should we keep the val in the primal, or should it be None?
    return LinTracer(self, None, ConstVar(val))

  def process_primitive(self, primitive, tracers, params):
    primals_in, recipes_in = unzip2((t.primal, t.recipe) for t in tracers)
    jvp_rule = ad.primitive_jvps[primitive]
    primal_out, jaxpr, consts = unzip_jvp_rule(jvp_rule, primals_in, params)
    const_tracers = map(self.instantiate_const, consts)
    tracer_out = LinTracer(self, primal_out, None)
    recipe_out = SubJaxpr(object(), [*const_tracers, *tracers], [tracer_out], jaxpr)
    tracer_out.recipe = recipe_out
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
    recipe = t.recipe
    if isinstance(recipe, SubJaxpr):
      assert recipe.id not in processed_eqn_ids
      processed_eqn_ids.add(recipe.id)
      eqns.append(
        new_jaxpr_eqn(map(getvar, recipe.in_tracers), map(getvar, recipe.out_tracers),
                      core.call_p, dict(call_jaxpr=recipe.jaxpr)))
    elif isinstance(recipe, LambdaBinding):
      pass
    elif isinstance(recipe, Zeros):
      eqns.append(
        new_jaxpr_eqn([Literal(0.)], getvar(t),
                      lax.broadcast_in_dim_p,
                      dict(shape=recipe.shape,
                           broadcast_dimensions=range(len(recipe.shape)))))
    elif isinstance(recipe, ConstVar):
      v = t_to_var[id(t)] = getconstvar(recipe.val)
      consts[v] = recipe.val
    else:
      raise TypeError(recipe)

  constvars, constvals = unzip2(consts.items())
  jaxpr = core.Jaxpr((), (*constvars, *invars), map(getvar, out_tracers), eqns)
  core.skip_checks or core.check_jaxpr(jaxpr)
  return jaxpr, constvals
