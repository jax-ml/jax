from collections import namedtuple

from jax import core
from jax import ad_util
from jax.util import unzip2, unzip3, toposort, safe_map, safe_zip
from jax.linear_util import wrap_init

from . import ad
from . import partial_eval as pe
from ..core import new_jaxpr_eqn, TypedJaxpr
from .. import numpy as jnp


map = safe_map
zip = safe_zip


# TODO: Use weakrefs (for out_tracers?)?
NonLinearSubJaxpr = namedtuple('NonLinearSubJaxpr', ['id', 'in_tracers', 'out_tracers', 'jaxpr'])
SubJaxpr = namedtuple('SubJaxpr', ['id', 'in_nonlin_tracers', 'in_lin_tracers', 'out_tracers', 'jaxpr']) # Some arguments have to be non-linear!
LambdaBinding = namedtuple('LambdaBinding', [])
Zeros = namedtuple('Zeros', ['shape'])
ConstVar = namedtuple('ConstVar', ['val'])


def lin(fun, *primals):
  with core.new_master(LinTrace) as master:
    trace = LinTrace(master, core.cur_sublevel())
    in_tracers = [LinTracer(trace, p, None, LambdaBinding()) for p in primals]
    out = fun.call_wrapped(*in_tracers)
    out_tracer = trace.full_raise(out)
    return tracers_to_jaxpr(in_tracers, [out_tracer])

def rematlin(fun, *primals):
  with core.new_master(LinTrace) as master:
    trace = LinTrace(master, core.cur_sublevel())
    in_tracers = [LinTracer(trace, p, ConstVar(p), LambdaBinding()) for p in primals]
    out = fun.call_wrapped(*in_tracers)
    out_tracer = trace.full_raise(out)
    return tracers_to_jaxpr(in_tracers, [out_tracer])


class LinTracer(core.Tracer):
  __slots__ = ['primal', 'primal_recipe', 'linear_recipe', '_aval']

  def __init__(self, trace, primal, primal_recipe, linear_recipe):
    self._trace = trace
    self.primal = primal
    self._aval = None
    self.primal_recipe = primal_recipe
    self.linear_recipe = linear_recipe  # Recipe for the linearized version of the primal

  def __repr__(self):
    return 'Linearized<{}:{}>'.format(self.aval, self._trace)

  @property
  def aval(self):
    if self._aval is not None:
      return self._aval
    else:
      return core.get_aval(self.primal)

  @property
  def parents(self):
    parents = []
    if isinstance(self.linear_recipe, SubJaxpr):
      parents += self.linear_recipe.in_lin_tracers + self.linear_recipe.in_nonlin_tracers
    if isinstance(self.primal_recipe, NonLinearSubJaxpr):
      parents += self.primal_recipe.in_tracers
    return parents

  def full_lower(self):
    return self


class LinTrace(core.Trace):
  def pure(self, val):
    return LinTracer(self, val, None, Zeros(jnp.shape(val)))

  lift = pure

  def sublift(self, tracer):
    assert False  # TODO

  def instantiate_const(self, val):
    # TODO: Should we keep the val in the primal, or should it be None?
    return LinTracer(self, None, None, ConstVar(val))

  def process_primitive(self, primitive, tracers, params):
    primals_in, primal_recipes_in, _ = unzip3((t.primal, t.primal_recipe, t.linear_recipe) for t in tracers)

    primal_pvals_in = [pe.PartialVal.unknown(core.get_aval(p)) if recipe is not None else pe.PartialVal.known(p)
                       for p, recipe in zip(primals_in, primal_recipes_in)]
    tangent_pvals_in = [pe.PartialVal.unknown(core.get_aval(p).at_least_vspace())
                        for p in primals_in]
    flat_jvp_rule = get_flat_jvp_rule(primitive, params)
    jaxpr, pvals_out, consts = pe.trace_to_jaxpr(flat_jvp_rule, primal_pvals_in + tangent_pvals_in)

    num_args = len(primals_in)
    num_outputs = len(pvals_out) // 2

    typed_jaxpr = TypedJaxpr(jaxpr,
                             consts,
                             map(core.raise_to_shaped, map(core.get_aval, primals_in)) * 2,
                             [p.get_aval() for p in pvals_out])
    typed_primal_jaxpr, typed_linear_jaxpr, out_unknowns = pe.partial_eval_jaxpr(
      typed_jaxpr,
      unknowns=([False] * num_args + [True] * num_args),
      instantiate=False,
      trace_type=None)
    assert out_unknowns == [False] * num_outputs + [True] * num_outputs
    primal_jaxpr = typed_primal_jaxpr.jaxpr
    linear_jaxpr = typed_linear_jaxpr.jaxpr

    # Primal jaxpr outputs only primals and possibly residuals
    primal_jaxpr.invars = primal_jaxpr.invars[:num_args]
    primal_jaxpr.outvars = primal_jaxpr.outvars[:num_outputs] + primal_jaxpr.outvars[num_outputs * 2:]
    residual_avals = map(core.raise_to_shaped, typed_primal_jaxpr.out_avals[num_outputs * 2:])
    # Linear jaxpr only takes in tangents and residulas, and only returns the tangents
    linear_jaxpr.invars = linear_jaxpr.invars[num_args:]
    linear_jaxpr.outvars = linear_jaxpr.outvars[num_outputs:]

    # Wrap results back in tracers
    primal_tracers = [LinTracer(self, None, None, None) for _ in range(num_outputs)]
    residual_tracers = [LinTracer(self, None, None, None) for _ in primal_jaxpr.outvars[num_outputs:]]
    const_tracers = map(self.instantiate_const, consts)

    primal_recipe = NonLinearSubJaxpr(object(),
                                      [*const_tracers, *tracers],
                                      [*primal_tracers, *residual_tracers],
                                      primal_jaxpr)
    linear_recipe = SubJaxpr(object(),
                             residual_tracers,
                             tracers,
                             primal_tracers,
                             linear_jaxpr)

    primal_pvals_out, tangent_pvals_out = split(pvals_out, parts=2)
    for pt, pval in zip(primal_tracers, primal_pvals_out):
      if pval.is_known():
        pt.primal = pval.get_known()
      else:
        pt.primal = pval.get_aval().val  # TODO: This is very ConcreteArray centric...
        pt.primal_recipe = primal_recipe
      pt.linear_recipe = linear_recipe
    for rt, aval in zip(residual_tracers, residual_avals):
      rt.primal_recipe = primal_recipe
      rt._aval = aval

    assert not primitive.multiple_results
    return primal_tracers[0]


def get_flat_jvp_rule(primitive, params):
  jvp_rule = ad.primitive_jvps[primitive]

  @wrap_init
  def flat_jvp_rule(*inputs):
    primals_in, tangents_in = split(inputs, parts=2)
    primal_out, tangent_out = jvp_rule(primals_in, tangents_in, **params)
    # TODO: Why do we need to instantiate zeros here?
    if primitive.multiple_results:
      tangent_out = [ad.instantiate_zeros(p, t) for p, t in zip(primal_out, tangent_out)]
      return primal_out + tangent_out
    else:
      tangent_out = ad.instantiate_zeros(primal_out, tangent_out)
      return primal_out, tangent_out

  return flat_jvp_rule



def unzip_jvp_rule(jvp_rule, primals_in, primal_recipes_in, params):
  return primals_out, primal_recipies_out, jaxpr, consts


def tracers_to_jaxpr(in_tracers, out_tracers):
  newvar = core.gensym('')

  t_to_linvar = {}
  def getlinvar(t):
    var = t_to_linvar.get(id(t))
    if var is None:
      var = t_to_linvar[id(t)] = newvar(t.aval)
    return var

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
  invars = map(getlinvar, in_tracers)
  eqns = []
  consts = {}
  processed_eqn_ids = set()
  for t in sorted_tracers:
    # TODO: Make sure to emit nonlinear recipes too!
    recipe = t.linear_recipe
    if isinstance(recipe, SubJaxpr):
      if recipe.id not in processed_eqn_ids:
        processed_eqn_ids.add(recipe.id)
        eqns.append(
          new_jaxpr_eqn(map(getlinvar, recipe.in_lin_tracers) + map(getvar, recipe.in_nonlin_tracers),
                        map(getlinvar, recipe.out_tracers),
                        core.call_p,
                        dict(call_jaxpr=recipe.jaxpr)))
    elif isinstance(recipe, LambdaBinding):
      pass
    elif isinstance(recipe, Zeros):
      eqns.append(
        new_jaxpr_eqn([Literal(0.)], getlinvar(t),
                      lax.broadcast_in_dim_p,
                      dict(shape=recipe.shape,
                           broadcast_dimensions=range(len(recipe.shape)))))
    elif isinstance(recipe, ConstVar):
      v = t_to_var[id(t)] = getconstvar(recipe.val)
      consts[v] = recipe.val
    elif recipe is None:  # Residual tracers in remat don't have a linear recipe
      pass
    else:
      raise TypeError(recipe)

    recipe = t.primal_recipe
    if recipe is None:  # Nonlinear recipes are not used outside of remat
      pass
    elif isinstance(recipe, NonLinearSubJaxpr):
      if recipe.id not in processed_eqn_ids:
        processed_eqn_ids.add(recipe.id)
        eqns.append(
          new_jaxpr_eqn(map(getvar, recipe.in_tracers),
                        map(getvar, recipe.out_tracers),
                        core.call_p,
                        dict(call_jaxpr=recipe.jaxpr)))
    elif isinstance(recipe, ConstVar):
      v = t_to_var[id(t)] = getconstvar(recipe.val)
      consts[v] = recipe.val
    else:
      raise TypeError(recipe)

  constvars, constvals = unzip2(consts.items())
  jaxpr = core.Jaxpr(constvars, invars, map(getlinvar, out_tracers), eqns)
  core.skip_checks or core.check_jaxpr(jaxpr)
  return jaxpr, constvals


def split(l, parts):
  assert len(l) % parts == 0
  chunk = len(l) // parts
  return [l[i:i + chunk] for i in range(0, len(l), chunk)]
