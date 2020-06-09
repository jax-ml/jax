from typing import Callable
from functools import partial
from enum import Enum
import itertools as it

import jax
from jax import core
from jax import linear_util as lu
from . import ad
from . import partial_eval as pe
from .partial_eval import (PartialVal, partial_eval_jaxpr, _dce_jaxpr, _reconstruct_pval,
                           JaxprTracer, ConstVar, convert_constvars_jaxpr, new_eqn_recipe)
from ..core import raise_to_shaped, get_aval, unitvar, abstract_unit, Literal, Jaxpr
from ..custom_derivatives import _initial_style_jaxpr, _resolve_kwargs
from ..abstract_arrays import ConcreteArray
from ..api_util import flatten_fun_nokwargs
from ..tree_util import tree_flatten, tree_unflatten
from ..util import safe_map, safe_zip, unzip2, split_list, cache

map = safe_map
zip = safe_zip

################################################################################
# Reverse call primitive
################################################################################

invertible_call_p = core.Primitive('invertible_call')
invertible_call_p.call_primitive = True
invertible_call = partial(core.call_bind, invertible_call_p)
invertible_call_p.def_custom_bind(invertible_call)
invertible_call_p.def_impl(core.call_impl)
invertible_call_p.multiple_results = True

def _invertible_call_make_output_tracers(trace, typed_jaxpr, in_tracers, out_known_pvals, out_unknown_pvals, _, params):
  unknown_output_tracers = [JaxprTracer(trace, out_pval, None) for out_pval in out_unknown_pvals]
  lifted_jaxpr = convert_constvars_jaxpr(typed_jaxpr.jaxpr)
  # Add dummy arguments representing the outputs to the jaxpr. Those should remain unused in case
  # the expression actually ends up being evaluated, but they make it well-formed.
  out_known_avals = tuple(raise_to_shaped(get_aval(pval.get_known())) for pval in out_known_pvals)
  lifted_jaxpr = _append_invars(lifted_jaxpr, out_known_avals)
  new_params = dict(params, call_jaxpr=lifted_jaxpr)
  # We also append some dummy outputs that correspond to the known outputs we left in the call_jaxpr
  dummy_outputs = [JaxprTracer(trace, pval, core.unit) for pval in out_known_pvals]

  output_constants = [JaxprTracer(trace, pval, ConstVar(pval.get_known())) for pval in out_known_pvals]
  eqn = new_eqn_recipe(tuple(it.chain(in_tracers, output_constants)),
                       dummy_outputs + unknown_output_tracers,
                       invertible_call_p,
                       new_params)
  for t in unknown_output_tracers: t.recipe = eqn
  return unknown_output_tracers

pe.call_partial_eval_rules[invertible_call_p] = partial(
    pe._remat_partial_eval, _invertible_call_make_output_tracers)


@cache()
def _append_invars(jaxpr, avals):
  newvar = core.gensym([jaxpr])
  return core.Jaxpr(jaxpr.constvars, jaxpr.invars + map(newvar, avals), jaxpr.outvars, jaxpr.eqns)


def _invertible_call_transpose(params, call_jaxpr, args, ct, _):
  # TODO: Is the vjp invertible too? In principle yes, because the derivative is
  #       a linear transform with coefficients derived from the primal, but do we
  #       want to preserve this annotation?

  # All this code is an awkward attempt to inverse engineer the structure of our
  # arguments in a way that lets us separate primal arguments and constants from the
  # primal outputs. We need to do that, because even though the jaxpr has arguments
  # corresponding to the primal outputs, we need to fill them in in the primal environment
  # under the names corresponding to the outvars of the jaxpr.
  # The general idea here is that due to the way linearization works, all tangent
  # arguments always come after all the primal args. Additionally, _inverse_partial_eval
  # appends the constants corresponding to primal outputs as trailing arguments.
  # In the end we expect is_tangent to be of the form:
  #   [False, ..., False, True, ..., True, False, ..., False]
  # where the first primal block gives us the constants and regular primal args,
  # while the second block gives us the saved primal outputs.
  is_tangent = map(ad.is_undefined_primal, args)
  first_tangent = is_tangent.index(True)
  last_tangent = (len(is_tangent) - 1) - is_tangent[::-1].index(True)
  # Make sure that there are some primals before and after the tangent segment
  assert first_tangent > 0 and last_tangent < len(is_tangent)
  # Make sure that the tangents form a contiguous range
  assert all(is_tangent[first_tangent:last_tangent + 1])
  outputs = args[last_tangent + 1:]
  return inv_backward_pass(call_jaxpr, (), args, outputs, ct)
ad.primitive_transposes[invertible_call_p] = _invertible_call_transpose

################################################################################
# Custom inverse
################################################################################

class custom_ivjp:
  def __init__(self, fun):
    self.fun = fun
    self.ivjp = None

  def defivjp(self, ivjp):
    # ivjp(inputs, outputs, output_cotangents) -> (inputs, input_cotangents)
    self.ivjp = ivjp

  def __call__(self, *args, **kwargs):
    if self.ivjp is None:
      msg = "No VJP defined for custom_vjp function {}. Did you forget to use defivjp?"
      raise AttributeError(msg.format(self.__name__))
    args = _resolve_kwargs(self.fun, args, kwargs)
    # TODO: Support nondiff_argnums
    fun, ivjp = lu.wrap_init(self.fun), lu.wrap_init(self.ivjp)
    args_flat, in_tree = tree_flatten(args)
    flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)
    flat_ivjp = _flatten_ivjp(ivjp, in_tree, out_tree)
    out_flat = _custom_ivjp(flat_fun, flat_ivjp, args_flat)
    return tree_unflatten(out_tree(), out_flat)

def zip_with(fun, *args):
  return map(lambda p: fun(*p), zip(*args))

@lu.transformation
def _flatten_ivjp(in_tree, out_tree, *args):
  out_tree = out_tree()
  num_inputs, num_outputs = in_tree.num_leaves, out_tree.num_leaves
  assert len(args) == num_inputs + 2 * num_outputs
  arg_leaves = split_list(args, [num_inputs, num_outputs])
  py_args = zip_with(tree_unflatten, [in_tree, out_tree, out_tree], arg_leaves)
  pair_out = yield py_args, {}
  if not isinstance(pair_out, (list, tuple)) or len(pair_out) != 2:
    raise TypeError("Expected a two element pair as output of custom ivjp")
  yield tree_flatten(pair_out)[0]

def _custom_ivjp(fun, ivjp, args):
  in_avals = [raise_to_shaped(get_aval(x)) for x in args]
  fun_jaxpr = _initial_style_jaxpr(fun, in_avals)
  try:
    ivjp_jaxpr = _initial_style_jaxpr(ivjp, in_avals + fun_jaxpr.out_avals * 2)
  except RecursionError:
    raise ValueError("Calls to {} from its custom ivjp aren't supported yet".format(fun.__name__))
  return custom_ivjp_p.bind(*args, fun_jaxpr=fun_jaxpr,
                                   ivjp_jaxpr=ivjp_jaxpr)

def _custom_ivjp_impl(*args, fun_jaxpr, **_):
  return core.jaxpr_as_fun(fun_jaxpr)(*args)

custom_ivjp_p = core.Primitive('custom_ivjp')
custom_ivjp_p.multiple_results = True
custom_ivjp_p.def_impl(_custom_ivjp_impl)
custom_ivjp_p.def_abstract_eval(lambda *_, fun_jaxpr, **__: fun_jaxpr.out_avals)

def _custom_ivjp_jvp(primals, tangents, *, fun_jaxpr, ivjp_jaxpr):
  primals_out = custom_ivjp_p.bind(*primals, fun_jaxpr=fun_jaxpr,
                                             ivjp_jaxpr=ivjp_jaxpr)
  fun = core.jaxpr_as_fun(fun_jaxpr)
  # FIXME: This might compute the primals multiple times, but we only need to do
  #        this trick while linearizing. It should be possible to do it through
  #        a custom partial eval rule.
  _, tangents_out = ad.jvp(lu.wrap_init(fun)).call_wrapped(primals, tangents)
  return primals_out, tangents_out
ad.primitive_jvps[custom_ivjp_p] = _custom_ivjp_jvp

################################################################################
# Backward pass implementation
################################################################################

def inv_backward_pass(jaxpr: core.Jaxpr, consts, primals_in, primals_out, cotangents_in):
  if all(type(ct) is ad.Zero for ct in cotangents_in):
    return zero_vars(jaxpr.invars)

  def write_cotangent(v, ct):
    # assert v not in primal_env
    if ct is not None and type(v) is not Literal:
      ct_env[v] = ad.add_tangents(ct_env[v], ct) if v in ct_env else ct

  def read_cotangent(v):
    return ct_env.get(v, ad.Zero(v.aval))

  def read_primal(v):
    if type(v) is Literal:
      return v.val
    else:
      return primal_env.get(v, ad.UndefinedPrimal(v.aval))

  def write_primal(v, val):
    if type(v) is Literal:
      return
    primal_env.setdefault(v, val)

  # Structure of arguments is [primal_invars, tangent_invars, unused_primal_outvar_placeholders]
  primal_invars, tangent_invars = split(jaxpr.invars[:-len(primals_out)], parts=2)
  primal_outvars, tangent_outvars = split(jaxpr.outvars, parts=2)

  def is_tangent(var):
    return type(var) is not Literal and var in tangent_vars

  tangent_vars = set(tangent_invars)
  primal_eqns = []
  tangent_eqns = []
  for eqn in jaxpr.eqns:
    if not eqn.primitive.call_primitive:
      if any(map(is_tangent, eqn.invars)):
        tangent_eqns.append(eqn)
        tangent_vars.update(eqn.outvars)
      else:
        primal_eqns.append(eqn)
    else:
      assert False

  # Invert while computing the cotangents
  ct_env: Dict[Any, Any] = {}
  primal_env: Dict[Any, Any] = {}
  write_primal(core.unitvar, core.unit)
  map(write_primal, jaxpr.invars, primals_in)
  map(write_primal, jaxpr.outvars[:len(primals_out)], primals_out)
  map(write_cotangent, primal_outvars, split(cotangents_in, parts=2)[1])
  for eqn in primal_eqns[::-1]:
    primals_in = map(read_primal, eqn.invars)
    primals_out = map(read_primal, eqn.outvars)
    cts_in = map(read_cotangent, eqn.outvars)
    should_invert = any(type(primal) is not ad.UndefinedPrimal
                        for primal in primals_out)
    should_vjp = any(type(ct) is not ad.Zero for ct in cts_in)
    assert not eqn.primitive.call_primitive
    assert not (should_invert ^ should_vjp)  # Either both true or both false

    # Skip primals equations that are only jvp coefficients and don't affect
    # primal outputs.
    if not should_invert and not should_vjp:
      continue

    def abstract(value):
      return raise_to_shaped(value.aval if ad.is_undefined_primal(value) else get_aval(value))

    # Get the ivjp_jaxpr
    if eqn.primitive is custom_ivjp_p:
      ivjp_jaxpr = eqn.params['ivjp_jaxpr']
    else:
      if eqn.primitive in primitive_ivjps:
        complete_ivjp = lu.wrap_init(primitive_ivjps[eqn.primitive])
      else:
        complete_ivjp = lu.wrap_init(partial(synthesize_ivjp, eqn, map(ad.is_undefined_primal, primals_in)))
      _, in_tree = tree_flatten(
          tuple(map(abstract, x) for x in (primals_in, primals_out, primals_out)))
      complete_ivjp_flat, _ = flatten_fun_nokwargs(complete_ivjp, in_tree)

      in_avals = map(abstract, primals_in + primals_out + primals_out)
      ivjp_jaxpr, out_pvals, _ = pe.trace_to_jaxpr(
        complete_ivjp_flat,
        map(PartialVal.unknown, in_avals),
        instantiate=True,
        stage_out=False)
      assert not ivjp_jaxpr.constvars  # That might happen some time, but don't bother until then
      out_avals = map(raise_to_shaped, unzip2(out_pvals)[0])
      ivjp_jaxpr = core.TypedJaxpr(ivjp_jaxpr, [], in_avals, out_avals)

    # Once we know what the ivjp can do exactly, we have to isolate the part we are
    # actually able to compute with the values we have at hand.
    num_inputs = len(eqn.invars)
    unknowns = (map(ad.is_undefined_primal, primals_in) +
                map(ad.is_undefined_primal, primals_out) +
                [False] * len(cts_in))
    jaxpr_known, jaxpr_unknown, out_unknowns = partial_eval_jaxpr(ivjp_jaxpr,
                                                                  unknowns,
                                                                  instantiate=False,
                                                                  trace_type=None)
    unknown_rec_primals_in, unknown_cotangents = split_list(out_unknowns, [num_inputs])
    # Make sure we're able to compute all cotangents. We don't really care if we
    # can reconstruct or primals or not, although failure to do so might result in
    # failing to compute cotangents later.
    assert not any(unknown_cotangents)
    # Remove residual outputs -- we won't be computing the unknown jaxpr anyway.
    jaxpr_known.out_avals = jaxpr_known.out_avals[:num_inputs * 2]
    jaxpr_known.jaxpr.outvars = jaxpr_known.jaxpr.outvars[:num_inputs * 2]
    # TODO: We could drop the outputs that correspond to primals that we already know.
    #       This only matters in eager mode, so leaving it out for now...
    ivjp = core.jaxpr_as_fun(jaxpr_known)
    rec_primals_in, cts_out = split_list(ivjp(*primals_in, *primals_out, *cts_in),
                                         [num_inputs])
    # Unknown rec_primals_in are core.units, so we have to replace them
    # with UnknownPrimals because that's what write_primal will ignore.
    rec_primals_in = [prev if unknown else rec
                      for prev, rec, unknown
                      in zip(primals_in, rec_primals_in, unknown_rec_primals_in)]
    map(write_primal, eqn.invars, rec_primals_in)
    map(write_cotangent, eqn.invars, cts_out)

  # NOTE: We keep the cotangents associated with primal variables, while the contract of a
  #       transpose is to return them in positions associated with tangent variables, which
  #       is what causes this whole confusion.
  return zero_vars(primal_invars) + map(read_cotangent, primal_invars) + zero_vars(primals_out)

def zero_vars(vs):
  return map(lambda v: ad.Zero(v.aval), vs)

primitive_ivjps = {}

def synthesize_ivjp(eqn, unknown_primals, primals_in, primals_out, cts_in):
  # Invert eqn
  if not eqn.primitive.multiple_results:
    primals_out, = primals_out
  rec_primals_in = get_primitive_inverse(eqn.primitive)(primals_out, *primals_in)
  if len(eqn.invars) == 1:
    rec_primals_in = (rec_primals_in,)

  # Use the reconstructed primals if some primals_in were unknown, because we
  # might have reconstructed some of them.
  primals_in = map(lambda p, rp, unknown: rp if unknown else p,
                   primals_in, rec_primals_in, unknown_primals)

  # Compute the VJP of eqn
  variable_invars = [v for v in eqn.invars if type(v) is not Literal]
  variable_primals_in = [p for p, v in zip(primals_in, eqn.invars) if type(v) is not Literal]
  eqn_jaxpr = Jaxpr([], variable_invars, eqn.outvars, [eqn])
  eqn_callable = lambda args: core.eval_jaxpr(eqn_jaxpr, (), *args)
  _, eqn_vjp = jax.vjp(eqn_callable, variable_primals_in)
  # TODO: Instantiate zeros or (better) figure out how to avoid it!
  cts_out, = eqn_vjp(cts_in)

  return rec_primals_in, cts_out

def split(l, parts):
  assert len(l) % parts == 0
  chunk = len(l) // parts
  return [l[i:i + chunk] for i in range(0, len(l), chunk)]

################################################################################
# Primitive inverses
################################################################################

primitive_inverses = {}

def get_primitive_inverse(p):
  try:
    return primitive_inverses[p]
  except KeyError:
    pass
  raise NotImplementedError(
    "Inverse rule for '{}' not implemented".format(p))


def definverse(primitive, inverse_rule):
  primitive_inverses[primitive] = inverse_rule
  return inverse_rule

