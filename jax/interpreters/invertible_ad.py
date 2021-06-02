# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from functools import partial
from typing import Dict, Any, Callable

import jax
from jax import core
from jax import linear_util as lu
from . import ad
from . import partial_eval as pe
from ..core import raise_to_shaped, get_aval, Literal, Jaxpr
from ..api_util import flatten_fun_nokwargs
from ..tree_util import tree_flatten, tree_unflatten, register_pytree_node
from .._src.util import safe_map, safe_zip, split_list
from .._src import custom_derivatives

map = safe_map
zip = safe_zip

def _initial_style_jaxpr(fun, in_avals):
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(fun, in_avals)
  return core.ClosedJaxpr(jaxpr, consts)

################################################################################
# Reverse call primitive
################################################################################

class DontFlatten:
  def __init__(self, val):
    self.val = val

register_pytree_node(DontFlatten,
                     lambda x: ((), x.val),
                     lambda val, _: DontFlatten(val))

def get_concrete_array(aval):
  assert isinstance(aval, core.ConcreteArray), aval
  return aval.val

def invertible(fun):
  # TODO: Avoid materializing zeros!
  ifun = custom_derivatives.custom_vjp(fun)

  def fwd(*args):
    flat_args, in_tree = tree_flatten(args)
    in_pvals = tuple(pe.PartialVal.unknown(raise_to_shaped(get_aval(arg))) for arg in flat_args)
    fun_flat, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
    jaxpr, out_pvals, consts = pe.trace_to_jaxpr(fun_flat, in_pvals)
    # TODO: Don't warn if consts contain JVP tracers?
    if consts:
      warnings.warn("Values that an @invertible function closes over will not have their " +
                    "gradients computed correctly (their uses inside this function will be ignored)!")
    # TODO: This requires the body to be jittable, but this shouldn't be necessary.
    #       Is there a way to trace a jaxpr while running it?
    flat_outs = core.eval_jaxpr(jaxpr, consts, *flat_args)
    return tree_unflatten(out_tree(), flat_outs), (flat_args, flat_outs, consts, DontFlatten((jaxpr, in_tree)))

  def bwd(res, cts):
    flat_args, flat_outs, consts, aux = res
    jaxpr, in_tree = aux.val
    flat_cts, _ = tree_flatten(cts)
    return tree_unflatten(in_tree, inv_backward_pass(jaxpr, consts, flat_args, flat_outs, flat_cts))

  ifun.defvjp(fwd, bwd)

  return ifun

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
      msg = "No IVJP defined for custom_vjp function {}. Did you forget to use defivjp?"
      raise AttributeError(msg.format(self.__name__))
    args = custom_derivatives._resolve_kwargs(self.fun, args, kwargs)
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
    return map(lambda v: ad.Zero(v.aval), jaxpr.invars)

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

  # Invert while computing cotangents
  ct_env: Dict[Any, Any] = {}
  primal_env: Dict[Any, Any] = {}
  write_primal(core.unitvar, core.unit)
  map(write_primal, jaxpr.invars, primals_in)
  map(write_primal, jaxpr.outvars, primals_out)
  map(write_primal, jaxpr.constvars, consts)
  map(write_cotangent, jaxpr.outvars, cotangents_in)
  for eqn in jaxpr.eqns[::-1]:
    primals_in = map(read_primal, eqn.invars)
    primals_out = map(read_primal, eqn.outvars)
    cts_in = map(read_cotangent, eqn.outvars)
    should_invert = any(type(primal) is not ad.UndefinedPrimal
                        for primal in primals_out)
    should_vjp = any(type(ct) is not ad.Zero for ct in cts_in)
    assert not eqn.primitive.call_primitive

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
      # TODO: Actually we do know some of the inputs, because they might be literals!
      ivjp_jaxpr, out_pvals, _ = pe.trace_to_jaxpr(
          complete_ivjp_flat, map(pe.PartialVal.unknown, in_avals), instantiate=True)
      assert not ivjp_jaxpr.constvars  # That might happen some time, but don't bother until then
      ivjp_jaxpr = core.ClosedJaxpr(ivjp_jaxpr, [])

    # Once we know what the ivjp can do exactly, we have to isolate the part we are
    # actually able to compute with the values we have at hand.
    num_inputs = len(eqn.invars)
    unknowns = (map(ad.is_undefined_primal, primals_in) +
                map(ad.is_undefined_primal, primals_out) +
                [False] * len(cts_in))
    jaxpr_known, jaxpr_unknown, out_unknowns = pe.partial_eval_jaxpr(  # type: ignore
        ivjp_jaxpr, unknowns, instantiate=False)  # type:ignore
    unknown_rec_primals_in, unknown_cotangents = split_list(out_unknowns, [num_inputs])
    # Make sure we're able to compute all cotangents. We don't really care if we
    # can reconstruct or primals or not, although failure to do so might result in
    # failing to compute cotangents later.
    assert not any(unknown_cotangents)
    # Remove residual outputs -- we won't be computing the unknown jaxpr anyway.
    num_outputs = len(jaxpr_unknown.jaxpr.outvars)
    jaxpr_known.jaxpr.outvars = jaxpr_known.jaxpr.outvars[:num_outputs]
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
    map(write_cotangent, [v for v in eqn.invars if type(v) is not Literal], cts_out)

  # NOTE: We keep the cotangents associated with primal variables, while the contract of a
  #       transpose is to return them in positions associated with tangent variables, which
  #       is what causes this whole confusion.
  return map(read_cotangent, jaxpr.invars)

primitive_ivjps: Dict[core.Primitive, Callable] = {}

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

primitive_inverses: Dict[core.Primitive, Callable] = {}

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
