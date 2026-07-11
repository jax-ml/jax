# Copyright 2024 The JAX Authors.
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

from collections.abc import Mapping
from functools import partial, wraps
from typing import Any

from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import flattree as ft
from jax._src import linear_util as lu
from jax._src import tree_util
from jax._src import xla_metadata_lib
from jax._src.api_util import debug_info
from jax._src.interpreters import ad, batching, mlir, partial_eval as pe
from jax._src.lib import _jax
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.tree_util import tree_flatten, tree_leaves, tree_unflatten
from jax._src.util import (safe_map, safe_zip, weakref_lru_cache, unzip2,
                           split_list, subs_list)

config_ext = _jax.config

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


class _XlaMetadataWrapper:
  """A wrapper class to allow XlaMetadataContextManager to be used as a decorator.

  When XlaMetadataContextManager is used as a decorator on a function `f`, it
  returns an instance of this class. This wrapper ensures that when `f` is
  called, it runs within the metadata context. It also forwards attribute
  access to `f` via `__getattr__`, and if an attribute of `f` is callable (e.g.,
  the `.lower()` method of a jitted function), it wraps that attribute so it
  too runs within the metadata context when called. This allows decorated
  functions to be used seamlessly with JAX transformations like `jax.jit`.
  """

  def __init__(self, f, ctx):
    self._f = f
    self._ctx = ctx
    wraps(f)(self)

  def __call__(self, *args, **kwargs):
    with self._ctx:
      return self._f(*args, **kwargs)

  def __getattr__(self, name):
    attr = getattr(self._f, name)
    if not callable(attr):
      return attr

    @wraps(attr)
    def wrapper(*args, **kwargs):
      with self._ctx:
        return attr(*args, **kwargs)

    return wrapper


class XlaMetadataContextManager:
  __slots__ = ["prev", "updates"]

  def __init__(self, updates):
    self.updates = updates

  def __enter__(self):
    if not self.updates:
      return

    self.prev = config.xla_metadata_context_manager.get_local()
    config.xla_metadata_context_manager.set_local(
        xla_metadata_lib.update_metadata(self.prev, self.updates)
    )

  def __exit__(self, exc_type, exc_value, traceback):
    if not self.updates:
      return
    config.xla_metadata_context_manager.set_local(self.prev)

  def __call__(self, f):
    return _XlaMetadataWrapper(f, self)


def set_xla_metadata(x=None, **kwargs):
  if x is None:
    return XlaMetadataContextManager(kwargs)
  else:
    hashable_metadata = tuple(sorted(kwargs.items()))
    return tree_util.tree_map(
        lambda v: xla_metadata_value_p.bind(
            v, xla_metadata_kvs=hashable_metadata
        ),
        x,
    )


# `xla_metadata_value_p` is an identity primitive for attaching frontend_attributes
# to the primitive's producing (parent/owner) op.
xla_metadata_value_p = core.Primitive("xla_metadata_value")
xla_metadata_value_p.def_impl(
    partial(dispatch.apply_primitive, xla_metadata_value_p)
)
xla_metadata_value_p.def_abstract_eval(lambda aval, *, xla_metadata_kvs: aval)
batching.defvectorized(xla_metadata_value_p)
# TODO(nbasile): Implement tagging gradient ops with metadata.
ad.deflinear2(xla_metadata_value_p, lambda ct, _, **kwargs: (ct,))


def _xla_metadata_value_lowering_rule(
    ctx: mlir.LoweringRuleContext, val: ir.Value, *, xla_metadata_kvs
):
  xla_metadata = dict(xla_metadata_kvs)
  op_to_attach_metadata = _target_op_to_attach_metadata(val)
  if op_to_attach_metadata is not None:
    _attach_xla_metadata_to_op(xla_metadata, op_to_attach_metadata)
  return [val]


# If we leave `cacheable=True`, when we are in the lowering rule, the `val.owner`
# becomes a cached `FuncOp`. FuncOp.owners are Blocks, which we can't tag.
mlir.register_lowering(
    xla_metadata_value_p, _xla_metadata_value_lowering_rule, cacheable=False
)


def _target_op_to_attach_metadata(value_mlir: ir.Value) -> ir.Operation | None:
  op = value_mlir.owner
  if op is None or isinstance(op, ir.Block):
    return None
  return op.operation


def _attach_xla_metadata_to_op(
    xla_metadata: dict[str, Any], op: ir.Operation
) -> None:
  if xla_metadata:
    ctx_attributes, existing_attributes = {}, {}
    for k, v in xla_metadata.items():
      v_str = str(v).lower() if isinstance(v, bool) else str(v)
      ctx_attributes[k] = ir.StringAttr.get(v_str)
    # Combine with existing mhlo.frontend_attributes
    for attr in op.attributes:
      if attr == "mhlo.frontend_attributes":
        for a in ir.DictAttr(op.attributes[attr]):
          existing_attributes[a.name] = a.attr
    op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
        ctx_attributes | existing_attributes
    )


def xla_metadata_call(f=None, /, **meta):
  """Wraps a function so it lowers to a call op tagged with XLA metadata.

  Sugar for :func:`xla_metadata_call2` with the metadata passed as keyword
  arguments and default options. The wrapped function is staged out as a
  separate computation, invoked via a call op annotated with the given
  metadata as ``frontend_attributes``. XLA propagates the attributes to the
  ops inside the call when it inlines it.

  Unlike the ``set_xla_metadata`` context manager, this does not perturb the
  tracing context, so it doesn't cause retracing or recompilation of jitted
  functions. The metadata also follows the computation through
  transformations: under ``jax.grad``, both the forward- and backward-pass
  computations derived from the function carry the metadata. To leave the
  backward pass untagged, or tag it differently, use
  :func:`xla_metadata_call2` and its ``ad_metadata`` option.

  Args:
    f: the function to wrap. If not given, returns a decorator.
    **meta: metadata to attach, as keyword arguments. Values may be strings,
      bools, ints, or floats; they are attached as strings (with bools
      rendered as ``"true"``/``"false"``).

  Returns:
    A wrapped version of ``f`` with the metadata applied, or a decorator.

  Example:

    >>> import jax, jax.numpy as jnp
    >>> from jax.experimental.xla_metadata import xla_metadata_call
    >>> @xla_metadata_call(tag="my_block")
    ... def f(x):
    ...   return jnp.sin(x) * jnp.cos(x)
  """
  if f is None:
    return lambda g: _xla_metadata_call(g, meta, 'same')
  return _xla_metadata_call(f, meta, 'same')


def xla_metadata_call2(f=None, /, metadata=None, *, ad_metadata='same'):
  """Like :func:`xla_metadata_call`, with the metadata as a dict plus options.

  Args:
    f: the function to wrap. If not given, returns a decorator.
    metadata: a dict of metadata to attach to the call op as
      ``frontend_attributes``. Values may be strings, bools, ints, or floats;
      they are attached as strings (with bools rendered as
      ``"true"``/``"false"``).
    ad_metadata: metadata for the computations that autodiff derives from the
      function, beyond the forward pass: the linearized (tangent) and
      transposed (backward-pass) computations. The default ``'same'`` attaches
      ``metadata`` to them too; ``'drop'`` leaves them untagged; a dict
      attaches that metadata instead. Under forward-mode ``jax.jvp``, the
      primal and tangent computations are staged out fused, so they keep
      ``metadata`` regardless.

  Returns:
    A wrapped version of ``f`` with the metadata applied, or a decorator.

  Example:

    >>> import jax, jax.numpy as jnp
    >>> from jax.experimental.xla_metadata import xla_metadata_call2
    >>> @xla_metadata_call2({"scheduling_group": "l3"},
    ...                     ad_metadata={"scheduling_group": "l3_bwd"})
    ... def layer3(x):
    ...   return jnp.sin(x) * jnp.cos(x)
  """
  if isinstance(f, Mapping) and metadata is None:
    f, metadata = None, f
  if f is not None and not callable(f):
    raise TypeError(f"expected a callable to wrap, got {f!r}")
  if f is None:
    return lambda g: _xla_metadata_call(g, metadata, ad_metadata)
  return _xla_metadata_call(f, metadata, ad_metadata)


def _canonicalize_metadata(meta):
  canonical = {}
  for k, v in meta.items():
    if isinstance(v, bool):
      canonical[k] = str(v).lower()
    elif isinstance(v, (str, int, float)):
      canonical[k] = str(v)
    else:
      raise TypeError(
          "xla_metadata_call metadata values must be str, bool, int, or "
          f"float, got {type(v)} for key {k!r}")
  return tuple(sorted(canonical.items()))

def _canonicalize_ad_metadata(ad_metadata):
  if ad_metadata == 'same':
    return 'same'
  elif ad_metadata == 'drop':
    return ()
  elif isinstance(ad_metadata, Mapping):
    return _canonicalize_metadata(ad_metadata)
  else:
    raise TypeError(
        "ad_metadata must be 'same', 'drop', or a dict of metadata, got "
        f"{ad_metadata!r}")

# TODO(yashkatariya): Figure out a way to reuse code with compute_on2_p, fused_p
def _xla_metadata_call(fun, metadata, ad_metadata):
  if metadata is not None and not isinstance(metadata, Mapping):
    raise TypeError(f"metadata must be a dict, got {metadata!r}")
  metadata_kvs = _canonicalize_metadata(metadata or {})
  ad_metadata_kvs = _canonicalize_ad_metadata(ad_metadata)
  @wraps(fun)
  def wrapped(*args, **kwargs):
    dbg = debug_info('xla_metadata_call', fun, args, kwargs)
    args_ft = ft.flatten((args, kwargs))
    in_avals = args_ft.map(core.shaped_abstractify)
    jaxpr, out_avals = pe.trace_to_jaxpr(fun, in_avals, dbg)
    if any(isinstance(c, core.Tracer) for c in jaxpr.consts):
      jaxpr, consts = pe.separate_consts(jaxpr)
    else:
      consts = []
    outs_flat = xla_metadata_call_p.bind(*consts, *args_ft.vals, jaxpr=jaxpr,
                                         xla_metadata=metadata_kvs,
                                         ad_metadata=ad_metadata_kvs)
    return tree_unflatten(out_avals.tree, outs_flat)
  return wrapped

xla_metadata_call_p = core.Primitive('xla_metadata_call')
xla_metadata_call_p.multiple_results = True
dispatch.simple_impl(xla_metadata_call_p)


def _xla_metadata_call_abstract_eval(*in_avals, jaxpr, xla_metadata,
                                     ad_metadata):
  return jaxpr.out_avals
xla_metadata_call_p.def_abstract_eval(_xla_metadata_call_abstract_eval)


def _resolve_ad_metadata(xla_metadata, ad_metadata):
  return xla_metadata if ad_metadata == 'same' else ad_metadata


def _xla_metadata_call_lowering(ctx, *args, jaxpr, xla_metadata, ad_metadata):
  const_args_and_avals = core.jaxpr_const_args(jaxpr)
  const_args, const_avals = unzip2(const_args_and_avals)
  in_avals = (*const_avals, *jaxpr.in_avals)
  func_op, output_types, effects = mlir.lower_called_computation(
      "xla_metadata_call", jaxpr, ctx.module_context, len(const_args), in_avals,
      ctx.avals_out, ctx.tokens_in)

  symbol_name = func_op.name.value
  flat_output_types, treedef = mlir.ir_tree_registry.flatten(output_types)
  tokens = [ctx.tokens_in.get(eff) for eff in effects]
  hoisted_const_values, _ = mlir.ir_tree_registry.flatten([
      mlir.ir_constants(c, const_lowering=ctx.const_lowering, aval=aval)
      for c, aval in const_args_and_avals
  ])
  args = (*ctx.dim_var_values, *tokens, *hoisted_const_values, *args)
  flat_args, _ = mlir.ir_tree_registry.flatten(args)
  call = func_dialect.CallOp(
      flat_output_types, ir.FlatSymbolRefAttr.get(symbol_name),
      flat_args)
  if xla_metadata:
    call.operation.attributes['mhlo.frontend_attributes'] = ir.DictAttr.get(
        {k: ir.StringAttr.get(v) for k, v in xla_metadata})
  out_nodes = treedef.unflatten(call.results)
  tokens, out_nodes = split_list(out_nodes, [len(effects)])
  tokens_out = ctx.tokens_in.update_tokens(mlir.TokenSet(dict(zip(effects, tokens))))
  ctx.set_tokens_out(tokens_out)
  return out_nodes
mlir.register_lowering(xla_metadata_call_p, _xla_metadata_call_lowering)


def _xla_metadata_call_batcher(axis_data, vals_in, dims_in, *, jaxpr,
                               xla_metadata, ad_metadata):
  batched_jaxpr, dims_out = batching.batch_jaxpr2(jaxpr, axis_data, dims_in)
  outs = xla_metadata_call_p.bind(*vals_in, jaxpr=batched_jaxpr,
                                  xla_metadata=xla_metadata,
                                  ad_metadata=ad_metadata)
  return outs, dims_out
batching.fancy_primitive_batchers[xla_metadata_call_p] = _xla_metadata_call_batcher


def _xla_metadata_call_jvp(primals, tangents, *, jaxpr, xla_metadata,
                           ad_metadata):
  # The jvp jaxpr fuses primal and tangent ops, so it keeps the primal
  # metadata; ad_metadata still governs any later linearization/transposition.
  nzs = [not isinstance(t, ad.Zero) for t in tangents]
  jaxpr_jvp, out_nzs = ad.jvp_jaxpr(jaxpr, nzs, False)
  nz_tangents = [t for t in tangents if not isinstance(t, ad.Zero)]
  outs = xla_metadata_call_p.bind(*primals, *nz_tangents, jaxpr=jaxpr_jvp,
                                  xla_metadata=xla_metadata,
                                  ad_metadata=ad_metadata)
  primals_out, nz_tangents_out = outs[:len(out_nzs)], outs[len(out_nzs):]
  nz_outs = iter(nz_tangents_out)
  tangents_out = [next(nz_outs) if nz else ad.Zero(aval.to_tangent_aval())
                  for aval, nz in zip(jaxpr.out_avals, out_nzs)]
  assert next(nz_outs, None) is None
  return primals_out, tangents_out
ad.primitive_jvps[xla_metadata_call_p] = _xla_metadata_call_jvp


def _xla_metadata_call_lin(is_vjp, nzs, *primals, jaxpr, xla_metadata,
                           ad_metadata):
  primal_jaxpr, out_tree, nzs_out, in_fwd_res, tangent_jaxpr = \
      ad.linearize_jaxpr(jaxpr, nzs, is_vjp=is_vjp)
  _, ures_avals, sres_avals = out_tree.unpack()
  num_residuals_out = len(ures_avals) + len(sres_avals)

  tangent_avals_out = [a.to_tangent_aval() for a in jaxpr.out_avals]
  tangent_metadata = _resolve_ad_metadata(xla_metadata, ad_metadata)

  def _filter_zeros(is_nz_l, l):
    return tuple(x for nz, x in zip(is_nz_l, l) if nz)

  def tangent_fun(residuals, structured_residuals, *tangents):
    tangents_nz = _filter_zeros(nzs, tangents)
    sres_flat = tree_leaves(structured_residuals)
    assert (len(residuals) + len(tangents_nz) + len(sres_flat)
            == len(tangent_jaxpr.invars)), (
        len(residuals), len(tangents_nz), len(sres_flat),
        len(tangent_jaxpr.invars))
    nz_outs = xla_metadata_call_p.bind(*residuals, *tangents_nz, *sres_flat,
                                       jaxpr=tangent_jaxpr,
                                       xla_metadata=tangent_metadata,
                                       ad_metadata='same')
    nz_outs_ = iter(nz_outs)
    outs = [next(nz_outs_) if nz else ad.Zero(a)
            for nz, a in zip(nzs_out, tangent_avals_out)]
    assert next(nz_outs_, None) is None
    return outs

  ans = xla_metadata_call_p.bind(*primals, jaxpr=primal_jaxpr,
                                 xla_metadata=xla_metadata,
                                 ad_metadata=ad_metadata)
  primal_ans, residuals_ans = split_list(ans, [len(ans) - num_residuals_out])
  ures, sres_flat = split_list(residuals_ans, [len(ures_avals)])
  ures = subs_list(in_fwd_res, [*jaxpr.consts, *primals], ures)
  sres = sres_avals.update(sres_flat).unflatten()
  return primal_ans, nzs_out, ures, sres, tangent_fun
ad.primitive_linearizations[xla_metadata_call_p] = _xla_metadata_call_lin


@weakref_lru_cache
def _transpose_jaxpr(jaxpr, in_tree, in_avals, specs):
  out_tree = None
  def transposed(*in_flat):
    nonlocal out_tree
    primals_ctrefs, cts_in = tree_unflatten(in_tree, in_flat)
    args = ad.unproject_accums(specs, primals_ctrefs)
    ad.backward_pass3(jaxpr, False, jaxpr.consts, args, cts_in)
    cts_out = [x.freeze() if isinstance(x, ad.ValAccum) else None for x in args]
    cts_out, out_tree = tree_flatten(cts_out)
    return cts_out
  dbg = jaxpr.debug_info.with_unknown_names()
  trans_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(transposed, debug_info=dbg), in_avals)
  return trans_jaxpr.with_consts(consts), out_tree


def _xla_metadata_call_transpose(cts_in, *args, jaxpr, xla_metadata,
                                 ad_metadata):
  primals_ctrefs, specs = ad.project_accums(args)
  in_flat, in_tree = tree_flatten((primals_ctrefs, cts_in))
  in_avals = [core.typeof(x) for x in in_flat]
  trans_jaxpr, out_tree = _transpose_jaxpr(jaxpr, in_tree, (*in_avals,), specs)

  cts_out = xla_metadata_call_p.bind(
      *in_flat, jaxpr=trans_jaxpr,
      xla_metadata=_resolve_ad_metadata(xla_metadata, ad_metadata),
      ad_metadata='same')

  for x, ct in zip(args, tree_unflatten(out_tree, cts_out)):
    if isinstance(x, ad.ValAccum):
      x.accum(ct)


ad.fancy_transposes[xla_metadata_call_p] = _xla_metadata_call_transpose


def _xla_metadata_call_partial_eval_custom_params_updater(
    unks_in,
    inst_in,
    kept_outs_known,
    kept_outs_staged,
    num_res_out,
    num_res_in,
    params_known,
    params_staged,
):
  return params_known, params_staged


pe.partial_eval_jaxpr_custom_rules[xla_metadata_call_p] = partial(
    pe.closed_call_partial_eval_custom_rule,
    'jaxpr',
    _xla_metadata_call_partial_eval_custom_params_updater,
)


def dce_jaxpr_xla_metadata_rule(used_outputs: list[bool], eqn: pe.JaxprEqn
                                ) -> tuple[list[bool], pe.JaxprEqn | None]:
  if not any(used_outputs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None
  dced_jaxpr, used_inputs = pe._cached_closed_call_dce(
      eqn.params['jaxpr'], tuple(used_outputs))
  new_params = dict(eqn.params, jaxpr=dced_jaxpr)
  if not any(used_inputs) and not any(used_outputs) and not dced_jaxpr.effects:
    return used_inputs, None
  else:
    new_invars = [v for v, used in zip(eqn.invars, used_inputs) if used]
    new_effs = core.eqn_effects(dced_jaxpr, new_invars)
    new_eqn = pe.new_jaxpr_eqn(
        new_invars,
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive, new_params, new_effs, eqn.source_info, eqn.ctx)
    return used_inputs, new_eqn
pe.dce_rules[xla_metadata_call_p] = dce_jaxpr_xla_metadata_rule
