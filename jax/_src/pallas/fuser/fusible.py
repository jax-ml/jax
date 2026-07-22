# Copyright 2025 The JAX Authors.
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

"""Fusible primitive."""
from functools import partial
from typing import Any, Callable

import jax
from jax._src import api_util
from jax._src import core as jax_core
from jax._src import effects
from jax._src import flattree as ft
from jax._src import hijax
from jax._src import linear_util as lu
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax._src.lax.control_flow.loops import eval_jaxpr_p
from jax._src.pallas.fuser import fusible_dtype
from jax._src.pallas.fuser import fusion as fusion_lib
from jax._src.traceback_util import api_boundary


def _positional_effects(jaxpr: jax_core.Jaxpr) -> frozenset[effects.Effect]:
  if not any(isinstance(e, effects.JaxprInputEffect) for e in jaxpr.effects):
    return frozenset(jaxpr.effects)
  idx = {v: i for i, v in enumerate(jaxpr.constvars + jaxpr.invars)}
  return frozenset(jax_core.subst_input_effects(jaxpr.effects, idx))


class Fusible(hijax.VJPHiPrimitive):
  output_fusion_prefix: Any
  func: Callable
  jaxpr: jax_core.Jaxpr
  num_consts: int
  args_tree: tree_util.PyTreeDef

  def __init__(
      self,
      *,
      jaxpr: jax_core.Jaxpr | None,
      in_avals: tuple[Any, ...],
      out_aval: Any,
      output_fusion_prefix: Any,
      func: Callable,
      num_consts: int,
      args_tree: tree_util.PyTreeDef,
  ):
    assert jaxpr is None or isinstance(jaxpr, jax_core.Jaxpr)

    self.in_avals = tuple(in_avals)
    self.out_aval = out_aval
    self.params = {
        "jaxpr": jaxpr,
        "output_fusion_prefix": output_fusion_prefix,
        "func": func,
        "num_consts": num_consts,
        "args_tree": args_tree,
    }
    self.effects = frozenset() if jaxpr is None else _positional_effects(jaxpr)
    super().__init__()

  def expand(self, *consts_and_args):
    if self.jaxpr is None:  # infer_types
      args = consts_and_args
      jaxpr, consts, flat_args, _, out_tree = _trace_fusible(
          self.func, self.output_fusion_prefix, args, "expand_fusible"
      )
      if jaxpr.effects:
        raise ValueError(
            "Effects are not supported in fusible with infer_types. "
            f"Tracing revealed effects: {jaxpr.effects}."
        )
    else:
      consts, args = util.split_list(consts_and_args, [self.num_consts])
      out_tree = self.out_tree
      jaxpr = self.jaxpr
      flat_args = tree_util.tree_leaves(args)
    if jaxpr.is_high:
      arg_avals = [jax_core.typeof(a) for a in flat_args]
      lo_args = [
          lo_val
          for aval, x in zip(arg_avals, flat_args)
          for lo_val in (
              aval.read_loval(x) if aval.has_qdd else aval.lower_val(x)
          )
      ]
      lo_jaxpr = pe.lower_jaxpr2(jax_core.ClosedJaxpr(jaxpr, consts))
    else:
      lo_args = flat_args
      lo_jaxpr = jax_core.ClosedJaxpr(jaxpr, consts)
    lo_outs = eval_jaxpr_p.bind(*lo_args, jaxpr=lo_jaxpr)
    out_flat = pe.raise_lo_outs(jaxpr.out_avals, lo_outs)
    return tree_util.tree_unflatten(out_tree, out_flat)

  def vjp_fwd(self, in_nzs, *args):
    out, vjp_fun = jax.vjp(self.expand, *args)
    return out, vjp_fun

  def vjp_bwd_retval(self, vjp_fun, outgrad):
    return vjp_fun(outgrad)

  def jvp(self, primals, tangents):
    return jax.jvp(self.expand, primals, tangents)

  def batch(self, axis_data, args, dims):
    if axis_data.size != 1:
      raise NotImplementedError("Fusible does not support non-trivial batching")

    def unbatch_leaf(a, d):
      if d is None:
        return a
      return a[d]

    unbatched_args = tree_util.tree_map(unbatch_leaf, args, dims)
    out_unbatched = self(*unbatched_args)  # pyrefly: ignore[not-callable]

    def batch_leaf(o):
      return o[None]

    out = tree_util.tree_map(batch_leaf, out_unbatched)
    out_dims = tree_util.tree_map(lambda _: 0, self.out_aval)

    return out, out_dims

  def physicalize(self, _, *args):
    if self.jaxpr is None:  # infer_types
      jaxpr, consts, flat_args, in_tree, out_tree = _trace_fusible(
          self.func, self.output_fusion_prefix, args, "physicalize_fusible"
      )
      new_jaxpr = fusible_dtype.physicalize_closed_jaxpr(
          jax_core.ClosedJaxpr(jaxpr, list(consts))
      )
      out = _call_fusible(
          new_jaxpr.jaxpr,
          new_jaxpr.consts,
          flat_args,
          self.output_fusion_prefix,
          self.func,
          in_tree,
          out_tree,
      )
      return jax.tree.leaves(out)

    consts = args[: self.num_consts]
    new_jaxpr = fusible_dtype.physicalize_closed_jaxpr(
        jax_core.ClosedJaxpr(self.jaxpr, list(consts))
    )
    _, out_tree = tree_util.tree_flatten(self.out_aval)
    out = _call_fusible(
        new_jaxpr.jaxpr,
        new_jaxpr.consts,
        args[self.num_consts :],
        self.output_fusion_prefix,
        self.func,
        self.args_tree,
        out_tree,
    )
    return jax.tree.leaves(out)


def _make_trivial_fusion(x: jax.Array) -> fusion_lib.Fusion:
  return fusion_lib.Fusion(
      func=lambda: x,
      in_type=((), {}),
      out_type=jax.typeof(x),
  )


def _trace_fusible(func, output_fusion_prefix, args, debug_name):
  def wrapped(*args):
    in_fusions = tree_util.tree_map(_make_trivial_fusion, args)
    output_fusions = tree_util.tree_unflatten(
        tree_util.tree_structure(output_fusion_prefix),
        [None] * len(tree_util.tree_leaves(output_fusion_prefix)),
    )
    return func(*in_fusions, output_fusions)

  args_ft = ft.flatten(args)
  debug_info = api_util.debug_info(debug_name, wrapped, args, {})
  args_avals = tree_util.tree_map(jax_core.typeof, args)
  in_avals_ft = ft.flatten_args(*args_avals)
  jaxpr, out_avals_ft = pe.trace_to_jaxpr(wrapped, in_avals_ft, debug_info)
  return jaxpr, jaxpr.consts, args_ft.vals, args_ft.tree, out_avals_ft.tree


def _call_fusible(
    jaxpr, consts, flat_args, output_fusion_prefix, func, args_tree, out_tree
):
  out_avals_flat = [v.aval for v in jaxpr.outvars]
  out_aval = tree_util.tree_unflatten(out_tree, out_avals_flat)

  const_avals = [jax_core.typeof(x) for x in consts]
  arg_avals = [jax_core.typeof(x) for x in flat_args]
  in_avals = tuple(const_avals) + tuple(arg_avals)

  prim = Fusible(
      jaxpr=jaxpr,
      in_avals=in_avals,
      out_aval=out_aval,
      output_fusion_prefix=output_fusion_prefix,
      func=func,
      num_consts=len(consts),
      args_tree=args_tree,
  )
  return prim(*consts, *flat_args)


@partial(api_boundary, repro_api_name="fuser.fusible")
def fusible(
    f=None,
    *,
    output_fusion_prefix: Any = True,
    infer_types: Callable[..., Any] | None = None,
):
  def decorator(f):

    def fusible_wrapper(*args):
      if infer_types is not None:
        arg_avals = [tree_util.tree_map(jax_core.typeof, x) for x in args]
        out_aval = infer_types(*arg_avals)
        args_ft = ft.flatten(args)
        fusible_prim = Fusible(
            jaxpr=None,
            in_avals=tuple(arg_avals),
            out_aval=out_aval,
            output_fusion_prefix=output_fusion_prefix,
            func=f,
            num_consts=0,
            args_tree=args_ft.tree,
        )
        return fusible_prim(*args)

      jaxpr, consts, flat_args, in_tree, out_tree = _trace_fusible(
          f, output_fusion_prefix, args, "fusible"
      )
      return _call_fusible(
          jaxpr,
          consts,
          flat_args,
          output_fusion_prefix,
          f,
          in_tree,
          out_tree,
      )

    return fusible_wrapper

  if f is not None:
    return decorator(f)
  return decorator
