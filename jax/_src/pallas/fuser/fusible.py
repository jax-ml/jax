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
from jax._src import flattree as ft
from jax._src import hijax
from jax._src import tree_util
from jax._src.interpreters import ad
from jax._src.interpreters import partial_eval as pe
from jax._src.lax.control_flow.loops import eval_jaxpr_p
from jax._src.pallas.fuser import fusible_dtype
from jax._src.pallas.fuser import fusion as fusion_lib
from jax._src.traceback_util import api_boundary


class Fusible(hijax.HiPrim):
  output_fusion_prefix: Any
  func: Callable
  jaxpr: jax_core.Jaxpr
  num_consts: int
  args_tree: tree_util.PyTreeDef

  def __init__(
      self,
      jaxpr,
      in_avals,
      out_aval,
      output_fusion_prefix,
      func,
      num_consts,
      args_tree,
  ):
    assert isinstance(jaxpr, jax_core.Jaxpr)

    self.in_avals = tuple(in_avals)
    self.out_aval = out_aval
    self.params = {
        "jaxpr": jaxpr,
        "output_fusion_prefix": output_fusion_prefix,
        "func": func,
        "num_consts": num_consts,
        "args_tree": args_tree,
    }
    self.effects = frozenset(jax_core.positional_effects(jaxpr))
    super().__init__()

  def inline(self, *consts_and_args):
    args_flat = tree_util.tree_leaves(consts_and_args)
    out_flat = eval_jaxpr_p.bind(*args_flat, call_jaxpr=self.jaxpr)
    return tree_util.tree_unflatten(self.out_tree, out_flat)

  expand = inline

  def vjp_fwd(self, in_nzs, *args):
    out, vjp_fun = jax.vjp(self.inline, *args)
    return out, vjp_fun

  def vjp_bwd(self, vjp_fun, outgrad, *arg_accums):
    vjp_fun.with_refs(*arg_accums)(outgrad)

  def jvp(self, primals, tangents):
    ps_ft = ft.flatten(primals)
    ts_ft = ft.flatten(tangents, is_leaf=lambda x: isinstance(x, ad.Zero))
    out_primals, out_tangents = ad.jvp(
        self.expand, ps_ft, ts_ft, instantiate=False
    )
    return out_primals.unflatten(), out_tangents.unflatten()

  def batch(self, axis_data, args, dims):
    if axis_data.size != 1:
      raise NotImplementedError("Fusible does not support non-trivial batching")

    def unbatch_leaf(a, d):
      if d is None:
        return a
      return jax.lax.squeeze(a, (d,))

    unbatched_args = tree_util.tree_map(unbatch_leaf, args, dims)
    out_unbatched = self(*unbatched_args)  # pyrefly: ignore[not-callable]

    def batch_leaf(o):
      return o[None]

    out = tree_util.tree_map(batch_leaf, out_unbatched)
    out_dims = tree_util.tree_map(lambda _: 0, self.out_aval)

    return out, out_dims

  def physicalize(self, ctx, *args):
    del ctx
    consts = args[: self.num_consts]
    new_jaxpr = fusible_dtype.physicalize_closed_jaxpr(
        self.jaxpr.with_consts(consts)
    )

    const_avals = tuple(map(jax_core.typeof, new_jaxpr.consts))
    flat_avals = tree_util.tree_map(jax_core.typeof, args[self.num_consts :])

    out_avals_flat = [v.aval for v in new_jaxpr.outvars]
    _, out_tree = tree_util.tree_flatten(self.out_aval)
    new_out_aval = tree_util.tree_unflatten(out_tree, out_avals_flat)

    new_jaxpr, consts = pe.separate_consts(new_jaxpr)

    new_prim = Fusible(
        jaxpr=new_jaxpr,
        in_avals=const_avals + flat_avals,
        out_aval=new_out_aval,
        output_fusion_prefix=self.output_fusion_prefix,
        func=self.func,
        num_consts=len(consts),
        args_tree=self.args_tree,
    )
    out = new_prim(*consts, *args[self.num_consts :])
    return jax.tree.leaves(out)


def _make_trivial_fusion(x: jax.Array) -> fusion_lib.Fusion:
  return fusion_lib.Fusion(
      func=lambda: x,
      in_type=((), {}),
      out_type=jax.typeof(x),
  )


@partial(api_boundary, repro_api_name="fuser.fusible")
def fusible(f=None, *, output_fusion_prefix: Any = True):
  """Decorator that marks a kernel wrapper function as a fusible for `fuse`.

  The decorated function `f` is invoked as `f(*in_fusions, output_fusions,
  **kwargs)`, where each positional argument in `in_fusions` preserves the
  PyTree structure of the corresponding input in `*args` with array leaves
  wrapped in `Fusion` objects, and `output_fusions` is a PyTree matching the
  structure of `output_fusion_prefix` whose leaves are `Fusion` objects or
  `None` when no output fusion is present. Keyword arguments are bound to `f`
  as static values. When called outside `fuse`, each input array leaf is
  wrapped in a trivial `Fusion` that returns it and all `output_fusions`
  leaves are `None`.

  Args:
    f: The fusible kernel function to decorate. If `None`, returns a
      decorator.
    output_fusion_prefix: A PyTree prefix mask whose PyTree structure specifies
      how the outputs of `f` are grouped into separate output fusions (with the
      default `True`, all outputs are grouped into a single `Fusion | None`).
      Multiple output fusion groups are not supported when the output fusion
      writes to `Ref`s.

  Returns:
    A wrapped function that binds a `Fusible` primitive when called, or a
    decorator if `f` is `None`.
  """

  def decorator(f):
    def wrapper(*args, **kwargs):
      f_bound = partial(f, **kwargs) if kwargs else f

      def wrapped(*args):
        in_fusions = tree_util.tree_map(_make_trivial_fusion, args)
        output_fusions = tree_util.tree_unflatten(
            tree_util.tree_structure(output_fusion_prefix),
            [None] * len(tree_util.tree_leaves(output_fusion_prefix)),
        )
        return f_bound(*in_fusions, output_fusions)

      args_ft = ft.flatten(args)
      debug_info = api_util.debug_info("fusible", wrapped, args, {})
      args_avals = tree_util.tree_map(jax_core.typeof, args)
      in_avals_ft = ft.flatten_args(*args_avals)
      jaxpr, out_avals_ft = pe.trace_to_jaxpr(
          wrapped, in_avals_ft, debug_info
      )
      jaxpr, consts = pe.separate_consts(jaxpr)
      prim = Fusible(
          jaxpr=jaxpr,
          in_avals=(*map(jax_core.typeof, consts), *in_avals_ft.vals),
          out_aval=tree_util.tree_unflatten(
              out_avals_ft.tree, out_avals_ft.vals),
          output_fusion_prefix=output_fusion_prefix,
          func=f_bound,
          num_consts=len(consts),
          args_tree=args_ft.tree,
      )
      return prim(*consts, *args_ft.vals)

    return wrapper

  if f is not None:
    return decorator(f)
  return decorator
