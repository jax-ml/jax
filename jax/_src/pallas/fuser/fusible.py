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
from jax._src import hijax
from jax._src.interpreters import batching
from jax._src import linear_util as lu
from jax._src.traceback_util import api_boundary
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas.fuser import fusion as fusion_lib
from jax._src.pallas.fuser import fusible_dtype


class Fusible(hijax.VJPHiPrimitive):
  output_fusion_prefix: Any
  func: Callable
  jaxpr: jax_core.Jaxpr
  num_consts: int
  args_tree: tree_util.PyTreeDef

  def __init__(self, jaxpr, in_avals, out_aval, output_fusion_prefix, func, num_consts, args_tree):
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
    self.effects = frozenset(jaxpr.effects) if jaxpr is not None else frozenset()
    super().__init__()

  def expand(self, *consts_and_args):
    if self.jaxpr is None:
      # this is called when fusible is being called with output_types
      # outside of fuse(). we simply create the fusions and call the user function.
      args = consts_and_args
      in_fusions = tree_util.tree_map(_make_trivial_fusion, args)
      output_fusions = tree_util.tree_unflatten(
          tree_util.tree_structure(self.output_fusion_prefix),
          [None] * len(tree_util.tree_leaves(self.output_fusion_prefix)),
      )
      return self.func(*in_fusions, output_fusions)

    consts, args = util.split_list(consts_and_args, [self.num_consts])
    flat_args = tree_util.tree_leaves(args)
    out_flat = jax_core.eval_jaxpr(self.jaxpr, consts, *flat_args)
    return tree_util.tree_unflatten(self.out_tree, out_flat)

  def vjp_fwd(self, in_nzs, *args):
    out, vjp_fun = jax.vjp(self.expand, *args)
    return out, vjp_fun

  def vjp_bwd_retval(self, vjp_fun, outgrad):
    return vjp_fun(outgrad)

  def batch(self, axis_data, args, dims):
    if axis_data.size != 1:
      raise NotImplementedError("Fusible does not support non-trivial batching")

    def unbatch_leaf(a, d):
      if d is batching.not_mapped or d is None:
        return a
      return a[d]

    unbatched_args = tree_util.tree_map(unbatch_leaf, args, dims)
    out_unbatched = self(*unbatched_args)

    def batch_leaf(o):
      return o[None]

    out = tree_util.tree_map(batch_leaf, out_unbatched)
    out_dims = tree_util.tree_map(lambda _: 0, self.out_aval)

    return out, out_dims

  def physicalize(self, _, *args):
    if self.jaxpr is None:
      jaxpr, consts, in_tree, _ = _trace_fusible(
          self.func, self.output_fusion_prefix, args, "physicalize_fusible"
      )
      new_jaxpr = fusible_dtype.physicalize_closed_jaxpr(jax_core.ClosedJaxpr(jaxpr, list(consts)))
      _, out_tree = tree_util.tree_flatten(self.out_aval)
      out = _call_fusible(
          new_jaxpr.jaxpr,
          new_jaxpr.consts,
          args,
          self.output_fusion_prefix,
          self.func,
          in_tree,
          out_tree,
      )
      return jax.tree.leaves(out)

    consts = args[:self.num_consts]
    new_jaxpr = fusible_dtype.physicalize_closed_jaxpr(jax_core.ClosedJaxpr(self.jaxpr, list(consts)))
    _, out_tree = tree_util.tree_flatten(self.out_aval)
    out = _call_fusible(
        new_jaxpr.jaxpr,
        new_jaxpr.consts,
        args[self.num_consts:],
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

  flat_args, in_tree = tree_util.tree_flatten(args)
  debug_info = api_util.debug_info(debug_name, wrapped, args, {})
  flat_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
      lu.wrap_init(wrapped, debug_info=debug_info), in_tree
  )
  flat_avals = [jax_core.typeof(x) for x in flat_args]
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, flat_avals)
  return jaxpr, consts, in_tree, out_tree_thunk


def _call_fusible(
    jaxpr, consts, args, output_fusion_prefix, func, args_tree, out_tree
):
  out_avals_flat = [v.aval for v in jaxpr.outvars]
  out_aval = tree_util.tree_unflatten(out_tree, out_avals_flat)

  const_avals = [jax_core.typeof(x) for x in consts]
  arg_avals = [tree_util.tree_map(jax_core.typeof, x) for x in args]
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
  return prim(*consts, *args)


@partial(api_boundary, repro_api_name="fuser.fusible")
def fusible(f=None, *, output_fusion_prefix: Any = True, output_types: Callable[..., Any] | None = None):
  def decorator(f):
    def wrapper(*args):
      if output_types is not None:
        arg_avals = [tree_util.tree_map(jax_core.typeof, x) for x in args]
        out_aval = output_types(*arg_avals)
        fusible_prim = Fusible(
            jaxpr=None,
            in_avals=tuple(arg_avals), # assuming no consts
            out_aval=out_aval,
            output_fusion_prefix=output_fusion_prefix,
            func=f,
            num_consts=0,
            args_tree=tree_util.tree_structure(args),
        )
        return fusible_prim(*args)

      jaxpr, consts, in_tree, out_tree_thunk = _trace_fusible(
          f, output_fusion_prefix, args, "fusible"
      )
      out_tree = out_tree_thunk()
      return _call_fusible(
          jaxpr,
          consts,
          args,
          output_fusion_prefix,
          f,
          in_tree,
          out_tree,
      )

    return wrapper

  if f is not None:
    return decorator(f)
  return decorator
