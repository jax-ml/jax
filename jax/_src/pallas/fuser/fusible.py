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
from typing import Any

import jax
from jax._src import api_util
from jax._src import core as jax_core
from jax._src.interpreters import batching
from jax._src import linear_util as lu
from jax._src.traceback_util import api_boundary
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas.fuser import fusion as fusion_lib

fusible_p = jax_core.Primitive('fusible')
fusible_p.multiple_results = True

def _fusible_is_high(*_, jaxpr, **params):
  del params
  return jaxpr.is_high

fusible_p.is_high = _fusible_is_high # type: ignore


def _make_trivial_fusion(x: jax.Array) -> fusion_lib.Fusion:
  return fusion_lib.Fusion(
      func=lambda: x,
      in_type=((), {}),
      out_type=jax.typeof(x),
  )


@partial(api_boundary, repro_api_name="fuser.fusible")
def fusible(f=None, *, output_fusion_prefix: Any = True):
  def decorator(f):
    def wrapper(*args):
      def wrapped(*args):
        in_fusions = tree_util.tree_map(_make_trivial_fusion, args)
        output_fusions = tree_util.tree_unflatten(
            tree_util.tree_structure(output_fusion_prefix),
            [None] * len(tree_util.tree_leaves(output_fusion_prefix)),
        )
        return f(*in_fusions, output_fusions)

      flat_args, in_tree = tree_util.tree_flatten(args)
      debug_info = api_util.debug_info('fusible', wrapped, args, {})
      flat_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
          lu.wrap_init(wrapped, debug_info=debug_info), in_tree
      )
      flat_avals = [jax_core.get_aval(x) for x in flat_args]
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, flat_avals)
      out_tree = out_tree_thunk()
      out = fusible_p.bind(
          *consts,
          *flat_args,
          jaxpr=jaxpr,
          num_consts=len(consts),
          in_tree=in_tree,
          out_tree=out_tree,
          func=f,
          output_fusion_prefix=output_fusion_prefix,
      )
      return tree_util.tree_unflatten(out_tree, out)

    return wrapper

  if f is not None:
    return decorator(f)
  return decorator


@fusible_p.def_impl
def _(*consts_and_args, jaxpr, num_consts, **_):
  consts, args = util.split_list(consts_and_args, [num_consts])
  return jax_core.eval_jaxpr(jaxpr, consts, *args)


mlir.register_lowering(fusible_p, mlir.lower_fun(fusible_p.impl))


@fusible_p.def_effectful_abstract_eval
def _(*args, jaxpr, **kwargs):
  del args, kwargs
  return [v.aval for v in jaxpr.outvars], jaxpr.effects


def _fusible_trivial_batching_rule(axis_data, args, dims, **kwargs):
  if axis_data.size != 1:
    raise NotImplementedError('fusible does not support non-trivial batching')

  unbatched_args = tuple(
      a if (d is batching.not_mapped or d is None) else a[d]
      for a, d in zip(args, dims, strict=True)
  )
  out_unbatched = fusible_p.bind(*unbatched_args, **kwargs)
  out = tuple(o[None] for o in out_unbatched)

  return out, (0,) * len(out)

batching.fancy_primitive_batchers[fusible_p] = _fusible_trivial_batching_rule


def _fusible_to_lojax(*hi_args, jaxpr, num_consts, **_):
  const_in_avals = jaxpr.in_aval_qdds[:num_consts]
  num_lo_consts = sum(len(aval.lo_ty()) for aval in const_in_avals)

  lo_args = [
      lo_val
      for aval, x in util.safe_zip(jaxpr.in_aval_qdds, hi_args)
      for lo_val in (aval.read_loval(x) if aval.has_qdd else aval.lower_val(x))
  ]

  closed_jaxpr = jax_core.ClosedJaxpr(jaxpr, lo_args[:num_lo_consts])

  lo_jaxpr = pe.lower_jaxpr(closed_jaxpr)
  all_outs = fusible_p.bind(*lo_args, jaxpr=lo_jaxpr.jaxpr, num_consts=num_lo_consts)

  out_mut, lo_outs = util.split_list(all_outs, [pe.num_himuts_out(jaxpr)])
  pe.apply_himut(jaxpr, hi_args, out_mut)
  return pe.raise_lo_outs(jaxpr.out_avals, lo_outs)


fusible_p.to_lojax = _fusible_to_lojax
