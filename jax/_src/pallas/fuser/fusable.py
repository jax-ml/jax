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

"""Fusable primitive."""

import jax
from jax._src import api_util
from jax._src import core as jax_core
from jax._src import linear_util as lu
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas.fuser import fusion as fusion_lib

fusable_p = jax_core.Primitive('fusable')
fusable_p.multiple_results = True


def _get_aval(x):
  return jax_core.raise_to_shaped(jax_core.get_aval(x))


def _make_trivial_fusion(x: jax.Array) -> fusion_lib.Fusion:
  return fusion_lib.Fusion(
      func=lambda: x,
      in_type=((), {}),
      out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
  )


def fusable(f):
  def wrapper(*args):
    def wrapped(*args):
      in_fusions = tree_util.tree_map(_make_trivial_fusion, args)
      return f(*in_fusions, None)

    flat_args, in_tree = tree_util.tree_flatten(args)
    debug_info = api_util.debug_info('fusable', wrapped, args, {})
    flat_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
        lu.wrap_init(wrapped, debug_info=debug_info), in_tree
    )
    flat_avals = [_get_aval(x) for x in flat_args]
    jaxpr, _, consts, _ = pe.trace_to_jaxpr_dynamic(flat_fun, flat_avals)
    out_tree = out_tree_thunk()
    out = fusable_p.bind(
        *consts,
        *flat_args,
        jaxpr=jaxpr,
        num_consts=len(consts),
        in_tree=in_tree,
        out_tree=out_tree,
        func=f,
    )
    return tree_util.tree_unflatten(out_tree, out)

  return wrapper


@fusable_p.def_impl
def _(*consts_and_args, jaxpr, num_consts, **_):
  consts, args = util.split_list(consts_and_args, [num_consts])
  return jax_core.eval_jaxpr(jaxpr, consts, *args)


mlir.register_lowering(fusable_p, mlir.lower_fun(fusable_p.impl))


@fusable_p.def_abstract_eval
def _(*args, jaxpr, **kwargs):
  del args, kwargs
  return [v.aval for v in jaxpr.outvars]
