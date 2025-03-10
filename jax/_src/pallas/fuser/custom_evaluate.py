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

"""Helpers for evaluating functions under certain constraints."""
import dataclasses
from typing import Any

from jax import lax
from jax._src import core
from jax._src import source_info_util
from jax._src import tree_util
from jax._src import util
from jax._src.pallas.fuser import fuser_utils


@dataclasses.dataclass
class CustomEvaluateSettings:
  allow_transpose: bool = True


def evaluate(f, *, allow_transpose: bool = True):
  def wrapped(*args, **kwargs):
    jaxpr, consts, _, out_tree = fuser_utils.make_jaxpr(f, *args, **kwargs)
    settings = CustomEvaluateSettings(allow_transpose=allow_transpose)
    flat_args = tree_util.tree_leaves(args)
    out_flat = _custom_evaluate_jaxpr(settings, jaxpr, consts, *flat_args)
    return tree_util.tree_unflatten(out_tree, out_flat)

  return wrapped


# Disallow most higher-order primitives for now.
disallowed_primitives = {lax.scan_p, lax.while_p, lax.cond_p}


def _custom_evaluate_jaxpr(
    settings: CustomEvaluateSettings, jaxpr: core.Jaxpr, consts, *args
):
  def read(v: core.Atom) -> Any:
    return v.val if isinstance(v, core.Literal) else env[v]

  def write(v: core.Var, val: Any) -> None:
    env[v] = val

  env: dict[core.Var, Any] = {}
  util.safe_map(write, jaxpr.constvars, consts)
  util.safe_map(write, jaxpr.invars, args)
  lu = core.last_used(jaxpr)
  for eqn in jaxpr.eqns:
    subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)

    if eqn.primitive in disallowed_primitives:
      raise NotImplementedError(f'Primitive {eqn.primitive} not supported.')
    if not settings.allow_transpose and eqn.primitive is lax.transpose_p:
      raise ValueError('Transpose not allowed.')
    name_stack = (
        source_info_util.current_name_stack() + eqn.source_info.name_stack
    )
    traceback = eqn.source_info.traceback
    with source_info_util.user_context(
        traceback, name_stack=name_stack
    ), eqn.ctx.manager:
      ans = eqn.primitive.bind(
          *subfuns, *util.safe_map(read, eqn.invars), **bind_params
      )
    if eqn.primitive.multiple_results:
      util.safe_map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
    core.clean_up_dead_vars(eqn, env, lu)
  return util.safe_map(read, jaxpr.outvars)
