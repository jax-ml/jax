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

from jax import lax
from jax._src import core
from jax._src import tree_util
from jax._src.pallas.fuser import fuser_utils


@dataclasses.dataclass
class CustomEvaluateSettings:
  allow_transpose: bool = True


def evaluate(f, *, allow_transpose: bool = True):
  """Re-binds `f`'s traced jaxpr equation-by-equation under fuser constraints.

  Used inside fused kernels to evaluate `f` primitive-by-primitive so that
  fuser evaluation rules apply while rejecting unsupported higher-order
  primitives (`scan`, `while_loop`, `cond`) and optionally `transpose`.

  Args:
    f: The function to trace and evaluate.
    allow_transpose: Whether `lax.transpose_p` primitives are allowed in `f`.

  Returns:
    A wrapped function taking `*args` and `**kwargs` that traces `f` and
    evaluates its jaxpr equation by equation, raising `NotImplementedError` for
    `scan`, `while_loop`, and `cond` primitives and, if `allow_transpose` is
    `False`, `ValueError` for transposes.
  """

  def wrapped(*args, **kwargs):
    jaxpr, consts, _, out_tree = fuser_utils.make_jaxpr(f, *args, **kwargs)
    settings = CustomEvaluateSettings(allow_transpose=allow_transpose)
    flat_args = tree_util.tree_leaves((args, kwargs))
    out_flat = _custom_evaluate_jaxpr(settings, jaxpr, consts, *flat_args)
    return tree_util.tree_unflatten(out_tree, out_flat)

  return wrapped


# Disallow most higher-order primitives for now.
disallowed_primitives = {lax.scan_p, lax.while_p, lax.cond_p}


def _custom_evaluate_jaxpr(
    settings: CustomEvaluateSettings, jaxpr: core.Jaxpr, consts, *args
):
  all_prims = fuser_utils.jaxpr_all_prims(jaxpr)
  if disallowed := all_prims & disallowed_primitives:
    raise NotImplementedError(f'Primitives {disallowed} not supported.')
  if not settings.allow_transpose and lax.transpose_p in all_prims:
    raise ValueError('Transpose not allowed.')
  return core.eval_jaxpr(jaxpr, consts, *args)
