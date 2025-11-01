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
"""JAX AOT API."""

from typing import Any, Callable

from absl import logging
from jax._src import aot_util
from jax._src import api
from jax._src import core
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import mlir


component_p = core.Primitive("component")


def _component(f: Callable[..., Any], component_key: Any):
  @util.wraps(f)
  @traceback_util.api_boundary
  def wrapper(*args):
    # TODO(dsuo): Flatten function as in shard_map pmap.
    return component_p.bind(*args, f=f, component_key=component_key)

  return wrapper


def component(
    f: Callable[..., Any] | None = None, component_key: Any = None
) -> Callable[..., Any]:
  if f is None:
    return lambda g: _component(g, component_key=component_key)
  return _component(f, component_key=component_key)


def component_impl(*args, f: Callable[..., Any], **_):
  return f(*args)


def component_abstract_eval(*args, f: Callable[..., Any], component_key: Any):
  logging.info("component_abstract_eval: %s", component_key)

  def abstract_eval():
    return tree_util.tree_map(
        lambda x: core.ShapedArray(x.shape, x.dtype), api.eval_shape(f, *args)
    )

  return aot_util.get_cached_or_put(
      component_key + '.abstract_eval',
      abstract_eval,
      aot_util.serialize_abstract_eval,
      aot_util.deserialize_abstract_eval,
  )


def component_lowering(ctx, *args, f: Callable[..., Any], component_key: Any):
  logging.info("component_lowering: %s", component_key)

  def lowering_rule():
    multiple_results = len(ctx.avals_out) > 1
    return mlir.lower_fun(component_impl, multiple_results=multiple_results)(
        ctx, *args, f=f, component_key=component_key
    )

  return aot_util.get_cached_or_put(
      component_key + '.lowering',
      lowering_rule,
      aot_util.serialize_lowering,
      aot_util.deserialize_lowering,
  )


component_p.def_impl(component_impl)
component_p.def_abstract_eval(component_abstract_eval)
mlir.register_lowering(component_p, component_lowering)
