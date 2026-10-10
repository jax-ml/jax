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

"""Fusion classes."""

from __future__ import annotations

import dataclasses
from typing import Any
from collections.abc import Callable

import jax
from jax._src import tree_util
from jax._src import util

safe_map = util.safe_map


@dataclasses.dataclass
class Fusion[**A, K]:

  func: Callable[A, K]
  in_type: tuple[tuple[Any, ...], dict[str, Any]]
  out_type: Any
  strict_mode: bool = True

  def __call__(self, *args: A.args, **kwargs: A.kwargs) -> K:
    return self.func(*args, **kwargs)

  @property
  def shape(self):
    return jax.tree.map(lambda x: x.shape, self.out_type)

  @property
  def dtype(self):
    return jax.tree.map(lambda x: x.dtype, self.out_type)

  @property
  def type(self):
    return self.out_type

  @property
  def in_shape(self):
    return jax.tree.map(lambda x: x.shape, self.in_type)

  @property
  def in_dtype(self):
    return jax.tree.map(lambda x: x.dtype, self.in_type)


# Under tracing (jit, eval_shape, or as `jax.experimental.rebindable`
# operands) a Fusion flattens to the values its `func` closes over, which
# requires `func` to be a `tree_util.Partial`; any other `func` keeps its
# captured values hidden, so capturing a tracer that way fails when the tracing
# boundary is crossed. User code calling `jax.tree.map` on fusions still sees
# them as leaves.
def _flatten_fusion(f: Fusion):
  func = f.func
  if not isinstance(func, tree_util.Partial):
    func = tree_util.Partial(func)  # captured values stay hidden in `func`
  type_leaves, type_tree = tree_util.tree_flatten((f.in_type, f.out_type))
  return (func,), (tuple(type_leaves), type_tree, f.strict_mode)

def _unflatten_fusion(aux, children):
  type_leaves, type_tree, strict_mode = aux
  in_type, out_type = type_tree.unflatten(type_leaves)
  return Fusion(children[0], in_type, out_type, strict_mode)

tree_util.tracing_registry.register_node(
    Fusion, _flatten_fusion, _unflatten_fusion)
