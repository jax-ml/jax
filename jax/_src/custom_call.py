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

from __future__ import annotations

import abc
import functools
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from jax import core
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import partial_eval
from jax.interpreters import xla
from jax.interpreters.mlir import ir
from jaxlib import xla_client

jax_to_xla_platform = {"cuda": "CUDA"}


def register_custom_call_targets(
    targets: Mapping[str, Any],
    platform: str | None = None,
    api_version: int = 1,
) -> None:
  for name, value in targets.items():
    xla_client.register_custom_call_target(
        name,
        value,
        platform=jax_to_xla_platform.get(platform, platform),
        api_version=api_version,
    )


class CustomCallPrimitive(abc.ABC):
  """A proposal for a custom call primitive interface."""

  primitive: core.Primitive
  platforms: set[str]
  vectorized: bool = False

  def __init__(self, *, platforms: Sequence[str], vectorized: bool = False):
    self.vectorized = vectorized
    name = self.__class__.__name__.lower()
    self.primitive = prim = core.Primitive(name)
    prim.multiple_results = True
    prim.def_impl(functools.partial(xla.apply_primitive, prim))
    prim.def_abstract_eval(self.abstract_eval)
    batching.primitive_batchers[prim] = self.batching
    self.platforms = set(platforms)
    for platform in platforms:
      mlir.register_lowering(
          prim,
          functools.partial(_custom_call_primitive_lowering_impl, self),
          platform=platform,
      )
    partial_eval.dce_rules[self.primitive] = self.dce

  def __call__(self, *operands, **params):
    return self.primitive.bind(*operands, **params)

  @abc.abstractmethod
  def abstract_eval(self, *operands, **params):
    ...

  def lowering(
      self, ctx: mlir.LoweringRuleContext, *operands: ir.Value, **params: Any
  ):
    return custom_call_lowering(self.primitive.name)(ctx, *operands, **params)

  def batching(
      self,
      batched_args: Sequence[Any],
      batch_dims: Sequence[int],
      **params: Any,
  ):
    if self.vectorized:
      # TODO(dfm): This is almost the same as batching.vectorized_batcher, but
      # that doesn't support multiple results
      assert all(batch_dims[0] == bd for bd in batch_dims[1:]), batch_dims
      result = self.primitive.bind(*batched_args, **params)
      return result, [batch_dims[0] for _ in result]
    else:
      raise NotImplementedError(
          f"Batching rule not implemented for {self.primitive.name}"
      )

  def jvp(self, primals, tangents, **params):
    raise NotImplementedError(
        f"JVP rule not implemented for {self.primitive.name}"
    )

  def transpose(self, *args, **params):
    raise NotImplementedError(
        f"Transpose rule not implemented for {self.primitive.name}"
    )

  def partition(self, *args, **params):
    raise NotImplementedError(
        f"Partition rule not implemented for {self.primitive.name}"
    )

  def dce(
      self, used_outputs: list[bool], eqn: core.JaxprEqn
  ) -> tuple[list[bool], core.JaxprEqn | None]:
    del used_outputs
    return [True for _ in eqn.invars], eqn


def _custom_call_primitive_lowering_impl(
    prim: CustomCallPrimitive,
    ctx: mlir.LoweringRuleContext,
    *operands: ir.Value,
    **params: Any,
) -> Sequence[ir.Value]:
  try:
    [platform] = ctx.module_context.platforms
  except ValueError:
    raise ValueError(
        f"Can only lower a custom call {prim.primitive.name} on a single"
        " platform."
    ) from None

  if platform not in prim.platforms:
    raise ValueError(
        f"Lowering of {prim.primitive.name} is only supported on"
        f" {prim.platforms}, not {platform}."
    )

  return prim.lowering(ctx, *operands, **params)


def _default_layouts(shapes):
  return [list(reversed(range(len(shape)))) for shape in shapes]


def _ir_attribute(obj: Any) -> ir.Attribute:
  if isinstance(obj, str):
    return ir.StringAttr.get(obj)
  if isinstance(obj, bool):
    return ir.BoolAttr.get(obj)
  if isinstance(obj, int):
    # TODO(slebedev): Consider supporting NumPy scalars.
    return mlir.i64_attr(obj)
  elif isinstance(obj, float):
    return ir.FloatAttr.get(ir.F32Type.get(), obj)
  else:
    raise TypeError(f"Unsupported attribute type: {type(obj)}")


class LoweringRule(Protocol):

  def __call__(
      self, ctx: mlir.LoweringRuleContext, *operands: ir.Value, **params: Any
  ) -> Sequence[ir.Value]:
    ...


def custom_call_lowering(name: str, **lowering_kwargs: Any) -> LoweringRule:
  """Returns a custom call lowering rule."""

  def _lowering(
      ctx: mlir.LoweringRuleContext, *operands: ir.Value, **params: Any
  ) -> Sequence[ir.Value]:
    kwargs = dict(lowering_kwargs)
    kwargs.setdefault(
        "result_types", [mlir.aval_to_ir_type(aval) for aval in ctx.avals_out]
    )
    kwargs.setdefault(
        "backend_config", {k: _ir_attribute(v) for k, v in params.items()}
    )
    kwargs.setdefault("api_version", 4)
    kwargs.setdefault(
        "operand_layouts", _default_layouts(aval.shape for aval in ctx.avals_in)  # pytype: disable=attribute-error
    )
    kwargs.setdefault(
        "result_layouts", _default_layouts(aval.shape for aval in ctx.avals_out)
    )
    return mlir.custom_call(name, operands=operands, **kwargs).results

  return _lowering
