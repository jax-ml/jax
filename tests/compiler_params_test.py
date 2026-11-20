# Copyright 2026 The JAX Authors.
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

"""Tests for the compiler params contract of lax.fusion and lax.pipeline."""

import dataclasses

from absl.testing import absltest
import jax
from jax._src import sharding_impls
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.lax import compiler_params as cp

jax.config.parse_flags_with_absl()


class _CallParams:
  """Nominal marker base class for one primitive's compiler params."""


@dataclasses.dataclass(frozen=True)
class _TpuParams(_CallParams):
  vmem: int = 0


@dataclasses.dataclass(frozen=True)
class _OtherParams:
  """Params of another primitive: does not subclass `_CallParams`."""


@dataclasses.dataclass
class _MutableParams:
  pass


@dataclasses.dataclass
class _MutableCallParams(_CallParams):
  vmem: int = 0


class _PlainCallParams(_CallParams):
  pass


class CompilerParamsTest(jtu.JaxTestCase):

  def test_none_is_accepted(self):
    self.assertIsNone(cp.check_compiler_params(None, "lax.fusion"))
    self.assertIsNone(cp.check_compiler_params(None, "lax.fusion", _CallParams))

  def test_protocol_object_is_accepted(self):
    p = _TpuParams(vmem=4)
    self.assertIs(cp.check_compiler_params(p, "lax.fusion"), p)
    self.assertIs(cp.check_compiler_params(p, "lax.fusion", _CallParams), p)
    other = _OtherParams()
    self.assertIs(cp.check_compiler_params(other, "lax.fusion"), other)
    hash(p)

  def test_non_protocol_object_is_rejected(self):
    with self.assertRaisesRegex(TypeError, "lax.fusion.*compiler_params"):
      cp.check_compiler_params({"vmem": 4}, "lax.fusion")

  def test_object_of_another_primitive_is_rejected(self):
    with self.assertRaisesRegex(
        TypeError, "lax.fusion.*_CallParams.*_OtherParams"
    ):
      cp.check_compiler_params(_OtherParams(), "lax.fusion", _CallParams)

  def test_mutable_dataclass_is_rejected(self):
    with self.assertRaisesRegex(TypeError, "frozen dataclass"):
      cp.check_compiler_params(_MutableParams(), "lax.pipeline")
    with self.assertRaisesRegex(TypeError, "frozen dataclass"):
      cp.check_compiler_params(_MutableCallParams(), "lax.fusion", _CallParams)
    with self.assertRaisesRegex(TypeError, "frozen dataclass"):
      cp.check_compiler_params(_PlainCallParams(), "lax.fusion", _CallParams)

  def test_module_state_survives_isolation(self):
    module_context = mlir.ModuleContext(
        backend=None,
        platforms=["cpu"],
        axis_context=sharding_impls.ShardingContext(1),
        keepalives=[],
        channel_iterator=iter([]),
        host_callbacks=[],
        lowering_parameters=mlir.LoweringParameters(),
    )
    module_context.lowering_cache["k"] = "v"
    state = cp.module_state(module_context)
    state["counter"] = 1
    iso = cp.isolated_lowering_context(module_context)
    self.assertEqual(iso.lowering_cache, {})
    self.assertIsNot(
        iso.cached_primitive_lowerings,
        module_context.cached_primitive_lowerings,
    )
    self.assertIs(cp.module_state(iso), state)
    self.assertIs(iso.module, module_context.module)
    self.assertEqual(module_context.lowering_cache, {"k": "v"})


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
