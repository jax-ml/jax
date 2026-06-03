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

"""Tests for jax.extend.xla compiler pass registration."""

from __future__ import annotations

import unittest

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax._src.lib import hlo as _hlo
from jax._src.lib import jaxlib_extension_version
import jax.extend.xla as jex_xla
import jax.numpy as jnp

jax.config.parse_flags_with_absl()


class XlaTransformTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform != "cpu":
      self.skipTest("Skipping test for non-CPU devices")

  @parameterized.parameters(
      jex_xla.PipelineStage.PRE_SCHEDULER,
      jex_xla.PipelineStage.POST_SCHEDULER,
  )
  @unittest.skipIf(
      jaxlib_extension_version < 461, "Requires jaxlib_extension_version >= 461"
  )
  def test_sin_to_cos(self, stage):
    """Register a pass that replaces sin ops with cos ops."""

    def sin_to_cos(serialized_hlo: bytes) -> bytes | None:
      """Replace all sin ops with cos ops in an HLO module."""
      module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
      changed = False

      for comp in module.computations():
        for inst in comp.instructions():
          if inst.opcode == _hlo.HloOpcode.kSin:
            operand = inst.operands()[0]
            new_inst = comp.create_unary_instruction(
                _hlo.HloOpcode.kCos, operand
            )
            comp.replace_instruction(inst, new_inst)
            changed = True

      if not changed:
        return None

      return module.as_serialized_hlo_module_proto()

    jex_xla.register_hlo_module_transformation(
        sin_to_cos,
        name=f"sin_to_cos_{stage.name}_test",
        stage=stage,
    )

    @jax.jit
    def f(x):
      return jnp.sin(x)

    x = jnp.array([0.0, 1.0, 2.0])
    result = f(x)
    expected = jnp.cos(x)
    self.assertAllClose(result, expected, atol=1e-5)

  @unittest.skipIf(
      jaxlib_extension_version < 461, "Requires jaxlib_extension_version >= 461"
  )
  def test_sin_to_cos_platform_filtering(self):
    """Register a pass for tpu only, compiling on cpu should not apply it."""

    def sin_to_cos(serialized_hlo: bytes) -> bytes | None:
      module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
      changed = False

      for comp in module.computations():
        for inst in comp.instructions():
          if inst.opcode == _hlo.HloOpcode.kSin:
            operand = inst.operands()[0]
            new_inst = comp.create_unary_instruction(
                _hlo.HloOpcode.kCos, operand
            )
            comp.replace_instruction(inst, new_inst)
            changed = True

      if not changed:
        return None

      return module.as_serialized_hlo_module_proto()

    # Registering for "tpu" only. Since our test runs on "cpu", this pass
    # should NOT be executed.
    jex_xla.register_hlo_module_transformation(
        sin_to_cos,
        name="sin_to_cos_tpu_only_test",
        stage=jex_xla.PipelineStage.PRE_SCHEDULER,
        platforms="tpu",
    )

    @jax.jit
    def f(x):
      return jnp.sin(x)

    x = jnp.array([0.0, 1.0, 2.0])
    result = f(x)
    # Since it's compiled on CPU, it should still return sin(x), not cos(x).
    expected = jnp.sin(x)
    self.assertAllClose(result, expected, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
