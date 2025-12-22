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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


class SideEffectsTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu():
      self.skipTest("TPU required")

  @parameterized.named_parameters(
      ("pure", pltpu.SideEffectType.PURE),
      ("side_effecting", pltpu.SideEffectType.SIDE_EFFECTING),
      ("dataflow_side_effecting", pltpu.SideEffectType.DATAFLOW_SIDE_EFFECTING),
      ("legacy_true", True),
      ("legacy_false", False),
  )
  def test_side_effects_enum(self, side_effect_type):
    def kernel(x_ref, o_ref):
      pltpu.sync_copy(x_ref, o_ref)

    @jax.jit
    def f(x):
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          compiler_params=pltpu.CompilerParams(
              has_side_effects=side_effect_type
          ),
      )(x)

    x = jnp.ones((8, 128), dtype=jnp.float32)
    y = f(x)
    np.testing.assert_array_equal(y, x)

  def test_invalid_side_effect_type(self):
    def kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    @jax.jit
    def f(x):
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          compiler_params=pltpu.CompilerParams(has_side_effects="invalid"),
      )(x)

    with self.assertRaisesRegex(ValueError, "Invalid side effect type"):
      f(jnp.ones((8, 8), dtype=jnp.float32))

  def test_side_effecting_dce(self):
    def kernel(x_ref, o_ref):
      pltpu.sync_copy(x_ref, o_ref)

    def get_compiled_hlo(side_effect_type):
      @jax.jit
      def f(x):
        # We use dce_sink to consume the output but allow DCE if the op is pure/dce-able.
        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            compiler_params=pltpu.CompilerParams(
                has_side_effects=side_effect_type
            ),
        )(x)
        jax.lax.dce_sink(out)
        return x

      lowered = f.lower(jnp.ones((8, 8), dtype=jnp.float32))
      shlo = lowered.as_text()
      hlo = lowered.compile().as_text()
      return shlo, hlo

    # PURE kernels should be DCE'd.
    shlo_pure, hlo_pure = get_compiled_hlo(pltpu.SideEffectType.PURE)
    self.assertIn("custom_call", shlo_pure)
    self.assertNotIn("custom-call", hlo_pure)

    # SIDE_EFFECTING kernels should NOT be DCE'd.
    shlo_side_effecting, hlo_side_effecting = get_compiled_hlo(
        pltpu.SideEffectType.SIDE_EFFECTING
    )
    self.assertIn("custom_call", shlo_side_effecting)
    self.assertIn("has_side_effect = true", shlo_side_effecting)
    self.assertIn("custom-call", hlo_side_effecting)
    self.assertIn("custom_call_has_side_effect=true", hlo_side_effecting)

    # DATAFLOW_SIDE_EFFECTING kernels SHOULD be DCE'd if outputs are unused.
    shlo_dataflow, hlo_dataflow = get_compiled_hlo(
        pltpu.SideEffectType.DATAFLOW_SIDE_EFFECTING
    )
    self.assertIn("custom_call", shlo_dataflow)
    self.assertIn("has_side_effect = true", shlo_dataflow)
    self.assertNotIn("custom-call", hlo_dataflow)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
