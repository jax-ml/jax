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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import core
from jax._src import lax_reference
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np


config.parse_flags_with_absl()


class RaggedDotGpuPallasTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(["gpu"]):
      self.skipTest("This test requires a GPU")
    else:
      if not jtu.is_cuda_compute_capability_at_least("8.0"):
        self.skipTest("This test requires a GPU")
    super().setUp()

  def _test_ragged_dot(self, m, k, n, num_groups, dtype):
    if dtype == np.float16:
      self.skipTest(f"unsupported dtype for ragged_dot: {dtype}")
    lhs_shape = (m, k)
    rhs_shape = (num_groups, k, n)

    def group_sizes(m, num_groups):
      ends_no_final = jnp.sort(self.rng().choice(m, size=num_groups - 1))
      ends = jnp.concatenate(
          [ends_no_final, jnp.array([m], dtype=ends_no_final.dtype)])
      starts = jnp.concatenate(
          [jnp.zeros(1, dtype=ends_no_final.dtype), ends_no_final])
      return ends - starts

    rng = jtu.rand_small(self.rng())
    args_maker = lambda: [
        rng(lhs_shape, dtype),
        rng(rhs_shape, dtype),
        group_sizes(m, num_groups),
    ]
    self._CompileAndCheck(jax.lax.ragged_dot, args_maker)
    self._CheckAgainstNumpy(lax_reference.ragged_dot, jax.lax.ragged_dot,
                            args_maker)

  @parameterized.parameters([True, False])
  def test_ragged_dot_use_gpu_pallas_triton_lowering(self, use_instruction):
    with config.jax_ragged_dot_use_gpu_pallas_triton_lowering(use_instruction):
      self._test_ragged_dot(16, 4, 3, 2, jnp.float32)
      stablehlo_text = (
          jax.jit(jax.lax.ragged_dot)
          .lower(
              core.ShapedArray((16, 4), dtype=jnp.float32),
              core.ShapedArray((2, 4, 3), dtype=jnp.float32),
              core.ShapedArray((2,), dtype=jnp.int32),
          )
          .as_text(dialect="stablehlo")
      )
      if use_instruction:
        self.assertIn("pallas_triton_ragged_dot", stablehlo_text)
      else:
        self.assertNotIn("pallas_triton_ragged_dot", stablehlo_text)

  @parameterized.named_parameters(
      dict(testcase_name="dlhs", grad_wrt="dlhs"),
      dict(testcase_name="drhs", grad_wrt="drhs"),
  )
  def test_no_transpose_in_grad(self, grad_wrt):
    m, k, n, g = 64, 32, 16, 2
    lhs = jnp.ones((m, k), dtype=jnp.float32)
    rhs = jnp.ones((g, k, n), dtype=jnp.float32)
    group_sizes = jnp.array([m // g] * g, dtype=jnp.int32)

    if grad_wrt == "dlhs":
      fn, arg = lambda lhs: jax.lax.ragged_dot(lhs, rhs, group_sizes).sum(), lhs
    elif grad_wrt == "drhs":
      fn, arg = lambda rhs: jax.lax.ragged_dot(lhs, rhs, group_sizes).sum(), rhs
    else:
      raise ValueError(f"Unknown grad_wrt: {grad_wrt}")

    with config.jax_ragged_dot_use_gpu_pallas_triton_lowering(True):
      stablehlo_text = jax.jit(jax.grad(fn)).lower(arg).as_text(dialect="stablehlo")
      hlo_text = jax.jit(jax.grad(fn)).lower(arg).as_text(dialect="hlo")
    msg = (
      f"The pallas lowering of ragged_dot {grad_wrt} in a basic case should "
      "require no transposes, since the kernel itself should be able to "
      f"handle the transposes.\nhlo = \n{hlo_text}\nstablehlo = \n"
      f"{stablehlo_text}"
    )
    self.assertNotIn("stablehlo.transpose", stablehlo_text, msg)
    self.assertNotIn(" transpose(", hlo_text, msg)


if __name__ == "__main__":
  absltest.main()
