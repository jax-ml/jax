# Copyright 2023 The JAX Authors.
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

"""Test exporting Pallas kernels."""

import sys

from absl.testing import absltest
import jax
from jax import export
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
import numpy as np
try:
  from jax._src.lib import triton
except ImportError:
  triton = None  # Windows builds don't have Triton.


jax.config.parse_flags_with_absl()


class ExportTest(jtu.JaxTestCase):

  def setUp(self):
    if sys.platform == "win32":
      self.skipTest("Only works on non-Windows platforms")

    super().setUp()

  def test_cross_platform(self):
    # TODO(apaszke): Remove after 12 weeks have passed.
    if not jtu.if_cloud_tpu_at_least(2024, 12, 19):
      self.skipTest("Requires libtpu built after 2024-12-19")
    def add_vectors_kernel(x_ref, y_ref, o_ref):
      x, y = x_ref[...], y_ref[...]
      o_ref[...] = x + y

    @jax.jit
    def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
      return pl.pallas_call(add_vectors_kernel,
                            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                            name="my_custom_kernel_name",
                            )(x, y)

    platforms = ["tpu"]
    # TODO(b/394629193): Remove True once the bug is fixed.
    if True or triton.has_compilation_handler("cuda"):
      # Only include CUDA if GPU support is linked in.
      platforms.append("cuda")

    a = np.arange(8 * 16, dtype=np.int32).reshape((8, 16))
    exp = export.export(
        add_vectors,
        platforms=platforms,
        # The Pallas GPU custom call is not enabled for export by default.
        disabled_checks=[
            export.DisabledSafetyCheck.custom_call("triton_kernel_call"),
            export.DisabledSafetyCheck.custom_call("__gpu$xla.gpu.triton")
        ]
    )(a, a)

    if (jtu.device_under_test() == "tpu" or
        (jtu.device_under_test() == "gpu" and
         jtu.is_cuda_compute_capability_at_least("8.0"))):
      res = exp.call(a, a)
      self.assertAllClose(res, a + a)

    # Check that we use the proper kernels names
    if "tpu" in platforms:
      self.assertRegex(
          exp.mlir_module(),
          r"stablehlo.custom_call @tpu_custom_call.+kernel_name\s*=\s*\"my_custom_kernel_name\"")
    if "cuda" in platforms:
      self.assertRegex(
          exp.mlir_module(),
          r"stablehlo.custom_call @__gpu\$xla\.gpu\.triton.+name\s*=\s*\"my_custom_kernel_name\"")


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
