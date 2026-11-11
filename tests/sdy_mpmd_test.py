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
"""Tests for compiling and running MPMD programs with jaxlib's `_sdy_mpmd`."""

from absl.testing import absltest
import jax
from jax._src import core
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.lib import jaxlib_extension_version
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mpmd
import numpy as np

# Before version 523, `_sdy_mpmd` fails to import from jaxlib wheels.
if jaxlib_extension_version >= 523:
  from jaxlib import _sdy_mpmd  # pylint: disable=g-import-not-at-top

jax.config.parse_flags_with_absl()
jtu.request_cpu_devices(2)

# Adds one on mesh "m0", transfers the result to mesh "m1" and doubles it there.
_TWO_MESH_PROGRAM = """
module @two_meshes attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=1]>
  func.func public @main(%arg0: !mpmd.mesh_tensor<"m0", tensor<8xf32>, sharding=<@mesh, [{}]>>)
      -> !mpmd.mesh_tensor<"m1", tensor<8xf32>, sharding=<@mesh, [{}]>>
      attributes {topology = #mpmd.topology<<"m0" : <["x"=1]>>, <"m1" : <["x"=1]>>>} {
    %0 = mpmd.fragment_call<mesh="m0", origin=["add_one"]> @add_one(%arg0)
        : (!mpmd.mesh_tensor<"m0", tensor<8xf32>, sharding=<@mesh, [{}]>>)
        -> !mpmd.mesh_tensor<"m0", tensor<8xf32>>
    %1 = mpmd.transfer %0
        : (!mpmd.mesh_tensor<"m0", tensor<8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<8xf32>>
    %2 = mpmd.fragment_call<mesh="m1", origin=["double"]> @double(%1)
        : (!mpmd.mesh_tensor<"m1", tensor<8xf32>>)
        -> !mpmd.mesh_tensor<"m1", tensor<8xf32>, sharding=<@mesh, [{}]>>
    return %2 : !mpmd.mesh_tensor<"m1", tensor<8xf32>, sharding=<@mesh, [{}]>>
  }
  func.func @add_one(%arg0: tensor<8xf32>) -> tensor<8xf32>
      attributes {mesh_shape = #sdy.mesh<["x"=1]>} {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<8xf32>
    %0 = stablehlo.add %arg0, %cst : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func @double(%arg0: tensor<8xf32>) -> tensor<8xf32>
      attributes {mesh_shape = #sdy.mesh<["x"=1]>} {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<8xf32>
    %0 = stablehlo.multiply %arg0, %cst : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
"""


class SdyMpmdTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jaxlib_extension_version < 523:
      self.skipTest("Requires jaxlib_extension_version >= 523")

  def test_compile_and_execute_on_two_meshes(self):
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest("Requires 2 devices.")
    context = mlir.make_ir_context()
    mpmd.register_dialect(context)
    with context:
      module = ir.Module.parse(_TWO_MESH_PROGRAM)
    _sdy_mpmd.lower_to_ifrt(module)

    out_mesh = jax.sharding.Mesh(devices[1:2], ("x",))
    executable = _sdy_mpmd.compile_mpmd(
        backend=devices[0].client,
        ifrt_mlir_module=module,
        devices=devices[:2],
        out_avals=[core.ShapedArray((8,), np.float32)],
        out_shardings=[
            jax.sharding.NamedSharding(out_mesh, jax.sharding.PartitionSpec())
        ],
    )
    x = jax.device_put(np.arange(8, dtype=np.float32), devices[0])
    (y,) = executable.execute([x])

    self.assertEqual(y.sharding.device_set, {devices[1]})
    np.testing.assert_array_equal(
        np.asarray(y), (np.arange(8, dtype=np.float32) + 1) * 2
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
