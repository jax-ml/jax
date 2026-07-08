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

import sys

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import mesh as mesh_lib
from jax._src import test_util as jtu
from jax._src.pallas.triton import gpu_info

if sys.platform != "win32":
  # pylint: disable=g-import-not-at-top
  from jax.experimental.pallas import triton as plgpu
  GpuVersion = plgpu.GpuVersion
else:
  plgpu = None
  GpuVersion = gpu_info.GpuVersion


class GpuInfoTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not plgpu:
      self.skipTest("Triton support is not available.")
    if jtu.is_device_cuda() or jtu.is_device_rocm():
      self.assertTrue(plgpu.is_gpu_device())
    else:
      self.assertFalse(plgpu.is_gpu_device())
      self.skipTest("Skipping test because device is not a GPU.")

  def test_get_gpu_info_real_device(self):
    # on real device
    device = jax.devices()[0]
    info = plgpu.get_gpu_info()
    self.assertIsInstance(info, plgpu.GpuInfo)

    expected_arch_name = str(device.compute_capability)
    if "cuda" in device.client.platform_version:
      expected_compute_capability = int(expected_arch_name.replace(".", ""))
    else:
      expected_compute_capability = 0
    self.assertEqual(info.arch_name, expected_arch_name)
    self.assertEqual(info.compute_capability, expected_compute_capability)

  @parameterized.parameters([
    "10.0",
    "gfx1200",
  ])
  def test_get_gpu_info_abs_device(self, device_kind):
    expected_info = gpu_info.get_gpu_info_from_version(
        gpu_info.gpu_version_from_device_kind(device_kind)
    )
    abstract_device = mesh_lib.AbstractDevice(device_kind, 1, platform='gpu')
    abstract_mesh = mesh_lib.AbstractMesh(
        (1,),
        ('x',),
        (mesh_lib.AxisType.Explicit,),
        abstract_device=abstract_device,
    )
    with mesh_lib.use_abstract_mesh(abstract_mesh):
      info = plgpu.get_gpu_info()
      self.assertEqual(info.arch_name, expected_info.arch_name)
      self.assertEqual(info.compute_capability, expected_info.compute_capability)

  def test_get_gpu_info_given_gpu_version(self):
    info = plgpu.get_gpu_info()
    if info.gpu_version is None:
      self.skipTest(
          f"Skipping test as GPU device: {gpu_info.get_device_kind()} "
          "is not in GpuVersion."
      )
    info_version = plgpu.get_gpu_info_from_version(info.gpu_version)
    self.assertEqual(info, info_version)

  @parameterized.parameters([
    ("10.0", GpuVersion.NV_10_0),
    ("12.1", GpuVersion.NV_12_1),
    ("UNKNOWN", None),
    ("gfx908", GpuVersion.GFX908),
    ("gfx90a", GpuVersion.GFX90A),
    ("gfx942", GpuVersion.GFX942),
    ("gfx950", GpuVersion.GFX950),
    ("gfx1030", GpuVersion.GFX1030),
    ("gfx1100", GpuVersion.GFX1100),
    ("gfx1101", GpuVersion.GFX1101),
    ("gfx1103", GpuVersion.GFX1103),
    ("gfx1150", GpuVersion.GFX1150),
    ("gfx1151", GpuVersion.GFX1151),
    ("gfx1200", GpuVersion.GFX1200),
    ("gfx1201", GpuVersion.GFX1201),
  ])
  def test_gpu_version_from_device_kind(self, device_kind, expected):
    info = gpu_info.gpu_version_from_device_kind(device_kind)
    self.assertEqual(info, expected)


if __name__ == "__main__":
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
