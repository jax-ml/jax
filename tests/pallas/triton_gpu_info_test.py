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
      self.skipTest("Skipping test because device is not a GPU.")
    if not jtu.is_device_cuda():
      self.assertFalse(plgpu.is_gpu_device())
      self.skipTest("Skipping test because device is not a GPU.")
    else:
      self.assertTrue(plgpu.is_gpu_device())

  def test_get_gpu_info(self):
    device = jax.devices()[0]
    info = plgpu.get_gpu_info()
    self.assertIsInstance(info, plgpu.GpuInfo)
    self.assertEqual(info.arch_name, device.compute_capability)
    expected_version = gpu_info.gpu_version_from_device_kind(device.device_kind)
    self.assertEqual(info.gpu_version, expected_version)

    cc = info.compute_capability
    match info.gpu_version:
      case GpuVersion.T4:
        self.assertEqual(cc, 75)
      case GpuVersion.A100 | GpuVersion.A30:
        self.assertEqual(cc, 80)
      case GpuVersion.A10:
        self.assertEqual(cc, 86)
      case GpuVersion.L4 | GpuVersion.L40 | GpuVersion.RTX_4090:
        self.assertEqual(cc, 89)
      case GpuVersion.H100 | GpuVersion.H200 | GpuVersion.GH200:
        self.assertEqual(cc, 90)
      case GpuVersion.B200 | GpuVersion.GB200:
        self.assertEqual(cc, 100)
      case GpuVersion.B300 | GpuVersion.GB300:
        self.assertEqual(cc, 103)
      case GpuVersion.RTX_PRO_4500 | GpuVersion.RTX_PRO_5000 | GpuVersion.RTX_PRO_6000:
        self.assertEqual(cc, 120)
      case GpuVersion.GB10:
        self.assertEqual(cc, 121)
      case _:
        self.assertEqual(cc, 0)

  def test_get_gpu_info_given_gpu_version(self):
    info = plgpu.get_gpu_info()
    info_version = plgpu.get_gpu_info_from_version(info.gpu_version)
    self.assertEqual(info, info_version)

  @parameterized.parameters([
    ("NVIDIA A100-SXM4-40GB", GpuVersion.A100),
    ("NVIDIA A100-SXM4-80GB", GpuVersion.A100),
    ("NVIDIA A100-PCIE-40GB", GpuVersion.A100),
    ("NVIDIA A100 80GB PCIe", GpuVersion.A100),
    ("NVIDIA A10 WHATEVER", GpuVersion.A10),
    ("NVIDIA H100 80GB HBM3", GpuVersion.H100),
    ("NVIDIA H100 PCIe", GpuVersion.H100),
    ("NVIDIA H100 NVL", GpuVersion.H100),
    ("NVIDIA RTX 123", None),
    ("UNKNOWN", None),
  ])
  def test_gpu_version_from_device_kind(self, device_kind, expected):
    info = gpu_info.gpu_version_from_device_kind(device_kind)
    self.assertEqual(info, expected)


if __name__ == "__main__":
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
