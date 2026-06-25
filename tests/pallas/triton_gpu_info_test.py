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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax._src.pallas.triton import gpu_info
from jax.experimental.pallas import triton as plgpu


class GpuInfoTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_cuda():
      self.assertFalse(plgpu.is_gpu_device())
      self.skipTest("Skipping test because device is not a GPU.")
    else:
      self.assertTrue(plgpu.is_gpu_device())

  def test_get_gpu_info(self):
    device = jax.devices()[0]
    info = plgpu.get_gpu_info()
    self.assertIsInstance(info, plgpu.GpuInfo)
    for version in plgpu.GpuVersion:
      if version.value in device.device_kind:
        self.assertEqual(info.gpu_version, version)
        return
    self.fail(f"Unexpected device kind: {device.device_kind}")

  def test_get_gpu_info_given_gpu_version(self):
    info = plgpu.get_gpu_info()
    info_version = plgpu.get_gpu_info_from_version(info.gpu_version)
    self.assertEqual(info, info_version)

  @parameterized.parameters([
    ("NVIDIA A100-SXM4-40GB", plgpu.GpuVersion.A100),
    ("NVIDIA A100-SXM4-80GB", plgpu.GpuVersion.A100),
    ("NVIDIA A100-PCIE-40GB", plgpu.GpuVersion.A100),
    ("NVIDIA A100 80GB PCIe", plgpu.GpuVersion.A100),
    ("NVIDIA A10 WHATEVER", plgpu.GpuVersion.A10),
    ("NVIDIA H100 80GB HBM3", plgpu.GpuVersion.H100),
    ("NVIDIA H100 PCIe", plgpu.GpuVersion.H100),
    ("NVIDIA H100 NVL", plgpu.GpuVersion.H100),
    ("NVIDIA RTX 123", None),
    ("UNKNOWN", None),
  ])
  def test_gpu_version_from_device_kind(self, device_kind, expected):
    info = gpu_info.gpu_version_from_device_kind(device_kind)
    self.assertEqual(info, expected)


if __name__ == "__main__":
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
