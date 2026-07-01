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

import re
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
    if jtu.is_device_cuda() or jtu.is_device_rocm():
      self.assertTrue(plgpu.is_gpu_device())
    else:
      self.assertFalse(plgpu.is_gpu_device())
      self.skipTest("Skipping test because device is not a GPU.")

  def test_get_gpu_info(self):
    device = jax.devices()[0]
    info = plgpu.get_gpu_info()
    self.assertIsInstance(info, plgpu.GpuInfo)
    self.assertEqual(info.arch_name, device.compute_capability)
    for version in plgpu.GpuVersion:
      # version.value is a regex fragment (see gpu_info.py), not a literal
      # prefix, e.g. AMD product-name families like "AMD Instinct MI3...".
      if re.match(version.value, device.device_kind):
        self.assertEqual(info.gpu_version, version)
        return
    self.fail(f"Unexpected device kind: {device.device_kind}")

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
    ("gfx908", GpuVersion.GFX908),
    ("gfx90a", GpuVersion.GFX90A),
    ("gfx90a:sramecc+:xnack-", GpuVersion.GFX90A),
    ("gfx942", GpuVersion.GFX942),
    ("gfx942:sramecc+:xnack-", GpuVersion.GFX942),
    ("gfx950", GpuVersion.GFX950),
    ("gfx950:sramecc+:xnack-", GpuVersion.GFX950),
    ("AMD Instinct MI100", GpuVersion.MI1XXX),
    ("AMD Instinct MI210", GpuVersion.MI2XXX),
    ("AMD Instinct MI250X", GpuVersion.MI2XXX),
    ("AMD Instinct MI300A", GpuVersion.MI3XXX),
    ("AMD Instinct MI300X", GpuVersion.MI3XXX),
    ("AMD Instinct MI325X", GpuVersion.MI3XXX),
    ("AMD Instinct MI350X", GpuVersion.MI35XXX),
    ("AMD Instinct MI350P", GpuVersion.MI35XXX),
    ("AMD Instinct MI355X", GpuVersion.MI35XXX),
  ])
  def test_gpu_version_from_device_kind(self, device_kind, expected):
    info = gpu_info.gpu_version_from_device_kind(device_kind)
    self.assertEqual(info, expected)


if __name__ == "__main__":
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
