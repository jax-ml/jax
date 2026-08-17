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
  GpuTargetConfig = plgpu.GpuTargetConfig
else:
  plgpu = None
  GpuTargetConfig = gpu_info.GpuTargetConfig


TEST_DEVICE_CONFIGS = [
    ("NVIDIA A100-SXM4-40GB", ("8.0", 80)),
    ("NVIDIA A100-SXM4-80GB", ("8.0", 80)),
    # XLA a100_pcie_80.txtpb report device_description_str: A100 80GB
    # but device_kind is "NVIDIA A100 80GB PCIe"
    # ("NVIDIA A100 80GB PCIe", ("8.0", 80)),
    ("NVIDIA A10 WHATEVER", None),
    ("NVIDIA H100 80GB HBM3", ("9.0", 90)),
    ("NVIDIA H100 PCIe", ("9.0", 90)),
    ("MI250", ("gfx90a:sramecc+:xnack-", 0)),
    ("UNKNOWN", None),
    ("GFX1250", ("gfx1250", 0)),
    ("gfx90a:sramecc+:xnack-", None),
    ("gfx942", None),
    ("gfx950", None),
]


class GpuInfoTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not plgpu:
      self.skipTest("GPU jaxlib is not available")
    if not jtu.is_device_cuda() and not jtu.is_device_rocm():
      self.skipTest("Needs a GPU device")

  def test_get_gpu_info(self):
    try:
      info = plgpu.get_gpu_info()
    except ValueError:
      self.skipTest("Unsupported GPU device")
    device = jax.devices()[0]
    self.assertIsInstance(info, GpuTargetConfig)
    self.assertEqual(info.arch_name, device.compute_capability)

    if "rocm" == device.platform:
      self.assertEqual(info.compute_capability, 0)
    else:
      cc = int(float(info.compute_capability) * 10)
      self.assertEqual(info.compute_capability, cc)

  @parameterized.parameters(TEST_DEVICE_CONFIGS)
  def test_gpu_version_from_device_kind(self, device_kind, expected):
    info = gpu_info.gpu_version_from_device_kind(device_kind)
    if expected is not None:
      self.assertEqual((info.arch_name, info.compute_capability), expected)
    else:
      self.assertIsNone(info)

  @parameterized.parameters(TEST_DEVICE_CONFIGS)
  def test_get_gpu_info_abs_device(self, device_kind, expected):
    abstract_device = jax.sharding.AbstractDevice(
        device_kind=device_kind, num_cores=None, platform="gpu")
    abstract_mesh = jax.sharding.AbstractMesh(
        (1,), ("x",), (jax.sharding.AxisType.Explicit,),
        abstract_device=abstract_device)
    with jax.sharding.use_abstract_mesh(abstract_mesh):
      try:
        info = plgpu.get_gpu_info()
      except ValueError:
        self.skipTest("Unsupported GPU device")
      if expected is not None:
        self.assertEqual((info.arch_name, info.compute_capability), expected)
      else:
        self.assertIsNone(info)


if __name__ == "__main__":
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
