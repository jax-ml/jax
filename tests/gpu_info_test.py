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

import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import gpu_info
from jax._src import test_util as jtu


def _abstract_mesh(device_kind, platform="gpu"):
  abstract_device = jax.sharding.AbstractDevice(
      device_kind=device_kind, num_cores=None, platform=platform
  )
  return jax.sharding.AbstractMesh(
      (1,),
      ("x",),
      (jax.sharding.AxisType.Explicit,),
      abstract_device=abstract_device,
  )


class GpuInfoTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_cuda() and not jtu.is_device_rocm():
      self.skipTest("Needs a GPU device")

  def test_get_gpu_info(self):
    info = gpu_info.get_gpu_info()
    device = jax.devices()[0]
    self.assertIsInstance(info, gpu_info.GpuInfo)
    self.assertGreater(info.core_count, 0)
    self.assertGreater(info.smem_capacity_bytes, 0)

    if jtu.is_device_rocm():
      self.assertEqual(
          info.arch_name.split(":", 1)[0], device.compute_capability
      )
      self.assertEqual(info.compute_capability, 0)
    else:
      # remove last alphabetic char in the arch name:
      # 10.0a -> 10.0
      arch_name = info.arch_name.rstrip("af")
      self.assertEqual(arch_name, device.compute_capability)
      self.assertEqual(info.compute_capability, int(arch_name.replace(".", "")))


class GpuInfoStaticTest(jtu.JaxTestCase):

  @parameterized.parameters(
      ("NVIDIA A100-SXM4-40GB", ("8.0", 80)),
      ("NVIDIA H100 80GB HBM3", ("9.0a", 90)),
      ("NVIDIA H200", ("9.0a", 90)),  # an alias of "NVIDIA H200 141GB HBM3e"
      ("NVIDIA Graphics Device", None),
      ("UNKNOWN", None),
      ("gfx90a:sramecc+:xnack-", None),
      ("NVIDIA VR NVL72", ("10.7a", 107)),
      ("MI350X", ("gfx950:sramecc+:xnack-", 0)),
  )
  def test_get_gpu_info_abs_device(self, device_kind, expected):
    if gpu_info._gpu_spec is None:
      self.skipTest("jaxlib._gpu_spec is not available")
    with jax.sharding.use_abstract_mesh(_abstract_mesh(device_kind)):
      if expected is not None:
        info = gpu_info.get_gpu_info()
        self.assertIsInstance(info, gpu_info.GpuInfo)
        self.assertEqual((info.arch_name, info.compute_capability), expected)
      else:
        with self.assertRaises(ValueError):
          gpu_info.get_gpu_info()

  @parameterized.parameters(
      ("Live GPU 8.9", "8.9", "CUDA", 89),
      ("NVIDIA Graphics Device", "9.0", "CUDA", 90),
      ("Live GPU gfx950", "gfx950", "ROCM", 0),
  )
  @jtu.thread_unsafe_test()  # mock.patch mutates pxla.get_default_device
  def test_get_gpu_info_live_device(
      self, kind, arch_name, platform_name, compute_capability
  ):
    self.addCleanup(gpu_info.get_gpu_info.cache_clear)
    device = types.SimpleNamespace(
        platform="gpu",
        device_kind=kind,
        compute_capability=arch_name,
        core_count=64,
        shared_memory_per_block_optin=1024,
    )
    expected = gpu_info.GpuInfo(
        platform_name=platform_name,
        device_description_str=kind,
        arch_name=arch_name,
        compute_capability=compute_capability,
        core_count=64,
        smem_capacity_bytes=1024,
    )
    with mock.patch.object(
        gpu_info.pxla, "get_default_device", return_value=device
    ):
      gpu_info.get_gpu_info.cache_clear()
      self.assertEqual(gpu_info.get_gpu_info(), expected)
      with jax.sharding.use_abstract_mesh(_abstract_mesh(f"Not {kind}")):
        with self.assertRaises(ValueError):
          gpu_info.get_gpu_info()
      with jax.sharding.use_abstract_mesh(_abstract_mesh(kind)):
        self.assertEqual(gpu_info.get_gpu_info(), expected)

  @jtu.thread_unsafe_test()  # mock.patch mutates pxla.get_default_device
  def test_get_gpu_info_virtual_spec_device(self):
    if gpu_info._gpu_spec is None:
      self.skipTest("jaxlib._gpu_spec is not available")
    self.addCleanup(gpu_info.get_gpu_info.cache_clear)
    device = types.SimpleNamespace(platform="gpu", device_kind="NVIDIA H200")
    with mock.patch.object(
        gpu_info.pxla, "get_default_device", return_value=device
    ):
      gpu_info.get_gpu_info.cache_clear()
      info = gpu_info.get_gpu_info()
      self.assertEqual((info.arch_name, info.compute_capability), ("9.0a", 90))

  @parameterized.parameters(
      ("gpu", True),
      ("cuda", True),
      ("rocm", True),
      ("cpu", False),
  )
  def test_is_gpu_device(self, platform, expected):
    with jax.sharding.use_abstract_mesh(_abstract_mesh("ANY", platform)):
      self.assertEqual(gpu_info.is_gpu_device(), expected)


if __name__ == "__main__":
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
