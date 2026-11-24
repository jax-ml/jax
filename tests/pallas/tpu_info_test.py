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
from jax._src import core as jax_core
from jax._src import test_util as jtu
from jax._src import tpu_info
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


class TpuInfoTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.is_device_tpu():
      self.assertFalse(pltpu.is_tpu_device())
      self.skipTest("Skipping test because device is not a TPU.")
    else:
      self.assertTrue(pltpu.is_tpu_device())

  def test_get_tpu_info(self):
    device = jax.devices()[0]
    info = pltpu.get_tpu_info()
    self.assertIsInstance(info, pltpu.TpuInfo)
    match device.device_kind:
      case "TPU v2":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_V2)
      case "TPU v3":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_V3)
      case "TPU v4 lite":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_V4I)
      case "TPU v4":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_V4)
      case "TPU v5 lite":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_V5E)
      case "TPU v5":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_V5P)
      case "TPU v6 lite":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_V6E)
      case "TPU7":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_7)
      case "TPU7x":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_7X)
      case "TPU8i":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_8I)
      case "TPU8t":
        self.assertEqual(info.chip_version, pltpu.ChipVersion.TPU_8T)
      case _:
        self.fail(f"Unexpected device kind: {device.device_kind}")

  def test_is_matmul_supported_input_formats(self):
    info = pltpu.get_tpu_info()
    with self.subTest("str"):
      self.assertTrue(info.is_matmul_supported("float32", "float32"))
    with self.subTest("dtype"):
      self.assertTrue(
          info.is_matmul_supported(
              jax.numpy.float32, jax.numpy.dtype("float32")
          )
      )
    with self.subTest("array.dtype"):
      a = jax.numpy.array([1.0], dtype=jax.numpy.float32)
      b = jax.numpy.array([2.0], dtype="float32")
      self.assertTrue(info.is_matmul_supported(a.dtype, b.dtype))

  def test_is_lite(self):
    info = pltpu.get_tpu_info()
    if info.chip_version in {
        pltpu.ChipVersion.TPU_V4I,
        pltpu.ChipVersion.TPU_V5E,
        pltpu.ChipVersion.TPU_V6E,
    }:
      self.assertTrue(info.is_lite)
    else:
      self.assertFalse(info.is_lite)

  def test_is_split_chip(self):
    info = pltpu.get_tpu_info()
    if info.chip_version in {
        pltpu.ChipVersion.TPU_V2,
        pltpu.ChipVersion.TPU_V3,
        pltpu.ChipVersion.TPU_7,
        pltpu.ChipVersion.TPU_7X,
        pltpu.ChipVersion.TPU_8I,
    }:
      self.assertTrue(info.is_split_chip)
    else:
      self.assertFalse(info.is_split_chip)

  def test_is_megacore(self):
    info = pltpu.get_tpu_info()
    if info.chip_version in {
        pltpu.ChipVersion.TPU_V4,
        pltpu.ChipVersion.TPU_V5P,
    }:
      self.assertTrue(info.is_megacore)
    else:
      self.assertFalse(info.is_megacore)

  def test_num_cores(self):
    info = pltpu.get_tpu_info()
    if info.chip_version in {
        pltpu.ChipVersion.TPU_V4,
        pltpu.ChipVersion.TPU_V5P,
    }:
      self.assertEqual(info.num_cores, 2)
    else:
      self.assertEqual(info.num_cores, 1)

  def test_get_tpu_info_given_chip_version(self):
    info = pltpu.get_tpu_info()
    num_cores = (
        2
        if info.chip_version
        in {
            pltpu.ChipVersion.TPU_V4,
            pltpu.ChipVersion.TPU_V5P,
        }
        else 1
    )
    info_for_chip = pltpu.get_tpu_info_for_chip(info.chip_version, num_cores)
    self.assertEqual(info, info_for_chip)


class TpuInfoStaticTest(jtu.JaxTestCase):

  def test_all_chip_versions_properties(self):
    for version in pltpu.ChipVersion:
      match version:
        case pltpu.ChipVersion.TPU_V4 | pltpu.ChipVersion.TPU_V5P:
          # Megacore support
          # 1. Split mode
          info_split = pltpu.get_tpu_info_for_chip(version, 1)
          self.assertFalse(info_split.is_megacore)
          self.assertTrue(info_split.is_split_chip)
          self.assertEqual(version.num_physical_tensor_cores_per_chip, 2)
          # 2. Megacore mode
          info_mega = pltpu.get_tpu_info_for_chip(version, 2)
          self.assertTrue(info_mega.is_megacore)
          self.assertFalse(info_mega.is_megacore and info_mega.is_split_chip)
          self.assertTrue(version.supports_megacore)

        case (
            pltpu.ChipVersion.TPU_V2
            | pltpu.ChipVersion.TPU_V3
            | pltpu.ChipVersion.TPU_7
            | pltpu.ChipVersion.TPU_7X
            | pltpu.ChipVersion.TPU_8I
        ):
          # Dual core, no megacore
          info = pltpu.get_tpu_info_for_chip(version, 1)
          self.assertFalse(info.is_megacore)
          self.assertTrue(info.is_split_chip)
          self.assertFalse(version.supports_megacore)
          with self.assertRaisesRegex(
              ValueError,
              "Lite chips, single core chips, and dual-core chips that do not"
              " support",
          ):
            pltpu.get_tpu_info_for_chip(version, 2)

        case (
            pltpu.ChipVersion.TPU_V4I
            | pltpu.ChipVersion.TPU_V5E
            | pltpu.ChipVersion.TPU_V6E
        ):
          # Single core
          info = pltpu.get_tpu_info_for_chip(version, 1)
          self.assertFalse(info.is_megacore)
          self.assertFalse(info.is_split_chip)
          self.assertTrue(info.is_lite)
          self.assertFalse(version.supports_megacore)
          with self.assertRaisesRegex(
              ValueError,
              "Lite chips, single core chips, and dual-core chips that do not"
              " support",
          ):
            pltpu.get_tpu_info_for_chip(version, 2)
        case pltpu.ChipVersion.TPU_8T:
          # Single core, fish chip
          info = pltpu.get_tpu_info_for_chip(version, 1)
          self.assertFalse(info.is_megacore)
          self.assertFalse(info.is_split_chip)
          self.assertFalse(info.is_lite)
          self.assertFalse(version.supports_megacore)
          with self.assertRaisesRegex(
              ValueError,
              "Lite chips, single core chips, and dual-core chips that do not"
              " support",
          ):
            pltpu.get_tpu_info_for_chip(version, 2)
        case _:
          raise ValueError(f"Unexpected chip version: {version}")

  def test_is_matmul_supported_all_gens(self):
    for version in pltpu.ChipVersion:
      info = pltpu.get_tpu_info_for_chip(version, 1)
      # F32/BF16 always supported on all TPUs
      self.assertTrue(
          info.is_matmul_supported("float32", "float32"), msg=f"{version} f32"
      )
      if info.generation >= 4:
        self.assertTrue(
            info.is_matmul_supported("bfloat16", "bfloat16"),
            msg=f"{version} bf16",
        )

  @parameterized.product(
      dtype_name=["float64", "int64", "uint64", "complex64", "complex128"],
      shape=[(128,), (8, 128), (2, 16, 128)],
      tiling=[tpu_info.Tiling.COMPACT, tpu_info.Tiling.SPARSE_CORE],
  )
  def test_infer_tiling_rejects_large_dtypes(self, dtype_name, shape, tiling):
    abstract_mesh = jax.sharding.AbstractMesh(
        (), (), abstract_device=jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
    )
    with jax.enable_x64(True), jax.sharding.use_abstract_mesh(abstract_mesh):
      aval = jax_core.ShapedArray(shape, jnp.dtype(dtype_name))
      with self.assertRaisesRegex(
          ValueError, "Cannot infer TPU tiling for dtype"
      ):
        tpu_info.infer_tiling(aval, tiling=tiling)

  def test_infer_tiling_valid_dtypes(self):
    abstract_mesh = jax.sharding.AbstractMesh(
        (), (), abstract_device=jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
    )
    with jax.sharding.use_abstract_mesh(abstract_mesh):
      self.assertEqual(
          tpu_info.infer_tiling(jax_core.ShapedArray((8, 128), jnp.float32)),
          (8, 128),
      )
      self.assertEqual(
          tpu_info.infer_tiling(jax_core.ShapedArray((2, 128), jnp.bfloat16)),
          (2, 128),
      )
      self.assertEqual(
          tpu_info.infer_tiling(jax_core.ShapedArray((1, 128), jnp.int8)),
          (4, 128),
      )
      self.assertEqual(
          tpu_info.infer_tiling(jax_core.ShapedArray((1, 128), jnp.int4)),
          (8, 128),
      )

  @parameterized.parameters(0, -1)
  def test_get_tiling_factor_rejects_nonpositive_packing(self, packing):
    abstract_mesh = jax.sharding.AbstractMesh(
        (), (), abstract_device=jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
    )
    with jax.sharding.use_abstract_mesh(abstract_mesh):
      with self.assertRaisesRegex(ValueError, "packing must be positive"):
        tpu_info._get_tiling_factor(8, 8, packing)


if __name__ == "__main__":
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
