# Copyright 2024 The JAX Authors.
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

import functools
import traceback

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.pallas import mpmd
from jax._src.pallas.mosaic import error_handling
from jax._src.pallas.mosaic import tpu_info
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np


config.parse_flags_with_absl()


class PallasTpuSparseCoreLoweringErrorTest(jtu.JaxTestCase):
  """Tests for SparseCore-related errors during Pallas lowering."""

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Test only works on TPU.")

  def test_sparsecore_availability_check(self):
    """Checks that a ValueError is raised when targeting SparseCore on a
       device without it.
    """
    has_sc = tpu_info.get_tpu_info().sparse_core is not None

    def pallas_kernel(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    @jax.jit
    def kernel(x):
      return pl.pallas_call(
          pallas_kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          compiler_params=pltpu.CompilerParams(
              kernel_type=pltpu.CoreType.SC_VECTOR_SUBCORE
          ),
      )(x)

    x = jnp.ones((8,), dtype=jnp.bfloat16)
    if not has_sc:
      with self.assertRaisesRegex(
          ValueError, "SparseCore is not available on the current device"
      ):
        kernel.lower(x).compile()
    else:
      # Should work on devices with SparseCore
      hlo = kernel.lower(x).as_text("hlo")
      self.assertIn("custom-call", hlo)

  def test_mpmd_map_sparsecore_availability_check(self):
    """Checks that mpmd_map raises a ValueError when targeting SparseCore on a
       device without it.
    """
    has_sc = tpu_info.get_tpu_info().sparse_core is not None

    # We use a real mesh to carry kernel_type
    devices = np.array(jax.devices()[:1]).reshape((1,))
    mesh = jax.sharding.Mesh(devices, ("x",))
    mesh.kernel_type = pltpu.CoreType.SC_VECTOR_SUBCORE
    mesh.dimension_semantics = [pltpu.GridDimensionSemantics.PARALLEL]
    mesh.default_memory_space = pltpu.VMEM

    def kernel_fn(x_ref, o_ref):
      o_ref[...] = x_ref[...]

    @jax.jit
    def run_mpmd(x):
      return mpmd.mpmd_map(
          [(mesh, kernel_fn)],
          out_shapes=jax.ShapeDtypeStruct(x.shape, x.dtype),
      )(x)

    x = jnp.ones((8,), dtype=jnp.bfloat16)
    if not has_sc:
      with self.assertRaisesRegex(
          ValueError, "SparseCore is not available on the current device"
      ):
        run_mpmd.lower(x).compile()
    else:
      # Should work on devices with SparseCore
      hlo = run_mpmd.lower(x).as_text("hlo")
      self.assertIn("custom-call", hlo)


LOCATION_TEST_STRING = (
    r'loc("/squeeze"'
    r'(callsite("foo_fn"("third_party/foo.py":104:22) at '
    r'callsite("bar_fn"("third_party/bar.py":115:6) at '
    r'"<module>"("third_party/pallas_error_handling_test.py":181:2'
    r")))))"
)


class PallasErrorHandlingTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Test only works on TPU.")

  def test_non_singular_stride(self):
    input_arr = jax.random.uniform(
        jax.random.key(0), (8, 128), dtype=jnp.float32)
    out_shape = jax.ShapeDtypeStruct((8, 16), jnp.float32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
    )

    @functools.partial(pl.pallas_call, out_shape=out_shape, grid_spec=grid_spec)
    def test_kernel(input_ref, output_ref):
      x = input_ref[:, ::8]
      output_ref[...] = x

    # Test that a Mosaic error is raised. This assert is a guard against
    # underlying changes in Mosaic.
    # If this is fixed in future Mosaic releases we will need to change
    # the test example to force a different error.
    with self.assertRaisesRegex(
        error_handling.MosaicError,
        "Not Implemented: Stride on last dim is not 1",
    ):
      test_kernel(input_arr)

    # Test that the python source is the final frame in the traceback.
    tb_string = ""
    try:
      test_kernel(input_arr)
    except error_handling.MosaicError as e:
      tb_string = traceback.format_tb(e.__traceback__)
      tb_string = "".join(tb_string)
    self.assertEndsWith(tb_string, "x = input_ref[:, ::8]\n")

    @jax.jit
    def kernel_in_jitted_fn(x):
      return test_kernel(x)

    with self.subTest("inside_jitted_fn"):
      tb_string = ""
      try:
        kernel_in_jitted_fn(input_arr)
      except error_handling.MosaicError as e:
        tb_string = traceback.format_tb(e.__traceback__)
        tb_string = "".join(tb_string)
      self.assertEndsWith(tb_string, "x = input_ref[:, ::8]\n")

  def test_index_with_f32_verification_error(self):
    input_arr = jax.random.uniform(jax.random.key(0), (2, 2), dtype=jnp.float32)
    out_shape = jax.ShapeDtypeStruct((1, 1), jnp.float32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
    )

    @functools.partial(pl.pallas_call, out_shape=out_shape, grid_spec=grid_spec)
    def test_kernel(input_ref, output_ref):
      idx = input_ref[0, 0]
      output_ref[idx, 0] = input_ref[0, 0]

    # Test that a verification error is raised. This assert is a guard against
    # underlying changes in Pallas lowering.
    # If this is fixed in future Pallas releases we will need to change
    # the test example to force a different error.
    with self.assertRaisesRegex(
        error_handling.VerificationError,
        "must be signless-.*integer-like or memref of signless-integer, "
        "but got 'f32'",
    ):
      test_kernel(input_arr)

    # Test that the python source is the final frame in the traceback.
    tb_string = ""
    try:
      test_kernel(input_arr)
    except error_handling.MosaicError as e:
      tb_string = traceback.format_tb(e.__traceback__)
      tb_string = "".join(tb_string)
    self.assertEndsWith(tb_string, "output_ref[idx, 0] = input_ref[0, 0]\n")

  @parameterized.parameters(
      ((128,), (64,), jnp.float32),
      ((256,), (128,), jnp.bfloat16),
      ((512,), (256,), jnp.int8),
      # block size is not a power of 2
      ((3072,), (384,), jnp.float32),
      ((2304,), (1152,), jnp.float32),
      ((3072,), (768,), jnp.bfloat16),
      ((2560,), (1280,), jnp.bfloat16),
      ((3072,), (1536,), jnp.int8),
  )
  def test_infeasible_1d_block_spec_raises(
      self, total_shape, block_shape, dtype
  ):
    def kernel(x_ref, y_ref):
      y_ref[...] = x_ref[...] * 2

    x = jnp.arange(np.prod(total_shape), dtype=dtype).reshape(total_shape)
    x_spec = pl.BlockSpec(block_shape, lambda *args: args)
    fn = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(total_shape, dtype),
        in_specs=[x_spec],
        out_specs=x_spec,
        grid=tuple(tot // blk for tot, blk in zip(total_shape, block_shape,
                                                  strict=True)),
    )
    with self.assertRaisesRegex(
        ValueError,
        "The Pallas TPU lowering currently requires that rank 1 block shapes",
    ):
      fn(x)

  def test_parse_location_string(self):
    name, frames = error_handling.parse_location_string(LOCATION_TEST_STRING)
    self.assertEqual(name, "/squeeze")
    self.assertLen(frames, 3)
    self.assertEqual(frames[0].func_name, "foo_fn")
    self.assertEqual(frames[0].filename, "third_party/foo.py")
    self.assertEqual(frames[0].lineno, 104)
    self.assertEqual(frames[0].colno, 22)


if __name__ == "__main__":
  absltest.main()
