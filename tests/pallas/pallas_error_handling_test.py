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
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax._src import config
from jax._src import core as jax_core
from jax._src import test_util as jtu
from jax._src.lib.mlir import ir
from jax._src.pallas.mosaic import error_handling
from jax._src.pallas.mosaic import lowering as mosaic_lowering
from jax._src.state import primitives as state_primitives
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np


config.parse_flags_with_absl()

LOCATION_TEST_STRING = (
    r'loc("/squeeze"'
    r'(callsite("foo_fn"("third_party/foo.py":104:22) at '
    r'callsite("bar_fn"("third_party/bar.py":115:6) at '
    r'"<module>"("third_party/pallas_error_handling_test.py":181:2'
    r")))))"
)


class PallasCachedTemplateLocationTest(jtu.JaxTestCase):

  def test_cached_template_does_not_keep_first_equation_location(self):
    primitive = jax_core.Primitive("location_test")
    primitive.multiple_results = True
    ctx = SimpleNamespace(user_grid_indices=None)
    ctx.replace = lambda **_: ctx
    rule_context = SimpleNamespace(
        avals_out=(), aval_to_ir_type=lambda *_: None
    )
    rule_context.replace = lambda **_: rule_context

    with ir.Context(), ir.Location.file("first_equation.py", 10, 1):
      template = mosaic_lowering._emit_pallas_lowering_rule_as_fun(
          ctx, primitive, lambda *_: (), rule_context, ()
      )
      for op in template.operation.regions[0].blocks[0]:
        self.assertEqual(str(op.location), "loc(unknown)")
      self.assertEqual(str(template.operation.location), "loc(unknown)")


class PallasErrorHandlingTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest("Test only works on TPU.")

  @parameterized.named_parameters(
      ("cache_hit", 0, 1, "second", "first"),
      ("cache_miss", 1, 0, "first", "second"),
  )
  def test_cached_lowering_reports_current_source(
      self, first_offset, second_offset, expected, unexpected
  ):
    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        mesh=pltpu.TensorCoreMesh(axis_name="core", num_cores=1),
        scratch_types=(
            pltpu.VMEM((8, 256), jnp.float32),
            pltpu.VMEM((8, 128), jnp.float32),
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, disable_semaphore_checks=True
        ),
    )
    def kernel(x_hbm, o_hbm, x_ref, o_ref):
      pltpu.sync_copy(x_hbm, x_ref)
      start = jax.lax.axis_index("core") * 128
      first = x_ref[:, pl.ds(start + first_offset, 128)]
      second = x_ref[:, pl.ds(start + second_offset, 128)]
      o_ref[...] = first + second
      pltpu.sync_copy(o_ref, o_hbm)

    original = mosaic_lowering._emit_pallas_lowering_rule_as_fun
    emitted_get = 0

    def count_get(ctx, primitive, *args, **kwargs):
      nonlocal emitted_get
      if primitive is state_primitives.get_p:
        emitted_get += 1
      return original(ctx, primitive, *args, **kwargs)

    with mock.patch.object(
        mosaic_lowering, "_emit_pallas_lowering_rule_as_fun", count_get
    ):
      try:
        jax.jit(kernel)(jnp.zeros((8, 256), jnp.float32)).block_until_ready()
      except error_handling.MosaicError as error:
        self.assertIn(
            "CompileTimeMosaicUnprovenMemoryAccessAlignment", str(error)
        )
        frames = "".join(traceback.format_tb(error.__traceback__))
      else:
        self.fail("Expected a Mosaic alignment error")

    self.assertEqual(emitted_get, 1)
    expected_line = (
        f"{expected} = x_ref[:, pl.ds(start + {expected}_offset, 128)]"
    )
    unexpected_line = (
        f"{unexpected} = x_ref[:, pl.ds(start + {unexpected}_offset, 128)]"
    )
    self.assertIn(expected_line, frames)
    self.assertEqual(frames.count(expected_line), 1)
    self.assertNotIn(unexpected_line, frames)

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
  absltest.main(testLoader=jtu.JaxTestLoader())
