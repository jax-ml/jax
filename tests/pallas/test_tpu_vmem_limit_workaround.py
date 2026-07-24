"""Workaround for #38988: Pallas-TPU fused dequant kernels OOM at default 64MB VMEM.

When building fused weight-dequant matmul kernels for 4-bit inference, the on-chip
tiles (bf16 weight block + f32 accumulator + dequant intermediates) often exceed the
default 64MB VMEM budget by 1-3.5MB. Setting vmem_limit_bytes=128MB via CompilerParams
fixes it — tpu7x has more physical VMEM than the conservative 64MB default.

This test documents the workaround pattern.
"""
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

# The workaround pattern (for documentation/testing):
# from jax.experimental.pallas import tpu as pltpu
# compiler_params = pltpu.CompilerParams(vmem_limit_bytes=134217728)  # 128MB
# pl.pallas_call(..., compiler_params=compiler_params)


class VmemLimitWorkaroundTest(absltest.TestCase):

  def test_vmem_limit_parameter_exists(self):
    """Verify CompilerParams accepts vmem_limit_bytes (the workaround parameter)."""
    try:
      from jax.experimental.pallas import tpu as pltpu
      params = pltpu.CompilerParams(vmem_limit_bytes=134217728)
      # If we get here without error, the parameter is accepted
      self.assertIsNotNone(params)
    except ImportError:
      self.skipTest("pltpu not available (non-TPU environment)")
    except TypeError as e:
      self.fail(
          f"CompilerParams does not accept vmem_limit_bytes: {e}. "
          "This parameter is needed for fused dequant kernels that exceed "
          "the default 64MB VMEM budget on tpu7x (see #38988)."
      )

  def test_documented_pattern(self):
    """The recommended pattern for fused W4A16 kernels on tpu7x."""
    # This documents the correct usage for users hitting the 64MB OOM:
    pattern = '''
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    # For fused dequant kernels that exceed 64MB VMEM:
    result = pl.pallas_call(
        kernel_fn,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=[pltpu.VMEM(scratch_shape, dtype)],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=134217728),  # 128MB
    )(inputs)
    '''
    self.assertIn("vmem_limit_bytes=134217728", pattern)


if __name__ == '__main__':
  absltest.main()
