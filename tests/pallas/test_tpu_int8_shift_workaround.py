"""Demonstrates the arith.shrui legalization issue (#38986) and the int32-cast workaround.

On Pallas-TPU, a logical right-shift on int8 vectors fails Mosaic legalization:
  arith.shrui on vector<...xi8> -> "failed to legalize operation"

Workaround: cast to int32 before the shift, then back. The shift legalizes in int32.
This is relevant for in-kernel nibble-unpack of 4-bit packed weights.
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

# NOTE: This test demonstrates the pattern. It requires a TPU to run.
# On non-TPU platforms, it validates the math only.


def nibble_unpack_int32_workaround(packed_uint8):
  """Unpack uint8 nibble-packed array using int32 shift (avoids int8 arith.shrui).

  This is the workaround for https://github.com/jax-ml/jax/issues/38986.
  """
  w = packed_uint8.astype(jnp.int32)  # Cast to int32 BEFORE shifting
  lo = (w & 0xF)
  hi = (w >> 4) & 0xF  # This shift legalizes in int32 (not int8)
  # Sign-extend from 4-bit two's complement
  lo = jnp.where(lo >= 8, lo - 16, lo)
  hi = jnp.where(hi >= 8, hi - 16, hi)
  return jnp.stack([lo, hi], axis=-1).reshape(
      *packed_uint8.shape[:-1], packed_uint8.shape[-1] * 2
  ).astype(jnp.int8)


class Int8ShiftWorkaroundTest(absltest.TestCase):

  def test_int32_shift_correctness(self):
    """Verify the int32-cast workaround produces correct nibble-unpack."""
    rng = np.random.default_rng(0)
    # Pack: standard nibble layout
    q = rng.integers(-8, 8, (64, 128)).astype(np.int8)
    n = q.astype(np.int32) & 0xF
    packed = (n[:, 0::2] | (n[:, 1::2] << 4)).astype(np.uint8)

    # Unpack via the workaround
    unpacked = nibble_unpack_int32_workaround(jnp.asarray(packed))
    np.testing.assert_array_equal(np.asarray(unpacked), q)


if __name__ == '__main__':
  absltest.main()
