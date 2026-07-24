"""Workaround for #38987: vector.extract_strided_slice verification error on Pallas-TPU.

When a Pallas-TPU kernel tries to access alternating columns (e.g. w[:,0::2], w[:,1::2])
the lowering produces a vector.extract_strided_slice with a stride that fails verification.

Workaround: use concat([lo, hi], axis=-1) layout instead of interleaved access.
Pack weights so even-indexed values are in the first half, odd in the second half (per tile).
Then the kernel accesses contiguous memory — no strided slice needed. Match with a block-local
permutation of the activation input.
"""
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest


def block_local_permute(x, block_k):
  """Permute x so within each block_k chunk, even cols come first, then odd.

  This matches the concat-unpack weight layout where w_tile = [lo_cols | hi_cols].
  """
  B, K = x.shape
  xb = x.reshape(B, K // block_k, block_k)
  xp = jnp.concatenate([xb[:, :, 0::2], xb[:, :, 1::2]], axis=2)
  return xp.reshape(B, K)


def concat_unpack_matmul(x_perm, packed_uint8, scale, block_k):
  """Matmul with concat-unpacked 4-bit weight (no strided-slice access).

  packed_uint8: [N, K/2] — nibble-packed weight (standard layout).
  x_perm: [B, K] — block-local-permuted activation (matches concat layout).
  scale: [N] — per-output-channel scale.

  Within each block_k tile, the unpack produces [lo_cols | hi_cols] which
  matches x_perm's [even | odd] layout. No extract_strided_slice needed.
  """
  w32 = packed_uint8.astype(jnp.int32)
  lo = (w32 & 0xF)
  hi = (w32 >> 4) & 0xF
  lo = jnp.where(lo >= 8, lo - 16, lo).astype(jnp.bfloat16)
  hi = jnp.where(hi >= 8, hi - 16, hi).astype(jnp.bfloat16)
  w = jnp.concatenate([lo, hi], axis=1)  # [N, K] in [even|odd] order
  out = (x_perm.astype(jnp.bfloat16) @ w.T).astype(jnp.float32)
  return out * scale[None, :]


class ConcatUnpackTest(absltest.TestCase):

  def test_concat_unpack_correctness(self):
    """Verify concat-unpack + block-local-permute gives correct matmul."""
    rng = np.random.default_rng(0)
    N, K, B, BK = 64, 32, 4, 16
    W = rng.standard_normal((N, K)).astype(np.float32) * 0.02
    s = np.maximum(np.abs(W).max(axis=1, keepdims=True) / 7.0, 1e-9)
    q = np.clip(np.round(W / s), -8, 7).astype(np.int8)
    x = rng.standard_normal((B, K)).astype(np.float32)

    # Reference: x @ (q * s).T
    ref = x @ (q * s).T

    # Pack
    n = q.astype(np.int32) & 0xF
    packed = (n[:, 0::2] | (n[:, 1::2] << 4)).astype(np.uint8)

    # Concat-unpack matmul with block-local permuted x
    x_perm = np.asarray(block_local_permute(jnp.asarray(x), BK))
    sf = jnp.asarray(s.reshape(N).astype(np.float32))
    out = concat_unpack_matmul(jnp.asarray(x_perm), jnp.asarray(packed), sf, BK)

    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-2, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
