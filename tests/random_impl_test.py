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

"""Test JAX PRNG functionality of built-in implementations."""

from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax import random
from jax._src import test_util as jtu
from jax._src.random import philox2x32 as philox2x32_internal
from jax._src.random import philox4x32 as philox4x32_internal
from jax._src.random import threefry2x32 as threefry2x32_internal
from jax._src.random import threefry4x32 as threefry4x32_internal
import numpy as np

jax.config.parse_flags_with_absl()


class _PRNGConfig(NamedTuple):
  impl: str
  dtype_name: str
  key_shape: tuple[int, ...]
  key_dtype: np.dtype


_PRNG_IMPLS = [
    _PRNGConfig('threefry2x32', 'key<fry>', (2,), np.uint32),
    _PRNGConfig('threefry4x32', 'key<fry4>', (4,), np.uint32),
    _PRNGConfig('philox2x32', 'key<phx2>', (1,), np.uint32),
    _PRNGConfig('philox4x32', 'key<phx4>', (2,), np.uint32),
]


class RandomImplTest(jtu.JaxTestCase):
  """Tests comparing JAX core hash primitives against KAT vectors."""

  @parameterized.parameters(
      # KAT vectors from random123/tests/kat_vectors (20 rounds).
      dict(
          key=[0x00000000, 0x00000000],
          counter=[0x00000000, 0x00000000],
          expected=[0x6B200159, 0x99BA4EFE],
      ),
      dict(
          key=[0xFFFFFFFF, 0xFFFFFFFF],
          counter=[0xFFFFFFFF, 0xFFFFFFFF],
          expected=[0x1CB996FC, 0xBB002BE7],
      ),
      dict(
          key=[0x13198A2E, 0x03707344],
          counter=[0x243F6A88, 0x85A308D3],
          expected=[0xC4923A9C, 0x483DF7A0],
      ),
  )
  def test_threefry2x32_kat_vectors(self, key, counter, expected):
    """Test threefry2x32 primitive against Known Answer Test vectors."""
    actual = threefry2x32_internal.threefry2x32_p.bind(
        np.uint32(key[0]),
        np.uint32(key[1]),
        np.uint32(counter[0]),
        np.uint32(counter[1]),
    )
    self.assertArraysEqual(
        np.asarray(actual, dtype=np.uint32),
        np.asarray(expected, dtype=np.uint32),
    )

  @parameterized.parameters(
      # KAT vectors from random123/tests/kat_vectors (10 rounds).
      dict(
          key=[0x00000000],
          counter=[0x00000000, 0x00000000],
          expected=[0xFF1DAE59, 0x6CD10DF2],
      ),
      dict(
          key=[0xFFFFFFFF],
          counter=[0xFFFFFFFF, 0xFFFFFFFF],
          expected=[0x2C3F628B, 0xAB4FD7AD],
      ),
      dict(
          key=[0x13198A2E],
          counter=[0x243F6A88, 0x85A308D3],
          expected=[0xDD7CE038, 0xF62A4C12],
      ),
  )
  def test_philox2x32_kat_vectors(self, key, counter, expected):
    """Test philox2x32 primitive against Known Answer Test vectors."""
    k_u32 = np.uint32(key[0])
    c0_u32 = np.uint32(counter[0])
    c1_u32 = np.uint32(counter[1])
    actual = philox2x32_internal.philox2x32_p.bind(k_u32, c0_u32, c1_u32)
    self.assertArraysEqual(
        np.asarray(actual, dtype=np.uint32),
        np.asarray(expected, dtype=np.uint32),
    )

  @parameterized.parameters(
      # KAT vectors from random123/tests/kat_vectors (10 rounds).
      dict(
          key=[0x00000000, 0x00000000],
          counter=[0x00000000, 0x00000000, 0x00000000, 0x00000000],
          expected=[0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8],
      ),
      dict(
          key=[0xFFFFFFFF, 0xFFFFFFFF],
          counter=[0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
          expected=[0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD],
      ),
      dict(
          key=[0xA4093822, 0x299F31D0],
          counter=[0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344],
          expected=[0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1],
      ),
  )
  def test_philox4x32_kat_vectors(self, key, counter, expected):
    """Test philox4x32 primitive against Known Answer Test vectors."""
    actual = philox4x32_internal.philox4x32_p.bind(
        np.uint32(key[0]),
        np.uint32(key[1]),
        np.uint32(counter[0]),
        np.uint32(counter[1]),
        np.uint32(counter[2]),
        np.uint32(counter[3]),
    )
    self.assertArraysEqual(
        np.asarray(actual, dtype=np.uint32),
        np.asarray(expected, dtype=np.uint32),
    )

  @parameterized.parameters(
      # KAT vectors from random123/tests/kat_vectors (20 rounds).
      dict(
          key=[0x00000000, 0x00000000, 0x00000000, 0x00000000],
          counter=[0x00000000, 0x00000000, 0x00000000, 0x00000000],
          expected=[0x9C6CA96A, 0xE17EAE66, 0xFC10ECD4, 0x5256A7D8],
      ),
      dict(
          key=[0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
          counter=[0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
          expected=[0x2A881696, 0x57012287, 0xF6C7446E, 0xA16A6732],
      ),
      dict(
          key=[0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89],
          counter=[0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344],
          expected=[0x59CD1DBB, 0xB8879579, 0x86B5D00C, 0xAC8B6D84],
      ),
  )
  def test_threefry4x32_kat_vectors(self, key, counter, expected):
    """Test threefry4x32 primitive against Known Answer Test vectors."""
    actual = threefry4x32_internal.threefry4x32_p.bind(
        np.uint32(key[0]),
        np.uint32(key[1]),
        np.uint32(key[2]),
        np.uint32(key[3]),
        np.uint32(counter[0]),
        np.uint32(counter[1]),
        np.uint32(counter[2]),
        np.uint32(counter[3]),
    )
    self.assertArraysEqual(
        np.asarray(actual, dtype=np.uint32),
        np.asarray(expected, dtype=np.uint32),
    )


@parameterized.named_parameters((p.impl, p) for p in _PRNG_IMPLS)
class PRNGImplTest(jtu.JaxTestCase):
  """Integration tests for jax.random with built-in PRNG implementations."""

  def test_key_creation(self, p):
    key = random.key(0, impl=p.impl)
    self.assertEqual(key.dtype.name, p.dtype_name)

  def test_key_data_shape(self, p):
    key = random.key(42, impl=p.impl)
    data = random.key_data(key)
    self.assertEqual(data.shape, p.key_shape)
    self.assertEqual(data.dtype, p.key_dtype)

  def test_split(self, p):
    key = random.key(0, impl=p.impl)
    keys = random.split(key, 5)
    data = random.key_data(keys)
    self.assertEqual(data.shape, (5, *p.key_shape))
    # All sub-keys should be different.
    for i in range(5):
      for j in range(i + 1, 5):
        self.assertFalse(np.array_equal(data[i], data[j]))

  def test_fold_in(self, p):
    key = random.key(0, impl=p.impl)
    k1 = random.fold_in(key, 0)
    k2 = random.fold_in(key, 1)
    d1 = random.key_data(k1)
    d2 = random.key_data(k2)
    self.assertFalse(np.array_equal(d1, d2))

  def test_random_bits(self, p):
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
      with self.subTest(dtype=dtype):
        key = random.key(0, impl=p.impl)
        with jax.enable_x64(dtype == np.uint64):
          bits = random.bits(key, shape=(100,), dtype=dtype)
          self.assertEqual(bits.shape, (100,))
          self.assertEqual(bits.dtype, dtype)
          # Basic plausibility check: not all zeros.
          self.assertTrue(np.any(np.asarray(bits) != 0))

  def test_uniform(self, p):
    key = random.key(0, impl=p.impl)
    vals = random.uniform(key, shape=(1000,))
    self.assertEqual(vals.shape, (1000,))
    self.assertTrue(np.all(np.asarray(vals) >= 0))
    self.assertTrue(np.all(np.asarray(vals) < 1))
    # Basic plausibility check: mean should be near 0.5.
    self.assertAlmostEqual(float(jnp.mean(vals)), 0.5, delta=0.02)

  def test_normal(self, p):
    key = random.key(0, impl=p.impl)
    vals = random.normal(key, shape=(1000,))
    self.assertEqual(vals.shape, (1000,))
    # Basic plausibility check: mean should be near 0, std near 1.
    self.assertAlmostEqual(float(jnp.mean(vals)), 0.0, delta=0.1)
    self.assertAlmostEqual(float(jnp.std(vals)), 1.0, delta=0.1)

  def test_deterministic(self, p):
    key1 = random.key(12345, impl=p.impl)
    key2 = random.key(12345, impl=p.impl)
    vals1 = random.uniform(key1, shape=(100,))
    vals2 = random.uniform(key2, shape=(100,))
    np.testing.assert_array_equal(vals1, vals2)

  def test_different_seeds_differ(self, p):
    key1 = random.key(0, impl=p.impl)
    key2 = random.key(1, impl=p.impl)
    vals1 = random.uniform(key1, shape=(100,))
    vals2 = random.uniform(key2, shape=(100,))
    self.assertFalse(np.array_equal(vals1, vals2))

  def test_jit_compatible(self, p):
    @jax.jit
    def generate(seed):
      key = random.key(seed, impl=p.impl)
      return random.uniform(key, shape=(10,))

    vals = generate(0)
    self.assertEqual(vals.shape, (10,))

  def test_vmap_split(self, p):
    key = random.key(0, impl=p.impl)
    keys = random.split(key, 8)
    vals = jax.vmap(lambda k: random.uniform(k, shape=(5,)))(keys)
    expected = jnp.stack([random.uniform(k, shape=(5,)) for k in keys])
    self.assertArraysEqual(vals, expected)

  def test_split_fold_in_equivalence(self, p):
    key = random.key(0, impl=p.impl)
    split_key = random.split(key, 5)
    folded_key = jax.vmap(lambda i: random.fold_in(key, i))(jnp.arange(5))
    self.assertArraysEqual(
        random.key_data(split_key), random.key_data(folded_key)
    )


if __name__ == '__main__':
  absltest.main()
