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
"""Implementation of the Threefry PRNG as a Pallas GPU kernel."""

import math
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton
from jax._src import prng


def make_threefry_kernel(block_size: int = 256):
    def kernel(key_ref, out_ref):
        pid = pl.program_id(0)
        idx = jax.lax.iota(jnp.uint32, block_size)
        global_idx = (pid.astype(jnp.uint32) * jnp.uint32(block_size)) + idx

        k0 = key_ref[0]
        k1 = key_ref[1]
        ks0 = k0
        ks1 = k1
        ks2 = k0 ^ k1 ^ jnp.uint32(0x1BD11BDA)

        c0 = jnp.zeros_like(global_idx)
        c1 = global_idx

        x0 = c0 + ks0
        x1 = c1 + ks1

        rotations = [13, 15, 26, 6, 17, 29, 16, 24]
        for r in range(20):
            x0 = x0 + x1
            rot = rotations[r % 8]
            x1 = (x1 << jnp.uint32(rot)) | (x1 >> jnp.uint32(32 - rot))
            x1 = x1 ^ x0

            if r == 3:
                x0 = x0 + ks1
                x1 = x1 + ks2 + jnp.uint32(1)
            elif r == 7:
                x0 = x0 + ks2
                x1 = x1 + ks0 + jnp.uint32(2)
            elif r == 11:
                x0 = x0 + ks0
                x1 = x1 + ks1 + jnp.uint32(3)
            elif r == 15:
                x0 = x0 + ks1
                x1 = x1 + ks2 + jnp.uint32(4)
            elif r == 19:
                x0 = x0 + ks2
                x1 = x1 + ks0 + jnp.uint32(5)

        out_ref[:, 0] = x0
        out_ref[:, 1] = x1

    return kernel


def plthreefry_random_bits(key, bit_width: int, shape, block_size: int = 256):
    """
    Generates random bits using natively unrolled Threefry2x32 via Triton.

    This matches JAX's `threefry_partitionable` implementation natively.
    """
    if bit_width not in (32, 64):
        raise ValueError(f"bit_width must be 32 or 64, got {bit_width}")

    flat_size = int(math.prod(shape))

    if flat_size > jnp.iinfo(jnp.uint32).max:
        raise ValueError(
            f"Shape too large ({flat_size} elements). "
            f"Shapes larger than {jnp.iinfo(jnp.uint32).max} are not yet supported "
            "due to 32-bit counter limits."
        )

    if flat_size == 0:
        dtype = jnp.uint64 if bit_width == 64 else jnp.uint32
        return jnp.empty(shape, dtype=dtype)

    padded_size = (flat_size + block_size - 1) // block_size * block_size
    num_blocks = padded_size // block_size

    out_flat = pl.pallas_call(
        make_threefry_kernel(block_size),
        in_specs=[pl.BlockSpec((2,), lambda i: (0,))],
        out_specs=pl.BlockSpec((block_size, 2), lambda i: (i, 0)),
        grid=(num_blocks,),
        out_shape=jax.ShapeDtypeStruct((num_blocks * block_size, 2), jnp.uint32),
        compiler_params=pltriton.CompilerParams(),
    )(key)

    out_flat = out_flat[:flat_size]

    if bit_width == 32:
        # JAX's threefry_partitionable uses counter=(counts_hi, counts_lo) and folds
        # output as x0 ^ x1 for 32-bit widths. This kernel uses (c0=0, c1=global_idx)
        # which matches JAX's scheme for flat_size < 2^32.
        # See jax._src.prng._threefry_random_bits_partitionable, approx line 630
        out = out_flat[:, 0] ^ out_flat[:, 1]
    elif bit_width == 64:
        out = (out_flat[:, 0].astype(jnp.uint64) << jnp.uint64(32)) | out_flat[:, 1].astype(jnp.uint64)

    return jnp.reshape(out, shape)


plthreefry_prng_impl = prng.PRNGImpl(
    key_shape=(2,),
    seed=prng.threefry_seed,
    split=prng.threefry_split,
    random_bits=plthreefry_random_bits,
    fold_in=prng.threefry_fold_in,
    name="pallas_threefry2x32_gpu",
    tag="plfry")

prng.register_prng(plthreefry_prng_impl)
