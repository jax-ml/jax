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
"""
Implementation of the Xoshiro128++ PRNG as a Pallas kernel.

Unlike the canonical Xoshiro128++ algorithm, which expects 128 bits of
pre-mixed entropy for initialization, this implementation is adapted for
massively parallel execution. Because thousands of GPU threads cannot share
the same initial 128-bit seed, this kernel derives the 128-bit state
dynamically for each thread. It mixes a global thread index with the provided
JAX PRNG key using a MurmurHash3 avalanche function and Weyl sequence steps,
guaranteeing a unique starting state for every thread.
"""
import math
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax._src import prng
from jax.experimental.pallas import triton as pltriton
from jax._src.prng import threefry_2x32

def rotl32(x, k):
    return (x << jnp.uint32(k)) | (x >> jnp.uint32(32 - k))

def _mix32(z):
    """MurmurHash3 / SplitMix32 finalizer"""
    z += jnp.uint32(1)
    z = (z ^ (z >> jnp.uint32(16))) * jnp.uint32(0x85ebca6b)
    z = (z ^ (z >> jnp.uint32(13))) * jnp.uint32(0xc2b2ae35)
    return z ^ (z >> jnp.uint32(16))

def make_xoshiro_kernel(block_size: int, elements_per_thread: int):
    def kernel(key_ref, out_ref):
        pid = pl.program_id(0)
        idx = jax.lax.iota(jnp.uint32, block_size)
        global_idx = (pid.astype(jnp.uint32) * jnp.uint32(block_size)) + idx

        k0 = key_ref[0]
        k1 = key_ref[1]

        # Mix k0 and k1. 0x6C62272E (the upper 32 bits of the FNV-128 offset basis)
        # is used as a multiplier to break symmetry between k0 and k1.
        k0_mixed = _mix32(k0)
        k1_mixed = _mix32(k1) * jnp.uint32(0x6C62272E)
        thread_state = global_idx ^ k0_mixed ^ k1_mixed

        # 0x9E3779B9 is a Weyl sequence constant (golden ratio * 2^32).
        # We use 4 Weyl steps to initialize the 4 separate 32-bit state words.
        weyl = jnp.uint32(0x9E3779B9)
        s0 = _mix32(thread_state + weyl)
        s1 = _mix32(thread_state + weyl * jnp.uint32(2))
        s2 = _mix32(thread_state + weyl * jnp.uint32(3))
        s3 = _mix32(thread_state + weyl * jnp.uint32(4))

        s0 = jnp.where(s0 == 0, jnp.uint32(1), s0)

        for i in range(elements_per_thread):
            result = rotl32(s0 + s3, 7) + s0

            t = s1 << jnp.uint32(9)
            s2 ^= s0
            s3 ^= s1
            s1 ^= s2
            s0 ^= s3
            s2 ^= t
            s3 = rotl32(s3, 11)

            # Note on memory layout (thread-first ordering):
            # Threads write to columns instead of rows. After flattening (.T) in the
            # wrapper, the output groups elements by thread. This is intentional to
            # maximize performance for sequential workloads (like MCTS) by keeping
            # the stateful generator within registers.
            out_ref[i, :] = result

    return kernel

def xoshiro_random_bits(key, bit_width: int, shape, block_size: int = 256, elements_per_thread: int = 8):
    if bit_width not in (32, 64):
        raise ValueError(f"bit_width must be 32 or 64, got {bit_width}")

    multiplier = 2 if bit_width == 64 else 1
    flat_size = int(math.prod(shape)) * multiplier

    if flat_size > jnp.iinfo(jnp.uint32).max:
        raise ValueError(
            f"Shape too large ({flat_size} elements). "
            f"Shapes larger than {jnp.iinfo(jnp.uint32).max} are not yet supported "
            "due to 32-bit counter limits."
        )

    if flat_size == 0:
        dtype = jnp.uint64 if bit_width == 64 else jnp.uint32
        return jnp.empty(shape, dtype=dtype)

    tile = block_size * elements_per_thread
    padded = (flat_size + tile - 1) // tile * tile
    num_blocks = padded // tile

    out_flat = pl.pallas_call(
        make_xoshiro_kernel(block_size, elements_per_thread),
        in_specs=[pl.BlockSpec((2,), lambda i: (0,))],
        out_specs=pl.BlockSpec(
            (elements_per_thread, block_size), lambda i: (0, i)
        ),
        grid=(num_blocks,),
        out_shape=jax.ShapeDtypeStruct(
            (elements_per_thread, num_blocks * block_size), jnp.uint32
        ),
        compiler_params=pltriton.CompilerParams(),
    )(key)

    out = jnp.reshape(out_flat.T, (-1,))[:flat_size]

    if bit_width == 64:
        out = jnp.reshape(out, (-1, 2))
        out = (
            (out[:, 0].astype(jnp.uint64) << jnp.uint64(32))
            | out[:, 1].astype(jnp.uint64)
        )

    return jnp.reshape(out, shape)

def xoshiro_split(key, shape):
    flat_size = int(math.prod(shape))
    indices = jnp.arange(flat_size, dtype=jnp.uint32)
    counters = jnp.stack([indices, jnp.zeros_like(indices)], axis=1)
    new_keys = jax.vmap(lambda c: threefry_2x32(key, c))(counters)
    return jnp.reshape(new_keys, tuple(shape) + (2,))

def xoshiro_fold_in(key, data):
    data32 = jnp.uint32(data)
    counter = jnp.array([data32, jnp.uint32(0)], dtype=jnp.uint32)
    return threefry_2x32(key, counter)

plxoshiro_prng_impl = prng.PRNGImpl(
    key_shape=(2,),
    seed=prng.threefry_seed,
    split=xoshiro_split,
    random_bits=xoshiro_random_bits,
    fold_in=xoshiro_fold_in,
    name="pallas_xoshiro128pp",
    tag="plxos",
)

prng.register_prng(plxoshiro_prng_impl)
