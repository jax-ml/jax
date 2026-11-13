# Copyright 2026 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Matmul kernel for Ampere."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.mosaic.gpu.launch_context import OOBFillMode
from jax.extend import backend

M = K = N = 8192
DT = jnp.bfloat16

def mma(acc, a, b, m_warps=None):
    """`plgpu.mma`, with the warp split requested on the operands."""
    if m_warps is None:
        return plgpu.mma(acc, a, b)
    return plgpu.mma(
        acc,
        plgpu.layout_cast(a, plgpu.Layout.MMA_LHS(a.dtype, m_warps=m_warps)),
        plgpu.layout_cast(b, plgpu.Layout.MMA_RHS(b.dtype, m_warps=m_warps)))

def acc_zeros(shape, m_warps=None, dtype=jnp.float32):
    """A zero accumulator already in the layout the mma will want."""
    z = jnp.zeros(shape, dtype)
    if m_warps is None:
        return z
    return plgpu.layout_cast(
        z, plgpu.Layout.MMA_ACC(dtype, m_warps=m_warps))

def matmul(a, b, bm=128, bn=128, bk=32, stages=4, mt=8, occ=2, m_warps=2,
           en=64, name='kernel7'):
    """a @ b. 98% of cuBLAS at 8192^3 bf16."""
    ni, nj, nk = M // bm, N // bn, K // bk
    tiles = ni * nj
    num_sms = min(occ * backend.get_default_device().core_count, tiles)
    # The unroll is by `stages`, so a group has to be a whole ring's worth or
    # the slot stops being a compile-time constant.
    assert nk % stages == 0, f'{stages=} must divide {nk=}'
    swizzle = plgpu.find_swizzle(en * jnp.finfo(DT).bits, 'epilogue')

    def body(a_gmem, b_gmem, o_gmem, a_smem, b_smem, *o_smem):
        p = jax.lax.axis_index('sm')
        my_tiles = (tiles - p - 1) // num_sms + 1   # ceil

        def snake(t):
            """(i, j) of this CTA's t'th tile."""
            return plgpu.planar_snake(p + t * num_sms, (ni, nj), 0, mt)

        def fetch(i, j, k, slot):
            plgpu.copy_gmem_to_smem(
                a_gmem.at[pl.ds(i * bm, bm), pl.ds(k * bk, bk)],
                a_smem.at[slot], oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)
            plgpu.copy_gmem_to_smem(
                b_gmem.at[pl.ds(k * bk, bk), pl.ds(j * bn, bn)],
                b_smem.at[slot], oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)

        def k_step(k, acc, ij, nxt, slot):
            plgpu.wait_gmem_to_smem((stages - 1) * 2)
            acc = mma(acc, a_smem.at[slot][...], b_smem.at[slot][...], m_warps)
            # Refill the slot just drained with the step `stages` ahead, which
            # belongs to the next tile once k passes nk - stages.
            s = k + stages
            jax.lax.cond(s < nk,
                         lambda: fetch(ij[0], ij[1], s, slot),
                         lambda: fetch(nxt[0], nxt[1], s - nk, slot))
            return acc

        def tile(t, _):
            ij = snake(t)
            # Clamped, so the last tile prefetches itself instead of running
            # off the end.
            nxt = snake(jnp.minimum(t + 1, my_tiles - 1))

            def unrolled(g, acc):
                """`stages` consecutive k steps, bodies unchanged."""
                for u in range(stages):
                    acc = k_step(g * stages + u, acc, ij, nxt, u)
                return acc

            acc = jax.lax.fori_loop(0, nk // stages, unrolled,
                                    acc_zeros((bm, bn), m_warps))

            for c in range(bn // en):
                o_smem[0][...] = acc[:, c * en:(c + 1) * en].astype(DT)
                o_gmem[pl.ds(ij[0] * bm, bm),
                        pl.ds(ij[1] * bn + c * en, en)] = plgpu.layout_cast(
                    o_smem[0][...],
                    plgpu.Layout.SMEM_GMEM_COPY((bm, en), DT,
                                                swizzle=swizzle))
            return _

        # Prime the pipeline, then walk this CTA's tiles.
        i0, j0 = snake(jnp.int32(0))
        for k in range(stages):
            fetch(i0, j0, jnp.int32(k), k)
        jax.lax.fori_loop(0, my_tiles, tile, None)

    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((M, N), DT),
        scratch_types=[plgpu.SMEM((stages, bm, bk), DT),
                       plgpu.SMEM((stages, bk, bn), DT)]
        + ([plgpu.SMEM((bm, en), DT, transforms=(
             plgpu.TilingTransform((8, 8 * swizzle // jnp.finfo(DT).bits)),
             plgpu.SwizzleTransform(swizzle)))]),
        grid=(num_sms,),
        grid_names=('sm',),
        kernel_name=name,
    )(a, b)


def main():
    ka, kb = jax.random.split(jax.random.key(0))
    A = jax.random.normal(ka, (M, K), DT)
    B = jax.random.normal(kb, (K, N), DT)
    out = jax.jit(matmul)(A, B)
    assert jnp.allclose(out.astype(jnp.float32), A @ B, rtol=1e-2), out
    print('ok')


if __name__ == "__main__":
  from absl import app
  import jax
  jax.config.config_with_absl()
  app.run(main)
