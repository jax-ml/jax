```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
```

# Kernel 1: Basic Ampere Matmul

```python
M = K = N = 8192
DT = jnp.bfloat16

def matmul1(a, b, bm=128, bn=128, bk=32, stages=4):
    def body(a_gmem, b_gmem, o_gmem):
        i, j = (jax.lax.axis_index(n) for n in 'ij')
        def step(indices, a_smem, b_smem, acc):
            return plgpu.mma(acc, a_smem[...], b_smem[...])

        # index_map returns *block* indices, not elements: block (i, k) of a
        # bm x bk-blocked A. i and j are closed over from the outer grid.
        acc = plgpu.emit_pipeline(
            step,
            grid=(K // bk,),
            in_specs=[
                plgpu.BlockSpec((bm, bk), lambda k: (i, k)),
                plgpu.BlockSpec((bk, bn), lambda k: (k, j)),
            ],
            max_concurrent_steps=stages,
            init_carry=jnp.zeros((bm, bn), jnp.float32),
        )(a_gmem, b_gmem)

        o_gmem[pl.ds(i * bm, bm), pl.ds(j * bn, bn)] = acc.astype(DT)

    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((M, N), DT),
        grid=(M // bm, N // bn),
        grid_names=('i', 'j'),
        kernel_name='matmul1',
    )(a, b)
```

# Kernel 2: Tile the grid for L2 locality

```python
def matmul2(a, b, bm=128, bn=128, bk=32, stages=4, mt=8):
    def body(a_gmem, b_gmem, o_gmem):
        # The only difference from matmul1: i is reassembled from the outer
        # band index and the position within the band.
        i = jax.lax.axis_index('i_outer') * mt + jax.lax.axis_index('i_inner')
        j = jax.lax.axis_index('j')

        def step(indices, a_smem, b_smem, acc):
            with jax.named_scope("mma"):
                return plgpu.mma(acc, a_smem[...], b_smem[...])

        acc = plgpu.emit_pipeline(
            step,
            grid=(K // bk,),
            in_specs=[
                plgpu.BlockSpec((bm, bk), lambda k: (i, k)),
                plgpu.BlockSpec((bk, bn), lambda k: (k, j)),
            ],
            max_concurrent_steps=stages,
            init_carry=jnp.zeros((bm, bn), jnp.float32),
        )(a_gmem, b_gmem)

        with jax.named_scope("storage"):
            o_gmem[pl.ds(i * bm, bm), pl.ds(j * bn, bn)] = acc.astype(DT)

    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((M, N), DT),
        grid=(M // bm // mt, N // bn, mt),
        grid_names=('i_outer', 'j', 'i_inner'),
        kernel_name='matmul2'
    )(a, b)
```

# Kernel 3: Persistent

TODO: this is actually slower than kernel 2 currently. 

```python
def matmul(a, b, bm=128, bn=128, bk=64, stages=2, mt=8, occ=2):
    """a @ b, persistent: each walking a stride of tiles."""
    ni, nj, nk = M // bm, N // bn, K // bk
    tiles = ni * nj
    assert ni % mt == 0, f'{mt=} must divide {ni=}'
    num_sms = min(occ * backend.get_default_device().core_count, tiles)
    assert nk >= stages, f'{stages=} needs at least that many k steps, got {nk}'

    def body(a_gmem, b_gmem, o_gmem, a_smem, b_smem):
        def fetch(i, j, k, slot):
            plgpu.copy_gmem_to_smem(
                a_gmem.at[pl.ds(i * bm, bm), pl.ds(k * bk, bk)],
                a_smem.at[slot],
                oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)
            plgpu.copy_gmem_to_smem(
                b_gmem.at[pl.ds(k * bk, bk), pl.ds(j * bn, bn)],
                b_smem.at[slot],
                oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)

        def tile(ij, nxt, last=False):
            """mma over all of k for tile `ij`, prefetching `stages` ahead.

            The last `stages` fetches of a tile belong to the next one, so
            compute lags a tile behind the loop index: `ij` is the tile
            being multiplied, `nxt` the one whose first slots we fill.
            """
            i, j = ij

            def k_step(k, acc):
                plgpu.wait_gmem_to_smem((stages - 1) * 2)
                slot = jax.lax.rem(k, stages)
                with jax.named_scope("mma"):
                    acc = plgpu.mma(acc, a_smem.at[slot][...],
                                    b_smem.at[slot][...])
                # Issued after the mma has read the slot, and refilling the
                # slot this step just freed keeps `stages` in flight -- the
                # count has to stay constant or the wait above is not exact.
                s = k + stages
                if last:
                    # Nothing follows, so re-read k=nk-1 into the dead slots.
                    fetch(i, j, jnp.minimum(s, nk - 1), slot)
                else:
                    head = s < nk
                    fetch(jnp.where(head, i, nxt[0]),
                          jnp.where(head, j, nxt[1]),
                          jnp.where(head, s, s - nk), slot)
                return acc

            acc = jax.lax.fori_loop(0, nk, k_step,
                                    jnp.zeros((bm, bn), jnp.float32))
            with jax.named_scope("storage"):
                o_gmem[pl.ds(i * bm, bm), pl.ds(j * bn, bn)] = acc.astype(DT)

        # nd_loop hands each CTA tiles p, p + num_sms, ... of the (ni, nj)
        # space; tiling=(mt, 1) walks mt rows of i before moving in j, the
        # same L2 locality planar_snake was giving us. The carry is the
        # previous iteration's index.
        @plgpu.nd_loop((ni, nj), collective_axes='sm', tiling=(mt, 1),
                       init_carry=(jnp.int32(0), jnp.int32(0)))
        def last_ij(loop_info, carry):
            ij = loop_info.index

            # Only the very first tile has to prime the pipeline itself.
            @pl.when(loop_info.local_index == 0)
            def _():
                for k in range(stages):
                    fetch(ij[0], ij[1], jnp.int32(k), k)

            @pl.when(loop_info.local_index > 0)
            def _():
                tile(carry, ij)

            return ij

        # The loop fetched one tile further than it computed.
        tile(last_ij, last_ij, last=True)

    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((M, N), DT),
        scratch_types=[plgpu.SMEM((stages, bm, bk), DT),
                       plgpu.SMEM((stages, bk, bn), DT)],
        grid=(num_sms,),
        grid_names=('sm',),
        kernel_name='matmul3',
    )(a, b)
```
