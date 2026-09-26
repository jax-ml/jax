```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
```

# Kernel 1: Basic Ampere Matmul with Swizzle

Stored naively, the columns of a row land in a handful of the 32 SMEM
banks, so the threads of an ldmatrix serialize. Tiling the tile into 8-row
sub-tiles and XOR-swizzling each line spreads them across all the banks.

The swizzle is a byte count, so it depends on the tile width and the dtype:
64 bf16 columns is exactly a 128-byte line. `find_swizzle` picks the
largest of 128/64/32/16 that divides the row, so BM/BN/BK can be changed
without hand-editing the transforms.

```python
M = K = N = 8192
DT = jnp.bfloat16

BITS = jnp.finfo(DT).bits

def swizzled(minor_dim):
    swizzle = plgpu.find_swizzle(minor_dim * BITS)
    return (plgpu.TilingTransform((8, 8 * swizzle // BITS)),
            plgpu.SwizzleTransform(swizzle))


def matmul1(a, b, bm=128, bn=128, bk=32, stages=4):
    def body(a_gmem, b_gmem, o_gmem):
        i, j = (jax.lax.axis_index(n) for n in 'ij')
        def step(indices, a_smem, b_smem, acc):
            return plgpu.mma(acc, a_smem[...], b_smem[...])

        # index_map returns *block* indices, not elements: block (i, k) of a
        # bm x bk-blocked A. i and j are closed over from the outer grid.
        # plgpu.BlockSpec (not pl.BlockSpec) is the one that takes transforms.
        acc = plgpu.emit_pipeline(
            step,
            grid=(K // bk,),
            in_specs=[
                plgpu.BlockSpec((bm, bk), lambda k: (i, k),
                                transforms=swizzled(bk)),
                plgpu.BlockSpec((bk, bn), lambda k: (k, j),
                                transforms=swizzled(bn)),
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
                plgpu.BlockSpec((bm, bk), lambda k: (i, k),
                                transforms=swizzled(bk)),
                plgpu.BlockSpec((bk, bn), lambda k: (k, j),
                                transforms=swizzled(bn)),
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

TODO: this is actually the slowest of all right now. 

```python
def matmul(a, b, bm=128, bn=128, bk=32, stages=4, mt=8):
    ni, nj, nk = M // bm, N // bn, K // bk
    tiles = ni * nj
    num_sms = min(backend.get_default_device().core_count, tiles)

    def body(a_gmem, b_gmem, o_gmem):
        p = jax.lax.axis_index('sm')
        # Tiles p, p + num_sms, ...: that is tiles // num_sms of them, plus one
        # for the first `tiles % num_sms` CTAs.
        my_tiles = tiles // num_sms + (p < tiles % num_sms)
        total = my_tiles * nk

        def ij(t):
            return plgpu.planar_snake(p + t * num_sms, (ni, nj),
                                      minor_dim=0, tile_width=mt)

        def scoped(a_smem, b_smem):
            def fetch(s):
                slot = jax.lax.rem(s, stages)
                s = jnp.minimum(s, total - 1)
                t, k = s // nk, jax.lax.rem(s, nk)
                i, j = ij(t)
                plgpu.copy_gmem_to_smem(
                    a_gmem.at[pl.ds(i * bm, bm), pl.ds(k * bk, bk)],
                    a_smem.at[slot],
                    oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)
                plgpu.copy_gmem_to_smem(
                    b_gmem.at[pl.ds(k * bk, bk), pl.ds(j * bn, bn)],
                    b_smem.at[slot],
                    oob_mode=OOBFillMode.PROMISE_IN_BOUNDS)

            for s in range(stages):
                fetch(jnp.int32(s))

            def tile(t, _):
                def k_step(k, acc):
                    s = t * nk + k
                    plgpu.wait_gmem_to_smem((stages - 1) * 2)
                    slot = jax.lax.rem(s, stages)
                    with jax.named_scope("mma"):
                        acc = plgpu.mma(acc, a_smem.at[slot][...],
                                        b_smem.at[slot][...])
                    fetch(s + stages)
                    return acc

                acc = jax.lax.fori_loop(0, nk, k_step,
                                        jnp.zeros((bm, bn), jnp.float32))
                i, j = ij(t)
                with jax.named_scope("storage"):
                    o_gmem[pl.ds(i * bm, bm), pl.ds(j * bn, bn)] = acc.astype(DT)

            jax.lax.fori_loop(0, my_tiles, tile, None)

        pl.run_scoped(
            scoped,
            a_smem=plgpu.SMEM((stages, bm, bk), DT, transforms=swizzled(bk)),
            b_smem=plgpu.SMEM((stages, bk, bn), DT, transforms=swizzled(bn)),
        )

    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((M, N), DT),
        grid=(num_sms,),
        grid_names=('sm',),
        kernel_name='matmul3',
    )(a, b)
```
